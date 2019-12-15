/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef JIT_GEN12HP_BF16_CONV_KERNEL_HPP
#define JIT_GEN12HP_BF16_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct jit_gen12hp_bf16_conv_bwd_weights_kernel {
    jit_gen12hp_bf16_conv_bwd_weights_kernel(const jit_conv_conf_t &ajcp)
        : jcp(ajcp) {}

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_t &src_md,
            const memory_desc_t &diff_weights_md,
            const memory_desc_t &diff_dst_md, const memory_desc_t &diff_bias_md,
            const primitive_attr_t &attr,
            memory_tracking::registrar_t &scratchpad) {

        set_default_conf(jcp, cd, src_md, diff_weights_md, diff_dst_md, attr);

        //TODO: move this to set_default conf
        const memory_desc_wrapper diff_bias_mdw(&diff_bias_md);
        jcp.bias_data_type = diff_bias_mdw.data_type();

        status_t status = status::success;

        // TODO: remove restrictions on oc and ic
        if (jcp.is_depthwise || jcp.mb % 32 != 0 || jcp.oc % 64 != 0
                || jcp.ic % 64 != 0)
            return status::unimplemented;
        jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;

        using namespace dnnl::impl::format_tag;

        jcp.src_tag = utils::pick(
                jcp.ndims - 3, NCw32n16c, NChw32n16c, NCdhw32n16c);
        jcp.dst_tag = utils::pick(
                jcp.ndims - 3, NCw32n16c, NChw32n16c, NCdhw32n16c);
        jcp.wei_tag = jcp.with_groups ? utils::pick(jcp.ndims - 3, gOIw16o16i,
                              gOIhw16o16i, gOIdhw16o16i)
                                      : utils::pick(jcp.ndims - 3, OIw16o16i,
                                              OIhw16o16i, OIdhw16o16i);

        jcp.sub_group_size = 8;
        jcp.mb_block = 16;
        jcp.oc_block = 16;
        jcp.ic_block = 16;
        // Each workgroup loads:
        // SRC: (mb_blk_unroll * mb_block) * (ic_blk_unroll * ic_block),
        // DIFF_DST: (mb_blk_unroll * mb_block) * (oc_blk_unroll * oc_block)
        // to compute and store WEI : (oc_blk_unroll * oc_block) * (ic_blk_unroll * ic_block).
        jcp.mb_blk_unroll = nstl::min(2, utils::div_up(jcp.mb, jcp.mb_block));
        jcp.oc_blk_unroll = ((jcp.oc / jcp.oc_block) % 8) == 0 ? 8 : 4;
        jcp.ic_blk_unroll = ((jcp.ic / jcp.ic_block) % 8) == 0 ? 8 : 4;

        const int num_buffers = 2;
        jcp.src_slm_size = num_buffers * jcp.mb_blk_unroll * jcp.mb_block
                * jcp.ic_block * jcp.ic_blk_unroll / 2;
        jcp.dst_slm_size = num_buffers * jcp.mb_blk_unroll * jcp.mb_block
                * jcp.oc_block * jcp.oc_blk_unroll / 2;

        // TODO: try workgroup size 32
        const int workgroup_size = 16; // 16 EUs/DSS
        // TODO: Fine-tune max_subgroups according to conv sizes
        //  1 tile with 4thr/EU can dispatch maximum 2048 subgroups,
        //  but 4096 seems to better for resnet_50 convolutions (when measured through gsim)
        const int max_subgroups = 4096;
        jcp.lws_d[0] = jcp.sub_group_size * 4;
        jcp.lws_d[1] = workgroup_size / 4;
        jcp.lws_d[2] = 1;

        jcp.gws_d[0] = jcp.ic
                / (jcp.ic_blk_unroll * jcp.ic_block
                        / jcp.lws_d[0]); // In best case, 4 ic/workitem
        jcp.gws_d[1] = jcp.oc
                / (jcp.oc_blk_unroll * jcp.oc_block
                        / jcp.lws_d[1]); // In best case, 32 oc/workitem

        // Parallelize along k-dimension to utilize all logical threads
        const int k_dim = (jcp.mb / (jcp.mb_blk_unroll * jcp.mb_block) * jcp.od
                * jcp.oh * jcp.ow);

        jcp.workgroups_along_k = utils::max_div(k_dim,
                utils::div_up(max_subgroups,
                        (jcp.gws_d[0] * jcp.gws_d[1] / jcp.sub_group_size)
                                * jcp.ngroups * jcp.kd * jcp.kh * jcp.kw));

        jcp.k_blocks = k_dim / jcp.workgroups_along_k;
        jcp.k_block_tail = k_dim % jcp.workgroups_along_k;

        jcp.gws_d[2] = jcp.ngroups * jcp.kd * jcp.kh * jcp.kw
                * jcp.workgroups_along_k;

        size_t wei_size = jcp.weights_data_type == data_type::bf16
                ? jcp.ngroups * jcp.oc * jcp.ic * jcp.kd * jcp.kh * jcp.kw
                        * sizeof(float)
                : 0;
        if (wei_size)
            scratchpad.book(
                    memory_tracking::names::key_conv_wei_reduction, wei_size);
        size_t bia_size
                = ((jcp.with_bias && jcp.bias_data_type == data_type::bf16)
                                  ? jcp.ngroups * jcp.oc
                                  : 0)
                * sizeof(float);
        if (bia_size)
            scratchpad.book(
                    memory_tracking::names::key_conv_bia_reduction, bia_size);

        return status;
    }

    static status_t init_const_def(compute::kernel_ctx_t &kernel_ctx,
            const jit_conv_conf_t &jcp, const jit_offsets &off) {
        kernel_ctx.define_int("NDIMS", jcp.ndims);
        kernel_ctx.define_int("WITH_GROUPS", jcp.with_groups);
        kernel_ctx.define_int("G", jcp.ngroups);
        kernel_ctx.define_int("MB", jcp.mb);
        kernel_ctx.define_int("IC", jcp.ic);
        kernel_ctx.define_int("ID", jcp.id);
        kernel_ctx.define_int("IH", jcp.ih);
        kernel_ctx.define_int("IW", jcp.iw);
        kernel_ctx.define_int("OC", jcp.oc);
        kernel_ctx.define_int("OD", jcp.od);
        kernel_ctx.define_int("OH", jcp.oh);
        kernel_ctx.define_int("OW", jcp.ow);
        kernel_ctx.define_int("KD", jcp.kd);
        kernel_ctx.define_int("KH", jcp.kh);
        kernel_ctx.define_int("KW", jcp.kw);
        kernel_ctx.define_int("SD", jcp.stride_d);
        kernel_ctx.define_int("SH", jcp.stride_h);
        kernel_ctx.define_int("SW", jcp.stride_w);
        kernel_ctx.define_int("PD", jcp.f_pad);
        kernel_ctx.define_int("PH", jcp.t_pad);
        kernel_ctx.define_int("PW", jcp.l_pad);
        kernel_ctx.define_int("PD_R", jcp.back_pad);
        kernel_ctx.define_int("PH_R", jcp.b_pad);
        kernel_ctx.define_int("PW_R", jcp.r_pad);
        kernel_ctx.define_int("DD", jcp.dilate_d);
        kernel_ctx.define_int("DH", jcp.dilate_h);
        kernel_ctx.define_int("DW", jcp.dilate_w);

        kernel_ctx.define_int("MB_BLOCK", jcp.mb_block);
        kernel_ctx.define_int("OC_BLOCK", jcp.oc_block);
        kernel_ctx.define_int("IC_BLOCK", jcp.ic_block);
        kernel_ctx.define_int("MB_BLK_UNROLL", jcp.mb_blk_unroll);
        kernel_ctx.define_int("IC_BLK_UNROLL", jcp.ic_blk_unroll);
        kernel_ctx.define_int("OC_BLK_UNROLL", jcp.oc_blk_unroll);
        kernel_ctx.define_int("K_WORKGROUPS", jcp.workgroups_along_k);
        kernel_ctx.define_int("K_BLOCKS", jcp.k_blocks);
        kernel_ctx.define_int("K_BLOCK_TAIL", jcp.k_block_tail);

        kernel_ctx.define_int("SRC_SLM_SIZE", jcp.src_slm_size);
        kernel_ctx.define_int("DST_SLM_SIZE", jcp.dst_slm_size);
        kernel_ctx.define_int("WITH_BIAS", jcp.with_bias);

        kernel_ctx.define_int("SUB_GROUP_SIZE", jcp.sub_group_size);
        kernel_ctx.define_int("LWS_0", jcp.lws_d[0]);
        kernel_ctx.define_int("LWS_1", jcp.lws_d[1]);
        kernel_ctx.define_int("LWS_2", jcp.lws_d[2]);
        kernel_ctx.define_int("GWS_0", jcp.gws_d[0]);
        kernel_ctx.define_int("GWS_1", jcp.gws_d[1]);
        kernel_ctx.define_int("GWS_2", jcp.gws_d[2]);

        def_offsets(off.src_off, kernel_ctx, "SRC", jcp.ndims);
        def_offsets(
                off.wht_off, kernel_ctx, "WHT", jcp.ndims + jcp.with_groups);
        def_offsets(off.dst_off, kernel_ctx, "DST", jcp.ndims);

        def_data_type(kernel_ctx, jcp.weights_data_type, "WEI");
        if (jcp.with_bias) def_data_type(kernel_ctx, jcp.bias_data_type, "BIA");
        kernel_ctx.set_data_type(
                data_type::bf16); // for enabling correct mmad8x8/dpas macro

        kernel_ctx.add_option("-cl-std=CL2.0");
        kernel_ctx.print_options();
        return status::success;
    }

    jit_conv_conf_t jcp;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif

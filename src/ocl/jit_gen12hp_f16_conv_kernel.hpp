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

#ifndef JIT_GEN12HP_F16_CONV_KERNEL_HPP
#define JIT_GEN12HP_F16_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct jit_gen12hp_f16_conv_fwd_kernel {
    jit_gen12hp_f16_conv_fwd_kernel(jit_conv_conf_t ajcp) : jcp(ajcp) {};

    ~jit_gen12hp_f16_conv_fwd_kernel() {};

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_t &src_md,
            const memory_desc_t &weights_md, const memory_desc_t &dst_md,
            const memory_desc_t &bias_md, const primitive_attr_t &attr) {

        set_default_conf(jcp, cd, src_md, weights_md, dst_md, attr);

        status_t status = status::success;

        if (jcp.is_depthwise) return status::unimplemented;

        if (jcp.mb < 8) return status::unimplemented;

        if (!jcp.is_depthwise && jcp.with_groups && jcp.ngroups > 1
                && (jcp.oc % 16 != 0 || jcp.ic % 16 != 0))
            return status::unimplemented;

        jcp.sub_group_size = 8;
        jcp.mb_block = 32;
        jcp.oc_block = 16;
        jcp.ic_block = 16;
        jcp.nchunk = utils::div_up(jcp.oc * jcp.ngroups, 32);
        int oc_group = nstl::min(jcp.nchunk, 2);

        jcp.lws_d[0] = 8 * oc_group;
        jcp.lws_d[1] = 8;
        jcp.lws_d[2] = 1;

        jcp.gws_d[0] = utils::rnd_up(jcp.nchunk * 8, jcp.lws_d[0]);
        jcp.gws_d[1] = jcp.od * jcp.oh * utils::rnd_up(jcp.ow, jcp.lws_d[1]);
        jcp.gws_d[2] = utils::div_up(jcp.mb, jcp.mb_block);

        jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
        format_tag_t src_tag, dst_tag, wei_tag;

        src_tag = utils::pick(jcp.ndims - 3, format_tag::NCw32n16c,
                format_tag::NChw32n16c, format_tag::NCdhw32n16c);
        dst_tag = utils::pick(jcp.ndims - 3, format_tag::NCw32n16c,
                format_tag::NChw32n16c, format_tag::NCdhw32n16c);
        wei_tag = jcp.with_groups
                ? utils::pick(jcp.ndims - 3, format_tag::gOIw2o2o8i8o2i,
                        format_tag::gOIhw2o2o8i8o2i,
                        format_tag::gOIdhw2o2o8i8o2i)
                : utils::pick(jcp.ndims - 3, format_tag::OIw2o2o8i8o2i,
                        format_tag::OIhw2o2o8i8o2i,
                        format_tag::OIdhw2o2o8i8o2i);
        jcp.src_tag = src_tag;
        jcp.wei_tag = wei_tag;
        jcp.dst_tag = dst_tag;

        jcp.wht_slm_size = jcp.kw * jcp.oc_block * jcp.ic_block
                * utils::div_up(jcp.lws_d[0], 8) * 2;

        return status;
    }

    static status_t init_const_def(
            compute::kernel_ctx_t &kernel_ctx, const jit_conv_conf_t &jcp) {
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
        kernel_ctx.define_int("DD", jcp.dilate_d);
        kernel_ctx.define_int("DH", jcp.dilate_h);
        kernel_ctx.define_int("DW", jcp.dilate_w);
        kernel_ctx.define_int("OW_PADDED", utils::rnd_up(jcp.ow, jcp.lws_d[1]));
        kernel_ctx.define_int("MB_BLOCK", jcp.mb_block);
        kernel_ctx.define_int("OC_BLOCK", jcp.oc_block);
        kernel_ctx.define_int("OC_CALC_BLOCK", 32);
        kernel_ctx.define_int(
                "WEI_BLOCK", 32 * 32 / types::data_type_size(data_type::f16));
        kernel_ctx.define_int("IC_BLOCK", jcp.ic_block);
        kernel_ctx.define_int("OC_GROUP", utils::div_up(jcp.lws_d[0], 8));
        kernel_ctx.define_int("MB_GROUP", 1);
        kernel_ctx.define_int("SP_GROUP", jcp.lws_d[1]);
        kernel_ctx.define_int("OC_NCHUNK", utils::div_up(jcp.oc, jcp.oc_block));
        kernel_ctx.define_int("IC_NCHUNK", utils::div_up(jcp.ic, jcp.ic_block));
        kernel_ctx.define_int("WITH_BIAS", jcp.with_bias);
        kernel_ctx.define_int("WITH_ELTWISE", jcp.with_eltwise);
        kernel_ctx.define_int("WITH_SUM", jcp.with_sum);
        kernel_ctx.define_int("SUM_SCALE", jcp.sum_scale == 1.0);
        kernel_ctx.define_int(
                "WITH_POST_SUM_ELTWISE", jcp.with_post_sum_eltwise);
        kernel_ctx.define_int("SUB_GROUP_SIZE", jcp.sub_group_size);
        kernel_ctx.define_int("LWS_0", jcp.lws_d[0]);
        kernel_ctx.define_int("LWS_1", jcp.lws_d[1]);
        kernel_ctx.define_int("LWS_2", jcp.lws_d[2]);
        kernel_ctx.define_int("SLM_WEI", jcp.wht_slm_size <= 8192);

        kernel_ctx.set_data_type(jcp.dst_data_type);

        if (jcp.with_eltwise || jcp.with_post_sum_eltwise) {
            def_postops(kernel_ctx, jcp.eltwise.alg);
        }

        return status::success;
    }

    jit_conv_conf_t jcp;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif

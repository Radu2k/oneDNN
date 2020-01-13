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

#ifndef JIT_GEN12HP_1x1_CONV_KERNEL_HPP
#define JIT_GEN12HP_1x1_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

using namespace dnnl::impl::format_tag;

static constexpr bool use_int8_slm_impl = false;
static constexpr bool use_xf16_src_slm_impl = false;
static constexpr bool use_xf16_wei_slm_impl = true;

struct jit_gen12hp_1x1_conv_fwd_kernel {
    jit_gen12hp_1x1_conv_fwd_kernel(const jit_conv_conf_t &ajcp) : jcp(ajcp) {}

    ~jit_gen12hp_1x1_conv_fwd_kernel() {}

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_t &src_md,
            const memory_desc_t &weights_md, const memory_desc_t &dst_md,
            const memory_desc_t &bias_md, const primitive_attr_t &attr) {

        const memory_desc_wrapper src_mdw(&src_md);
        const memory_desc_wrapper weights_mdw(&weights_md);
        const memory_desc_wrapper dst_mdw(&dst_md);
        const memory_desc_wrapper bias_mdw(&bias_md);

        set_default_conf(jcp, cd, src_md, weights_md, dst_md, attr);

        status_t status = status::success;

        bool is_bf16 = src_mdw.data_type() == data_type::bf16;
        bool is_fp16 = src_mdw.data_type() == data_type::f16;
        bool is_int8 = src_mdw.data_type() == data_type::u8;

        jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
        jcp.bias_data_type = bias_md.data_type;

        format_tag_t src_tag = dnnl_format_tag_undef;
        format_tag_t dst_tag = dnnl_format_tag_undef;
        format_tag_t wei_tag = dnnl_format_tag_undef;

        if (jcp.is_depthwise != false || (jcp.with_groups && jcp.ngroups > 1)
                || jcp.kh != 1 || jcp.kw != 1)
            return status::unimplemented;

        if (is_int8) {
            if (jcp.oc % 32 != 0 || jcp.ic % 32 != 0)
                return status::unimplemented;

            jcp.mb_block = 32;
            jcp.oc_block = 32;
            jcp.ic_block = 32;

            jcp.sub_group_size = 8;
            jcp.lws_d[0] = jcp.sub_group_size;
            if (use_int8_slm_impl)
                jcp.lws_d[1] = 8;
            else
                jcp.lws_d[1] = 1;
            jcp.lws_d[2] = 1;

            jcp.gws_d[0] = jcp.oc / jcp.oc_block * jcp.sub_group_size;
            if (use_int8_slm_impl)
                jcp.gws_d[1] = utils::rnd_up(jcp.ow, jcp.lws_d[1]) * jcp.oh;
            else
                jcp.gws_d[1] = jcp.ow * jcp.oh;
            jcp.gws_d[2] = utils::div_up(jcp.mb, jcp.mb_block);

            src_tag = utils::pick(jcp.ndims - 3, NCw32n32c, NChw32n32c);
            dst_tag = utils::pick(jcp.ndims - 3, NCw32n32c, NChw32n32c);
            wei_tag = jcp.with_groups
                    ? utils::pick(jcp.ndims - 3, gOIw4o8i8o4i, gOIhw4o8i8o4i)
                    : utils::pick(jcp.ndims - 3, OIw4o8i8o4i, OIhw4o8i8o4i);

        } else if (is_fp16 || is_bf16) {
            if (jcp.oc % 32 != 0 || jcp.ic % 16 != 0)
                return status::unimplemented;

            jcp.mb_block = 32;
            jcp.oc_block = 32;
            jcp.ic_block = 16;

            jcp.sub_group_size = 8;
            if (use_xf16_src_slm_impl
                    && ((jcp.oc / jcp.oc_block * jcp.sub_group_size) % 32 == 0))
                jcp.lws_d[0] = jcp.sub_group_size * 4;
            else
                jcp.lws_d[0] = jcp.sub_group_size;
            jcp.lws_d[1] = 8;
            jcp.lws_d[2] = 1;

            jcp.gws_d[0] = jcp.oc / jcp.oc_block * jcp.sub_group_size;
            jcp.gws_d[1] = utils::rnd_up(jcp.ow, jcp.lws_d[1]) * jcp.oh;
            jcp.gws_d[2] = utils::div_up(jcp.mb, jcp.mb_block);

            src_tag = utils::pick(jcp.ndims - 3, NCw32n16c, NChw32n16c);
            dst_tag = utils::pick(jcp.ndims - 3, NCw32n16c, NChw32n16c);
            wei_tag = jcp.with_groups
                    ? utils::pick(jcp.ndims - 3, format_tag::gOIw4o8i8o2i,
                            format_tag::gOIhw4o8i8o2i,
                            format_tag::gOIdhw4o8i8o2i)
                    : utils::pick(jcp.ndims - 3, format_tag::OIw4o8i8o2i,
                            format_tag::OIhw4o8i8o2i,
                            format_tag::OIdhw4o8i8o2i);
        } else {
            assert(!"not expected");
        }

        jcp.src_tag = src_tag;
        jcp.wei_tag = wei_tag;
        jcp.dst_tag = dst_tag;

        return status;
    };

    static status_t init_const_def(
            compute::kernel_ctx_t &kernel_ctx, const jit_conv_conf_t &jcp) {
        kernel_ctx.define_int("MB", jcp.mb);
        kernel_ctx.define_int("IC", jcp.ic_without_padding);
        kernel_ctx.define_int("IH", jcp.ih);
        kernel_ctx.define_int("IW", jcp.iw);
        kernel_ctx.define_int("OC", jcp.oc_without_padding);
        kernel_ctx.define_int("OH", jcp.oh);
        kernel_ctx.define_int("OW", jcp.ow);
        kernel_ctx.define_int("KH", jcp.kh);
        kernel_ctx.define_int("KW", jcp.kw);
        kernel_ctx.define_int("SH", jcp.stride_h);
        kernel_ctx.define_int("SW", jcp.stride_w);

        kernel_ctx.define_int("OW_PADDED", utils::rnd_up(jcp.ow, jcp.lws_d[1]));

        kernel_ctx.define_int("MB_BLOCK", jcp.mb_block);
        kernel_ctx.define_int("OC_BLOCK", jcp.oc_block);
        kernel_ctx.define_int("IC_BLOCK", jcp.ic_block);

        kernel_ctx.define_int("WITH_BIAS", jcp.with_bias);
        kernel_ctx.define_int("WITH_ELTWISE", jcp.with_eltwise);
        kernel_ctx.define_int("WITH_SUM", jcp.with_sum);
        kernel_ctx.define_int("SUM_SCALE", jcp.sum_scale == 1.0);
        kernel_ctx.define_int(
                "WITH_POST_SUM_ELTWISE", jcp.with_post_sum_eltwise);
        if (jcp.with_eltwise || jcp.with_post_sum_eltwise)
            def_postops(kernel_ctx, jcp.eltwise.alg);

        kernel_ctx.define_int("SUB_GROUP_SIZE", jcp.sub_group_size);

        kernel_ctx.define_int("LWS_0", jcp.lws_d[0]);
        kernel_ctx.define_int("LWS_1", jcp.lws_d[1]);
        kernel_ctx.define_int("LWS_2", jcp.lws_d[2]);

        kernel_ctx.define_int("OC_NCHUNK", jcp.oc / jcp.oc_block);
        kernel_ctx.define_int("IC_NCHUNK", jcp.ic / jcp.ic_block);

        if (jcp.src_data_type == data_type::u8)
            kernel_ctx.set_data_type(jcp.dst_data_type);
        else
            kernel_ctx.set_data_type(jcp.src_data_type);

        def_data_type(kernel_ctx, jcp.dst_data_type, "DST");
        def_data_type(kernel_ctx, jcp.dst_data_type, "BIA");

        if (use_int8_slm_impl) kernel_ctx.define_int("INT8_SLM", 1);
        if (use_xf16_src_slm_impl && (jcp.gws_d[0] % 32 == 0))
            kernel_ctx.define_int("XF16_SRC_SLM", 1);
        if (use_xf16_wei_slm_impl) kernel_ctx.define_int("XF16_WEI_SLM", 1);

        kernel_ctx.add_option("-Dcl_intel_subgroups_char");

        return status::success;
    }

    jit_conv_conf_t jcp;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif

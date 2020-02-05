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

#ifndef JIT_GEN12LP_X8S8S32X_CONV_KERNEL_HPP
#define JIT_GEN12LP_X8S8S32X_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

using namespace dnnl::impl::format_tag;

struct jit_gen12lp_x8s8s32x_conv_fwd_kernel {
    jit_gen12lp_x8s8s32x_conv_fwd_kernel(const jit_conv_conf_t &ajcp)
        : jcp(ajcp) {}

    ~jit_gen12lp_x8s8s32x_conv_fwd_kernel() {}

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

        if (!jcp.is_depthwise && jcp.with_groups && jcp.ngroups > 1
                && (jcp.oc % 32 != 0 || jcp.ic % 32 != 0))
            return status::unimplemented;

        jcp.dst_data_type = dst_mdw.data_type();
        jcp.src_data_type = src_mdw.data_type();

        jcp.oc_block = 32;
        if (jcp.is_depthwise) {
            jcp.sub_group_size = 16;
            jcp.mb_block = 32;
            jcp.ic_block = 32;
            jcp.ow_block = 1;

            jcp.lws_d[0] = 16;
            jcp.lws_d[1] = 1;
            jcp.lws_d[2] = 1;

            jcp.gws_d[0] = utils::div_up(jcp.ngroups, 32) * jcp.lws_d[0];
            jcp.gws_d[1] = jcp.od * jcp.oh * jcp.ow;
            jcp.gws_d[2] = utils::div_up(jcp.mb, jcp.mb_block / 4);
        } else {
            jcp.sub_group_size = 8;
            int ow_group = 1;
            int ow_nchunk;

            if (jcp.mb == 8 || jcp.mb % 16 == 0) {
                jcp.ver = ver_mb_block;
                jcp.mb_block = 32;
            } else {
                jcp.ver = ver_ow_block;
                jcp.mb_block = 1;
            }
            if (jcp.ic <= 4) jcp.ver = ver_1stconv;

            jcp.nchunk = utils::div_up(jcp.oc * jcp.ngroups, jcp.oc_block);

            int max_oc = 4;
            int oc_group = utils::max_div(
                    utils::div_up(jcp.oc, jcp.oc_block), max_oc);
            int max_subgroups = 32;
            int max_ow_group = max_subgroups / oc_group;
            switch (jcp.ver) {
                case ver_mb_block:
                    oc_group = 1;
                    jcp.ic_block = 32;
                    jcp.ow_block = 1;
                    ow_group = 1;
                    break;
                case ver_ow_block:
                    jcp.ic_block = 32;
                    jcp.ow_block
                            = (jcp.mb * jcp.oc * jcp.oh * jcp.ow < 49 * 1024)
                            ? 4
                            : 8;
                    ow_nchunk = utils::div_up(jcp.ow, jcp.ow_block);
                    ow_group = utils::max_div(ow_nchunk, max_ow_group);
                    if (ow_group == 1)
                        utils::max_div(ow_nchunk + 1, max_ow_group);
                    break;
                case ver_1stconv:
                    jcp.ic_block = 4;
                    jcp.ow_block = (jcp.kw * jcp.kh <= 49 && jcp.ow % 16 < 8)
                            ? 16
                            : 12;
                    ow_nchunk = utils::div_up(jcp.ow, jcp.ow_block);
                    ow_group = utils::max_div(ow_nchunk, max_ow_group);
                    if (ow_group == 1)
                        utils::max_div(ow_nchunk + 1, max_ow_group);
                    break;
            }

            jcp.lws_d[0] = 8 * oc_group;
            jcp.lws_d[1] = ow_group;
            jcp.lws_d[2] = 1;

            jcp.src_slm_size = jcp.ic_block / 4
                    * (jcp.lws_d[1] * jcp.stride_w * jcp.ow_block
                            + (jcp.kw - 1) * (1 + jcp.dilate_w));

            jcp.gws_d[0] = utils::rnd_up(jcp.nchunk * 8, jcp.lws_d[0]);
            jcp.gws_d[1] = jcp.od * jcp.oh
                    * utils::rnd_up(
                            utils::div_up(jcp.ow, jcp.ow_block), jcp.lws_d[1]);
            jcp.gws_d[2]
                    = utils::div_up(jcp.mb, utils::div_up(jcp.mb_block, 2));
            if (jcp.ver == ver_1stconv) {
                jcp.gws_d[2] = jcp.mb;
                // Save opportunity to use this implementation with nchw formats,
                // which will result in worse performance, but prevent us using reorder.
                // That can be efficient in some cases.
                jcp.is_nchw = src_mdw.matches_one_of_tag(ncw, nchw, ncdhw);
                // decrease src ic_block in case of input nchw
                if (jcp.is_nchw) jcp.ic_block = 1;
            }
        }

        jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

        format_tag_t src_tag, dst_tag, wei_tag;
        if (jcp.mb_block == 32) {
            src_tag = utils::pick(
                    jcp.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
            dst_tag = utils::pick(
                    jcp.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
        } else {
            src_tag = utils::pick(jcp.ndims - 3, nCw32c, nChw32c, nCdhw32c);
            dst_tag = utils::pick(jcp.ndims - 3, nCw32c, nChw32c, nCdhw32c);
        }

        if (!jcp.is_depthwise && jcp.ver == ver_1stconv) {
            src_tag = (jcp.is_nchw)
                    ? utils::pick(jcp.ndims - 3, ncw, nchw, ncdhw)
                    : utils::pick(jcp.ndims - 3, nCw4c, nChw4c, nCdhw4c);
        }
        if (jcp.is_depthwise) {
            wei_tag = utils::pick(jcp.ndims - 3, Goiw32g, Goihw32g, Goidhw32g);
        } else {
            if (jcp.ver == ver_1stconv) {
                wei_tag = jcp.with_groups ? utils::pick(jcp.ndims - 3, gOIw8o4i,
                                  gOIhw8o4i, gOIdhw8o4i)
                                          : utils::pick(jcp.ndims - 3, OIw8o4i,
                                                  OIhw8o4i, OIdhw8o4i);
            } else {
                wei_tag = jcp.with_groups
                        ? utils::pick(jcp.ndims - 3, gOIw4o8i8o4i,
                                gOIhw4o8i8o4i, gOIdhw4o8i8o4i)
                        : utils::pick(jcp.ndims - 3, OIw4o8i8o4i, OIhw4o8i8o4i,
                                OIdhw4o8i8o4i);
            }
        }
        jcp.src_tag = src_tag;
        jcp.wei_tag = wei_tag;
        jcp.dst_tag = dst_tag;

        return status;
    }

    static status_t init_const_def(
            compute::kernel_ctx_t &kernel_ctx, const jit_conv_conf_t &jcp) {
        kernel_ctx.define_int("NCHW", jcp.is_nchw);
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

        kernel_ctx.define_int("OW_PADDED",
                utils::rnd_up(
                        utils::div_up(jcp.ow, jcp.ow_block), jcp.lws_d[1]));

        kernel_ctx.define_int("MB_BLOCK", jcp.mb_block);
        kernel_ctx.define_int("OC_BLOCK", jcp.oc_block);
        kernel_ctx.define_int("IC_BLOCK", jcp.ic_block);
        kernel_ctx.define_int("OW_BLOCK", jcp.ow_block);

        kernel_ctx.define_int("OC_GROUP", utils::div_up(jcp.lws_d[0], 8));
        kernel_ctx.define_int("MB_GROUP", 1);
        kernel_ctx.define_int("SP_GROUP", jcp.lws_d[1]);
        kernel_ctx.define_int("OW_NCHUNK", utils::div_up(jcp.ow, jcp.ow_block));
        kernel_ctx.define_int("OC_NCHUNK", utils::div_up(jcp.oc, jcp.oc_block));
        kernel_ctx.define_int("IC_NCHUNK", utils::div_up(jcp.ic, jcp.ic_block));

        kernel_ctx.define_int("SLM_WORKING_GROUPS",
                nstl::min(utils::div_up(jcp.ow, jcp.ow_block),
                        utils::div_up(jcp.iw, jcp.ow_block * jcp.stride_w)));

        kernel_ctx.define_int("OW_TAIL", jcp.ow % jcp.ow_block);
        kernel_ctx.define_int(
                "IW_TAIL", abs(jcp.kw - 1) * (1 + jcp.dilate_w) - jcp.stride_w);
        kernel_ctx.define_int("OW_SLM_TAIL",
                jcp.iw
                        - jcp.stride_w * jcp.ow_block
                                * (nstl::min(
                                           utils::div_up(jcp.ow, jcp.ow_block),
                                           utils::div_up(jcp.iw,
                                                   jcp.ow_block * jcp.stride_w))
                                        - 1));
        kernel_ctx.define_int("ZERO_TAIL",
                utils::rnd_up(jcp.ow, jcp.ow_block) * jcp.stride_w - jcp.iw
                        + (jcp.kw - 1) * (1 + jcp.dilate_w) - jcp.l_pad);

        kernel_ctx.define_int("SRC_SLM_SIZE", jcp.src_slm_size);

        kernel_ctx.define_int("WITH_BIAS", jcp.with_bias);
        kernel_ctx.define_int("WITH_ELTWISE", jcp.with_eltwise);
        kernel_ctx.define_int("WITH_SUM", jcp.with_sum);
        kernel_ctx.define_int("SUM_SCALE", jcp.sum_scale == 1.0);
        kernel_ctx.define_int(
                "WITH_POST_SUM_ELTWISE", jcp.with_post_sum_eltwise);

        if (jcp.with_eltwise || jcp.with_post_sum_eltwise) {
            def_postops(kernel_ctx, jcp.eltwise.alg);
        }

        kernel_ctx.define_int("SUB_GROUP_SIZE", jcp.sub_group_size);
        kernel_ctx.define_int("LWS_0", jcp.lws_d[0]);
        kernel_ctx.define_int("LWS_1", jcp.lws_d[1]);
        kernel_ctx.define_int("LWS_2", jcp.lws_d[2]);

        kernel_ctx.set_data_type(jcp.src_data_type);
        def_data_type(kernel_ctx, jcp.src_data_type, "SRC");
        def_data_type(kernel_ctx, jcp.dst_data_type, "DST");
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");
        return status::success;
    }

    jit_conv_conf_t jcp;
};

struct jit_gen12lp_x8s8s32x_conv_bwd_data_kernel {
    jit_gen12lp_x8s8s32x_conv_bwd_data_kernel(const jit_conv_conf_t &ajcp)
        : jcp(ajcp) {}

    ~jit_gen12lp_x8s8s32x_conv_bwd_data_kernel() {}

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

        if (jcp.mb < 8) return status::unimplemented;

        if (jcp.with_groups && jcp.ngroups > 1
                && (jcp.oc % 32 != 0 || jcp.ic % 32 != 0))
            return status::unimplemented;

        jcp.dst_data_type = dst_mdw.data_type();
        jcp.src_data_type = src_mdw.data_type();

        jcp.sub_group_size = 8;
        jcp.mb_block = 32;
        jcp.oc_block = 32;
        jcp.ic_block = 32;
        jcp.nchunk = utils::div_up(jcp.ic * jcp.ngroups, jcp.ic_block);
        int ic_group = nstl::min(jcp.nchunk, 2);

        jcp.lws_d[0] = 8 * ic_group;
        jcp.lws_d[1] = 8;
        jcp.lws_d[2] = 1;

        jcp.gws_d[0] = utils::rnd_up(jcp.nchunk * 8, jcp.lws_d[0]);
        jcp.gws_d[1] = jcp.id * jcp.ih * utils::rnd_up(jcp.iw, jcp.lws_d[1]);
        jcp.gws_d[2] = utils::div_up(jcp.mb, jcp.mb_block);

        jcp.gws_d[2] = utils::div_up(jcp.mb, jcp.mb_block / 2);

        jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

        format_tag_t src_tag, dst_tag, wei_tag;

        src_tag = utils::pick(
                jcp.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
        dst_tag = utils::pick(
                jcp.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
        wei_tag = jcp.with_groups ? utils::pick(jcp.ndims - 3, gIOw4i8o8i4o,
                          gIOhw4i8o8i4o, gIOdhw4i8o8i4o)
                                  : utils::pick(jcp.ndims - 3, IOw4i8o8i4o,
                                          IOhw4i8o8i4o, IOdhw4i8o8i4o);

        jcp.src_tag = src_tag;
        jcp.wei_tag = wei_tag;
        jcp.dst_tag = dst_tag;

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

        kernel_ctx.define_int("IW_PADDED", utils::rnd_up(jcp.iw, jcp.lws_d[1]));

        kernel_ctx.define_int("MB_BLOCK", jcp.mb_block);
        kernel_ctx.define_int("OC_BLOCK", jcp.oc_block);
        kernel_ctx.define_int("IC_BLOCK", jcp.ic_block);

        kernel_ctx.define_int("IC_GROUP", utils::div_up(jcp.lws_d[0], 8));
        kernel_ctx.define_int("MB_GROUP", 1);
        kernel_ctx.define_int("SP_GROUP", jcp.lws_d[1]);

        kernel_ctx.define_int("OC_NCHUNK", utils::div_up(jcp.oc, jcp.oc_block));
        kernel_ctx.define_int("IC_NCHUNK", utils::div_up(jcp.ic, jcp.ic_block));

        kernel_ctx.define_int("WITH_BIAS", jcp.with_bias);

        kernel_ctx.define_int("SUB_GROUP_SIZE", jcp.sub_group_size);
        kernel_ctx.define_int("LWS_0", jcp.lws_d[0]);
        kernel_ctx.define_int("LWS_1", jcp.lws_d[1]);
        kernel_ctx.define_int("LWS_2", jcp.lws_d[2]);

        kernel_ctx.set_data_type(jcp.dst_data_type);
        def_data_type(kernel_ctx, jcp.src_data_type, "SRC");
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");

        return status::success;
    }

    jit_conv_conf_t jcp;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif

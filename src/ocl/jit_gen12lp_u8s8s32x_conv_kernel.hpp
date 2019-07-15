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

#ifndef JIT_GEN12LP_U8S8S32X_CONV_KERNEL_HPP
#define JIT_GEN12LP_U8S8S32X_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

using namespace mkldnn::impl::format_tag;

struct jit_gen12lp_u8s8s32x_conv_fwd_kernel {
    jit_gen12lp_u8s8s32x_conv_fwd_kernel(jit_conv_conf_t ajcp) : jcp(ajcp) {};

    ~jit_gen12lp_u8s8s32x_conv_fwd_kernel() {};

    static status_t init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_t &src_md,
        const memory_desc_t &weights_md, const memory_desc_t &dst_md,
        const memory_desc_t &bias_md, const primitive_attr_t &attr) {

        set_default_conf(jcp, cd, src_md, weights_md, dst_md, attr);

        status_t status = status::success;

        if (jcp.mb < 8)
            return status::unimplemented;

        if (!jcp.is_depthwise
            && jcp.with_groups && jcp.ngroups > 1
            && (jcp.oc % 32 != 0 || jcp.ic % 32 != 0))
            return status::unimplemented;



        jcp.sub_group_size = (jcp.is_depthwise) ? 16 : 8;
        jcp.mb_block = 32;
        jcp.oc_block = 32;
        jcp.ic_block = 32;
        jcp.nchunk = utils::div_up(jcp.oc * jcp.ngroups, jcp.oc_block);
        int oc_group = nstl::min(jcp.nchunk, 2);

        bool divide_mbblock = true;
        if (jcp.is_depthwise) {
            jcp.lws_d[0] = 16;
            jcp.lws_d[1] = 1;
            jcp.lws_d[2] = 1;

            jcp.gws_d[0] = utils::div_up(jcp.ngroups, 32) * jcp.lws_d[0];
            jcp.gws_d[1] = jcp.od * jcp.oh * jcp.ow;
            jcp.gws_d[2] = utils::div_up(jcp.mb, jcp.mb_block / 4);
        }
        else {
            jcp.lws_d[0] = 8 * oc_group;
            jcp.lws_d[1] = 8;
            jcp.lws_d[2] = 1;

            jcp.gws_d[0] = utils::rnd_up(jcp.nchunk * 8, jcp.lws_d[0]);
            jcp.gws_d[1] = jcp.od * jcp.oh * utils::rnd_up(jcp.ow, jcp.lws_d[1]);
            jcp.gws_d[2] = utils::div_up(jcp.mb, jcp.mb_block);
            if (divide_mbblock)
                jcp.gws_d[2] = utils::div_up(jcp.mb, jcp.mb_block / 2);
        }

        jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

        format_tag_t src_tag, dst_tag, wei_tag;

        src_tag = utils::pick(jcp.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
        dst_tag = utils::pick(jcp.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
        if (jcp.is_depthwise) {
            wei_tag = utils::pick(jcp.ndims - 3, Goiw32g, Goihw32g, Goidhw32g);
        } else {
        wei_tag = jcp.with_groups
            ? utils::pick(jcp.ndims - 3, gOIw4o8i8o4i, gOIhw4o8i8o4i, gOIdhw4o8i8o4i)
            : utils::pick(jcp.ndims - 3, OIw4o8i8o4i, OIhw4o8i8o4i, OIdhw4o8i8o4i);
        }

        jcp.src_tag = src_tag;
        jcp.wei_tag = wei_tag;
        jcp.dst_tag = dst_tag;

        return status;
    }

    static status_t init_const_def(ocl_jit_t &jit, const jit_conv_conf_t &jcp) {
        jit.define_int("G", jcp.ngroups);
        jit.define_int("MB", jcp.mb);
        jit.define_int("IC", jcp.ic);
        jit.define_int("ID", jcp.id);
        jit.define_int("IH", jcp.ih);
        jit.define_int("IW", jcp.iw);
        jit.define_int("OC", jcp.oc);
        jit.define_int("OD", jcp.od);
        jit.define_int("OH", jcp.oh);
        jit.define_int("OW", jcp.ow);
        jit.define_int("KD", jcp.kd);
        jit.define_int("KH", jcp.kh);
        jit.define_int("KW", jcp.kw);
        jit.define_int("SD", jcp.stride_d);
        jit.define_int("SH", jcp.stride_h);
        jit.define_int("SW", jcp.stride_w);
        jit.define_int("PD", jcp.f_pad);
        jit.define_int("PH", jcp.t_pad);
        jit.define_int("PW", jcp.l_pad);
        jit.define_int("DD", jcp.dilate_d);
        jit.define_int("DH", jcp.dilate_h);
        jit.define_int("DW", jcp.dilate_w);

        jit.define_int("OW_PADDED", utils::rnd_up(jcp.ow, jcp.lws_d[1]));

        jit.define_int("MB_BLOCK", jcp.mb_block);
        jit.define_int("OC_BLOCK", jcp.oc_block);
        jit.define_int("IC_BLOCK", jcp.ic_block);

        jit.define_int("OC_GROUP", utils::div_up(jcp.lws_d[0], 8));
        jit.define_int("MB_GROUP", 1);
        jit.define_int("SP_GROUP", jcp.lws_d[1]);


        jit.define_int("OC_NCHUNK", utils::div_up(jcp.oc, jcp.oc_block));
        jit.define_int("IC_NCHUNK", utils::div_up(jcp.ic, jcp.ic_block));

        jit.define_int("WITH_BIAS", jcp.with_bias);
        jit.define_int("WITH_RELU", jcp.with_relu);
        jit.define_int("WITH_SUM", jcp.with_sum);
        jit.define_int("WITH_SUM_ELTWISE", jcp.with_sum_eltwise);
        jit.define_int("SUM_SCALE", jcp.sum_scale == 1.0);

        jit.define_int("SUB_GROUP_SIZE", jcp.sub_group_size);
        jit.define_int("LWS_0", jcp.lws_d[0]);
        jit.define_int("LWS_1", jcp.lws_d[1]);
        jit.define_int("LWS_2", jcp.lws_d[2]);

        jit.set_data_type(jcp.dst_data_type);

        if (jcp.is_depthwise) {
            jit.add_option("-Dcl_intel_subgroups_char");
        }

        return status::success;
    }

    jit_conv_conf_t jcp;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif
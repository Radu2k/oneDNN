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

#include "gpu/ocl/gen12lp_x8s8s32x_1x1_convolution.hpp"

#include "gpu/ocl/ocl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t gen12lp_x8s8s32x_1x1_convolution_fwd_t::pd_t::init_conf() {
    using namespace format_tag;

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const memory_desc_wrapper bias_mdw(weights_md(1));

    set_default_conf(conf, cd, *src_md(), *weights_md(), *dst_md(),
            *weights_md(1), *attr());

    status_t status = status::success;

    if (conf.is_depthwise != false || (conf.with_groups && conf.ngroups > 1)
            || conf.kh != 1 || conf.kw != 1)
        return status::unimplemented;

    if (conf.oc % 32 != 0 || conf.ic % 32 != 0) return status::unimplemented;

    if (!(conf.mb == 8 || conf.mb % 16 == 0)) return status::unimplemented;
    conf.src_data_type = src_mdw.data_type();
    conf.dst_data_type = dst_mdw.data_type();

    conf.mb_block = 32;
    conf.oc_block = 32;
    conf.ic_block = 32;
    int ow_group = (conf.ow % 8) ? 1 : 8;

    conf.sub_group_size = 8;
    conf.lws_d[0] = conf.sub_group_size;
    conf.lws_d[1] = ow_group;
    conf.lws_d[2] = 1;

    conf.gws_d[0] = conf.oc / conf.oc_block * conf.sub_group_size;
    conf.gws_d[1] = utils::rnd_up(conf.ow, conf.lws_d[1]) * conf.oh;
    conf.gws_d[2] = utils::div_up(conf.mb, utils::div_up(conf.mb_block, 2));

    conf.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    format_tag_t src_tag, dst_tag, wei_tag;

    src_tag = utils::pick(conf.ndims - 3, NCw32n32c, NChw32n32c);
    dst_tag = utils::pick(conf.ndims - 3, NCw32n32c, NChw32n32c);
    wei_tag = conf.with_groups
            ? utils::pick(conf.ndims - 3, gOIw4o8i8o4i, gOIhw4o8i8o4i)
            : utils::pick(conf.ndims - 3, OIw4o8i8o4i, OIhw4o8i8o4i);

    conf.src_tag = src_tag;
    conf.wei_tag = wei_tag;
    conf.dst_tag = dst_tag;

    return status;
}

status_t gen12lp_x8s8s32x_1x1_convolution_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic_without_padding);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OC", conf.oc_without_padding);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);
    kernel_ctx.define_int("KH", conf.kh);
    kernel_ctx.define_int("KW", conf.kw);
    kernel_ctx.define_int("SH", conf.stride_h);
    kernel_ctx.define_int("SW", conf.stride_w);

    kernel_ctx.define_int("OW_PADDED", utils::rnd_up(conf.ow, conf.lws_d[1]));

    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);

    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    kernel_ctx.define_int("WITH_ELTWISE", conf.with_eltwise);
    kernel_ctx.define_int("WITH_SUM", conf.with_sum);
    kernel_ctx.define_int("SUM_SCALE", conf.sum_scale == 1.0);
    kernel_ctx.define_int("WITH_POST_SUM_ELTWISE", conf.with_post_sum_eltwise);
    if (conf.with_eltwise || conf.with_post_sum_eltwise)
        def_postops(kernel_ctx, conf.eltwise.alg);

    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.define_int("OC_NCHUNK", conf.oc / conf.oc_block);
    kernel_ctx.define_int("IC_NCHUNK", conf.ic / conf.ic_block);

    kernel_ctx.define_int("INT8_WEI_SLM", conf.ow % 8 == 0);

    kernel_ctx.set_data_type(conf.dst_data_type);
    def_data_type(kernel_ctx, conf.src_data_type, "SRC");

    kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    return status::success;
}

status_t gen12lp_x8s8s32x_1x1_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);
    arg_list.set(4, conf.eltwise.alpha);
    arg_list.set(5, conf.eltwise.beta);
    arg_list.set(6, conf.sum_scale);

    float scales = pd()->attr()->output_scales_.scales_[0];
    arg_list.set(7, scales);

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

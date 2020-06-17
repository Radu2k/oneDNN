/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "gpu/ocl/gen12hp_1x1_convolution.hpp"

#include "gpu/ocl/ocl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::format_tag;

static constexpr bool use_int8_slm_impl = false;
static constexpr bool use_xf16_src_slm_impl = false;
static constexpr bool use_xf16_wei_slm_impl = true;

status_t gen12hp_1x1_convolution_fwd_t::pd_t::init_conf() {
    const convolution_desc_t &cd = *desc();

    set_default_conf(conf, cd, *src_md(), *weights_md(), *dst_md(),
            *weights_md(1), *attr());

    status_t status = status::success;

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    bool is_bf16 = src_mdw.data_type() == data_type::bf16;
    bool is_fp16 = src_mdw.data_type() == data_type::f16;
    bool is_int8 = src_mdw.data_type() == data_type::u8;

    format_tag_t src_tag = dnnl_format_tag_undef;
    format_tag_t dst_tag = dnnl_format_tag_undef;
    format_tag_t wei_tag = dnnl_format_tag_undef;

    if (conf.is_depthwise != false || (conf.with_groups && conf.ngroups > 1)
            || conf.kh != 1 || conf.kw != 1)
        return status::unimplemented;

    if (is_int8) {
        if (conf.oc % 32 != 0 || conf.ic % 32 != 0)
            return status::unimplemented;

        conf.mb_block = 32;
        conf.oc_block = 32;
        conf.ic_block = 32;

        conf.sub_group_size = 8;
        conf.lws_d[0] = conf.sub_group_size;
        if (use_int8_slm_impl)
            conf.lws_d[1] = 8;
        else
            conf.lws_d[1] = 1;
        conf.lws_d[2] = 1;

        conf.gws_d[0] = conf.oc / conf.oc_block * conf.sub_group_size;
        if (use_int8_slm_impl)
            conf.gws_d[1] = utils::rnd_up(conf.ow, conf.lws_d[1]) * conf.oh;
        else
            conf.gws_d[1] = conf.ow * conf.oh;
        conf.gws_d[2] = utils::div_up(conf.mb, conf.mb_block);

        src_tag = utils::pick(conf.ndims - 3, NCw32n32c, NChw32n32c);
        dst_tag = utils::pick(conf.ndims - 3, NCw32n32c, NChw32n32c);
        wei_tag = conf.with_groups
                ? utils::pick(conf.ndims - 3, gOIw4o8i8o4i, gOIhw4o8i8o4i)
                : utils::pick(conf.ndims - 3, OIw4o8i8o4i, OIhw4o8i8o4i);

    } else if (is_fp16 || is_bf16) {
        if (conf.oc % 32 != 0 || conf.ic % 16 != 0)
            return status::unimplemented;

        conf.mb_block = 32;
        conf.oc_block = 32;
        conf.ic_block = 16;

        conf.sub_group_size = 8;
        if (use_xf16_src_slm_impl
                && ((conf.oc / conf.oc_block * conf.sub_group_size) % 32 == 0))
            conf.lws_d[0] = conf.sub_group_size * 4;
        else
            conf.lws_d[0] = conf.sub_group_size;
        conf.lws_d[1] = 8;
        conf.lws_d[2] = 1;

        conf.gws_d[0] = conf.oc / conf.oc_block * conf.sub_group_size;
        conf.gws_d[1] = utils::rnd_up(conf.ow, conf.lws_d[1]) * conf.oh;
        conf.gws_d[2] = utils::div_up(conf.mb, conf.mb_block);

        src_tag = utils::pick(conf.ndims - 3, NCw32n16c, NChw32n16c);
        dst_tag = utils::pick(conf.ndims - 3, NCw32n16c, NChw32n16c);
        wei_tag = conf.with_groups
                ? utils::pick(conf.ndims - 3, format_tag::gOIw4o8i8o2i,
                        format_tag::gOIhw4o8i8o2i, format_tag::gOIdhw4o8i8o2i)
                : utils::pick(conf.ndims - 3, format_tag::OIw4o8i8o2i,
                        format_tag::OIhw4o8i8o2i, format_tag::OIdhw4o8i8o2i);
    } else {
        assert(!"not expected");
    }

    conf.src_tag = src_mdw.format_kind() == format_kind::any
            ? src_tag
            : src_mdw.matches_one_of_tag(src_tag);
    conf.wei_tag = weights_mdw.format_kind() == format_kind::any
            ? wei_tag
            : weights_mdw.matches_one_of_tag(wei_tag);
    conf.dst_tag = dst_mdw.format_kind() == format_kind::any
            ? dst_tag
            : dst_mdw.matches_one_of_tag(dst_tag);

    if (conf.src_tag != src_tag || conf.wei_tag != wei_tag
            || conf.dst_tag != dst_tag)
        return status::unimplemented;

    return status;
}

status_t gen12hp_1x1_convolution_fwd_t::pd_t::init_kernel_ctx(
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
    def_attr_info(kernel_ctx, conf.attr_info);

    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.define_int("OC_NCHUNK", conf.oc / conf.oc_block);
    kernel_ctx.define_int("IC_NCHUNK", conf.ic / conf.ic_block);

    if (conf.src_data_type == data_type::u8)
        kernel_ctx.set_data_type(conf.dst_data_type);
    else
        kernel_ctx.set_data_type(conf.src_data_type);

    def_data_type(kernel_ctx, conf.dst_data_type, "DST");
    def_data_type(kernel_ctx, conf.dst_data_type, "BIA");

    if (use_int8_slm_impl) kernel_ctx.define_int("INT8_SLM", 1);
    if (use_xf16_src_slm_impl && (conf.gws_d[0] % 32 == 0))
        kernel_ctx.define_int("XF16_SRC_SLM", 1);
    if (use_xf16_wei_slm_impl) kernel_ctx.define_int("XF16_WEI_SLM", 1);

    kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    if (is_gen12hp) kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");

    return status::success;
}

status_t gen12hp_1x1_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);

    unsigned arg_idx = append_post_ops_to_arg_list(
            ctx, arg_list, 4, conf.attr_info.all_post_ops);

    if (conf.src_data_type == data_type::u8) {
        float scales = pd()->attr()->output_scales_.scales_[0];
        arg_list.set(arg_idx, scales);
    }

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gpu/jit/gen12hp_convolution.hpp"
#include "gpu/jit/gen12hp_conv_fwd_kernel.hpp"
#include "gpu/ocl/ocl_gpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;

status_t gen12hp_convolution_fwd_t::pd_t::init_conf(engine_t *engine) {
    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper wei_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    set_default_conf(conf, cd, *src_md(), *weights_md(), *dst_md(),
            *weights_md(1), *attr());

    if (conf.with_groups && conf.ngroups > 1) return status::unimplemented;
    if (conf.ic < 32) return status::unimplemented;
    if (conf.mb < 16) return status::unimplemented;

    // Reduce dimensions for 1x1 kernel.
    bool is_1x1 = (conf.kd * conf.kh * conf.kw == 1);
    bool is_stride1
            = (conf.stride_d == 1 && conf.stride_h == 1 && conf.stride_w == 1);
    bool is_eq_oi
            = (conf.od == conf.id && conf.oh == conf.ih && conf.ow == conf.iw);
    if (is_1x1 && is_stride1 && is_eq_oi) {
        assert(conf.f_pad == 0 && conf.t_pad == 0 && conf.l_pad == 0);
        conf.ow = conf.od * conf.oh * conf.ow;
        conf.iw = conf.id * conf.ih * conf.iw;
        conf.od = conf.id = conf.kd = 1;
        conf.oh = conf.ih = conf.kh = 1;
    }

    conf.mb_block = 32;
    conf.oc_block = 32;
    conf.ic_block = 32;

    bool enable_40n = (getenv_int("DNNL_ENABLE_CONV_40N", 0) != 0);
    conf.mb_block = (enable_40n ? 40 : 32);

    conf.oc_group = (conf.oc <= 64 ? 2 : 4);
    conf.ow_group = 4;

    conf.sub_group_size = 8;
    conf.gws_d[0] = utils::rnd_up(conf.oc, conf.oc_group * conf.oc_block)
            / conf.oc_block * 8;
    conf.gws_d[1] = conf.od * conf.oh * utils::rnd_up(conf.ow, conf.ow_group);
    conf.gws_d[2] = utils::div_up(conf.mb, conf.mb_block);
    conf.lws_d[0] = 8 * conf.oc_group;
    conf.lws_d[1] = conf.ow_group;
    conf.lws_d[2] = 1;

    bool is_int8
            = utils::one_of(conf.src_data_type, data_type::s8, data_type::u8);

    format_tag_t src_tag;
    format_tag_t wei_tag;
    format_tag_t dst_tag;

    if (is_int8) {
        if (conf.mb_block == 40) {
            src_tag = utils::pick(
                    conf.ndims - 3, NCw40n32c, NChw40n32c, NCdhw40n32c);
        } else {
            src_tag = utils::pick(
                    conf.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
        }
        wei_tag = conf.with_groups ? utils::pick(conf.ndims - 3, gOIw4o8i8o4i,
                          gOIhw4o8i8o4i, gOIdhw4o8i8o4i)
                                   : utils::pick(conf.ndims - 3, OIw4o8i8o4i,
                                           OIhw4o8i8o4i, OIdhw4o8i8o4i);
        dst_tag = src_tag;
    } else { // f16 or bf16.
        if (conf.mb_block == 40) {
            src_tag = utils::pick(
                    conf.ndims - 3, NCw40n16c, NChw40n16c, NCdhw40n16c);
        } else {
            src_tag = utils::pick(
                    conf.ndims - 3, NCw32n16c, NChw32n16c, NCdhw32n16c);
        }
        wei_tag = conf.with_groups ? utils::pick(conf.ndims - 3, gOIw4o8i8o2i,
                          gOIhw4o8i8o2i, gOIdhw4o8i8o2i)
                                   : utils::pick(conf.ndims - 3, OIw4o8i8o2i,
                                           OIhw4o8i8o2i, OIdhw4o8i8o2i);
        dst_tag = src_tag;
    }

    if (src_mdw.format_kind() != format_kind::any
            && src_mdw.format_kind() != src_tag)
        return status::unimplemented;

    if (wei_mdw.format_kind() != format_kind::any
            && wei_mdw.format_kind() != wei_tag)
        return status::unimplemented;

    if (dst_mdw.format_kind() != format_kind::any
            && dst_mdw.format_kind() != dst_tag)
        return status::unimplemented;

    conf.src_tag = src_tag;
    conf.wei_tag = wei_tag;
    conf.dst_tag = dst_tag;

    return status::success;
}

status_t gen12hp_convolution_fwd_t::init(engine_t *engine) {
    CHECK(gen12hp_conv_fwd_create_kernel(pd()->conf, &kernel_, this, engine));
    return status::success;
}

status_t gen12hp_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &wei = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bia = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;
    const auto &attr_info = conf.attr_info;

    auto &oscales
            = (attr_info.with_per_oc_oscales && !attr_info.with_runtime_oscales)
            ? CTX_GPU_RES_STORAGE(OSCALES_)
            : CTX_IN_STORAGE(DNNL_ARG_ATTR_OUTPUT_SCALES);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, wei);
    arg_list.set(2, bia);
    arg_list.set(3, dst);
    arg_list.set(4, oscales);
    arg_list.set(5, attr_info.common_oscales);

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

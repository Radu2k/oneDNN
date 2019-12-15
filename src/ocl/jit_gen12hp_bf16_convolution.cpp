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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/type_helpers.hpp"

#include "ocl/jit_gen12hp_bf16_convolution.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

using namespace dnnl::impl::memory_tracking::names;

status_t jit_gen12hp_bf16_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);
    auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    const auto &jcp = ker_->jcp;
    compute::kernel_arg_list_t arg_list, arg_list_zero, arg_list_cvt;
    std::unique_ptr<memory_storage_t> wei_f32_reduce, bia_f32_reduce;

    if (jcp.weights_data_type == data_type::bf16) {
        wei_f32_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                key_conv_wei_reduction);
        arg_list_zero.set(0, *wei_f32_reduce);
    } else {
        arg_list_zero.set(0, diff_weights);
    }
    if (jcp.with_bias) {
        if (jcp.bias_data_type == data_type::bf16) {
            bia_f32_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                    key_conv_bia_reduction);
            arg_list_zero.set(1, *bia_f32_reduce);
        } else
            arg_list_zero.set(1, diff_bias);
    } else
        arg_list_zero.set(1, memory_storage_t::empty_storage());

    auto nd_range
            = compute::nd_range_t({jcp.ic / (jcp.ic_block / jcp.sub_group_size)
                                                  * jcp.kd * jcp.kh * jcp.kw,
                                          jcp.oc * jcp.ngroups, 1},
                    {jcp.sub_group_size, 16, 1});
    CHECK(compute_stream->parallel_for(
            nd_range, zero_init_kernel_, arg_list_zero));

    arg_list.set(0, src);

    if (jcp.weights_data_type == data_type::bf16) {
        arg_list.set(1, *wei_f32_reduce);
    } else {
        arg_list.set(1, diff_weights);
    }

    if (jcp.with_bias) {
        if (jcp.bias_data_type == data_type::bf16)
            arg_list.set(2, *bia_f32_reduce);
        else
            arg_list.set(2, diff_bias);
    } else
        arg_list.set(2, memory_storage_t::empty_storage());

    arg_list.set(3, diff_dst);

    nd_range = compute::nd_range_t(jcp.gws_d, jcp.lws_d);
    status_t status
            = compute_stream->parallel_for(nd_range, conv_kernel_, arg_list);

    if (utils::one_of(
                data_type::bf16, jcp.weights_data_type, jcp.bias_data_type)) {
        if (jcp.weights_data_type == data_type::bf16) {
            arg_list_cvt.set(0, *wei_f32_reduce);
            arg_list_cvt.set(2, diff_weights);
        } else {
            arg_list_cvt.set(0, memory_storage_t::empty_storage());
            arg_list_cvt.set(2, memory_storage_t::empty_storage());
        }

        if (jcp.with_bias && jcp.bias_data_type == data_type::bf16) {
            arg_list_cvt.set(1, *bia_f32_reduce);
            arg_list_cvt.set(3, diff_bias);
        } else {
            arg_list_cvt.set(1, memory_storage_t::empty_storage());
            arg_list_cvt.set(3, memory_storage_t::empty_storage());
        }

        nd_range = compute::nd_range_t(
                {jcp.ic / (jcp.ic_block / jcp.sub_group_size) * jcp.kd * jcp.kh
                                * jcp.kw,
                        jcp.oc * jcp.ngroups, 1},
                {jcp.sub_group_size, 16, 1});
        status = compute_stream->parallel_for(
                nd_range, convert_f32_to_bf16_kernel_, arg_list_cvt);
    }

    return status;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl

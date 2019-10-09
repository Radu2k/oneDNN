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

status_t jit_gen12hp_bf16_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);
    auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    const auto &jcp = ker_->jcp;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, diff_weights);
    arg_list.set(2, diff_bias);
    arg_list.set(3, diff_dst);

    auto nd_range = compute::nd_range_t(jcp.gws_d, jcp.lws_d);
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl

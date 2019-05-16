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
#include "common/mkldnn_thread.hpp"
#include "common/mkldnn_traits.hpp"
#include "common/type_helpers.hpp"

#include "ocl/jit_gen12lp_u8s8s32x_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

template <data_type_t dst_type>
    status_t jit_gen12lp_u8s8s32x_convolution_fwd_t<dst_type>
    ::execute_forward(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(MKLDNN_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(MKLDNN_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(MKLDNN_ARG_DST);

    const auto &jcp = ker_->jcp;

    kernel_.set_arg(0, src);
    kernel_.set_arg(1, weights);
    kernel_.set_arg(2, bias);
    kernel_.set_arg(3, dst);
    kernel_.set_arg(4, jcp.relu_negative_slope);
    kernel_.set_arg(5, jcp.sum_scale);
    float scales = pd()->attr()->output_scales_.scales_[0];
    kernel_.set_arg(6, scales);

    auto &executor
        = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());

    auto nd_range = cl_nd_range_t(jcp.gws_d, jcp.lws_d);
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

using namespace data_type;

template struct jit_gen12lp_u8s8s32x_convolution_fwd_t<u8>;
template struct jit_gen12lp_u8s8s32x_convolution_fwd_t<s8>;

} // namespace ocl
} // namespace impl
} // namespace mkldnn


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

#include "ocl/ref_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

using math::saturate;

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type,
        data_type_t acc_type>
status_t ref_convolution_fwd_t<src_type, wei_type, dst_type, acc_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(MKLDNN_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(MKLDNN_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(MKLDNN_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(MKLDNN_ARG_DST);

    /* set kernel args */
    kernel_.set_arg(0, src);
    kernel_.set_arg(1, weights);
    kernel_.set_arg(2, bias);
    kernel_.set_arg(3, dst);
    kernel_.set_arg(4, pd()->negative_slope);
    kernel_.set_arg(5, pd()->sum_scale);

    auto nd_range = cl_nd_range_t(pd()->gws, pd()->lws);
    auto &executor
            = *(utils::downcast<cl_stream_t *>(ctx.stream())->cl_executor());
    status_t status = executor.parallel_for(nd_range, kernel_);

    return status;
}

using namespace data_type;

template struct ref_convolution_fwd_t<u8, s8, u8, s32>;
template struct ref_convolution_fwd_t<u8, s8, s8, s32>;

}
}
}

/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
* Copyright 2020-2022 Codeplay Software Limited
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

#include "common/compiler_workarounds.hpp"

#include "gpu/amd/miopen_softmax.hpp"
#include "gpu/amd/stream.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "xpu/sycl/buffer_memory_storage.hpp"
#include "xpu/sycl/memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

status_t miopen_softmax_fwd_t::execute(const exec_ctx_t &ctx) const {

    if (pd()->has_zero_dim_memory()) return status::success;

    amd::stream_t *hip_stream = utils::downcast<amd::stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);

        compat::host_task(cgh,
                [= WA_THIS_COPY_CAPTURE](const compat::interop_handle &ih) {
                    std::vector<void *> args;
                    auto &sycl_engine = *utils::downcast<amd::engine_t *>(
                            hip_stream->engine());
                    auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);

                    auto handle = hip_stream->get_miopen_handle();

                    args.push_back(arg_src.get_native_pointer(ih));
                    args.push_back(arg_dst.get_native_pointer(ih));

                    pd()->softmax_impl_->execute(
                            handle, args.data(), args.size());
                });
    });
}

status_t miopen_softmax_bwd_t::execute(const exec_ctx_t &ctx) const {

    if (pd()->has_zero_dim_memory()) return status::success;

    amd::stream_t *hip_stream = utils::downcast<amd::stream_t *>(ctx.stream());

    return hip_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DST);
        auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
        auto arg_diff_src = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);

        compat::host_task(cgh,
                [= WA_THIS_COPY_CAPTURE](const compat::interop_handle &ih) {
                    std::vector<void *> args;
                    auto &sycl_engine = *utils::downcast<amd::engine_t *>(
                            hip_stream->engine());
                    auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);

                    auto handle = hip_stream->get_miopen_handle();

                    args.push_back(arg_dst.get_native_pointer(ih));
                    args.push_back(arg_diff_dst.get_native_pointer(ih));
                    args.push_back(arg_diff_src.get_native_pointer(ih));

                    pd()->softmax_impl_->execute(
                            handle, args.data(), args.size());
                });
    });
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

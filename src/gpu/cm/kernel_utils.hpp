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

#ifndef GPU_CM_KERNEL_UTILS_HPP
#define GPU_CM_KERNEL_UTILS_HPP

#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/kernel_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cm {

inline bool is_cm_enabled() {
    return dnnl::impl::getenv_int("DNNL_ENABLE_CM", 0) != 0;
}

const char **get_kernel_source(const char *name);

inline status_t create_kernels(const compute::compute_engine_t *engine,
        compute::kernel_list_t &kernel_list,
        const compute::kernel_ctx_t &kernel_ctx) {
    return ocl::create_kernels(
            engine, kernel_list, kernel_ctx, cm::get_kernel_source);
}

inline compute::kernel_t create_kernel(const compute::compute_engine_t *engine,
        const std::string &name, const compute::kernel_ctx_t &kernel_ctx) {
    compute::kernel_t kernel;
    compute::kernel_list_t kernel_list;
    kernel_list.add(name.c_str(), &kernel);
    create_kernels(engine, kernel_list, kernel_ctx);
    return kernel;
}

} // namespace cm
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_CM_KERNEL_UTILS_HPP

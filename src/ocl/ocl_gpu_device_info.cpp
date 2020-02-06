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

#include "ocl/ocl_gpu_device_info.hpp"

#include "cpu/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

size_t ocl_gpu_device_info_t::get_llc_cache_size() const {
    // Integrated GPUs share LLC with CPU which is L3 cache on CPU.
    size_t cache_size = cpu::get_cache_size(3, false);
    return cache_size;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl
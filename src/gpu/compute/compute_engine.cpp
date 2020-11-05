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

#include "gpu/compute/compute_engine.hpp"

#include "common/utils.hpp"

bool dnnl_impl_gpu_mayiuse_ngen_kernels(dnnl::impl::engine_t *engine) {
    using namespace dnnl::impl;
    using namespace dnnl::impl::gpu::compute;

    auto *compute_engine = utils::downcast<compute_engine_t *>(engine);
    return compute_engine->mayiuse_ngen_kernels();
}

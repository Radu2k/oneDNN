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

#ifndef JIT_GEN12HP_SYSTOLIC_GEMM_COPY_KERNEL_HPP
#define JIT_GEN12HP_SYSTOLIC_GEMM_COPY_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct jit_gen12hp_systolic_gemm_copy_kernel {
    static status_t init_const_def(
            compute::kernel_ctx_t &kernel_ctx, bool copyb, bool trans) {

        kernel_ctx.define_int("COPY_A", int(!copyb));
        kernel_ctx.define_int("COPY_B", int(copyb));
        kernel_ctx.define_int("COPY_TRANS", int(trans));
        kernel_ctx.add_option("-cl-strict-aliasing");

        return status::success;
    }

    static constexpr int unroll_r(bool copyb, bool trans) {
        return !copyb ? 32 : 16;
    }

    static constexpr int unroll_c(bool copyb, bool trans) {
        return !copyb ? 16 : 48;
    }

    static constexpr int subgroup_size(bool copyb, bool trans) {
        return (copyb && trans) ? 16 : 8;
    }
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif

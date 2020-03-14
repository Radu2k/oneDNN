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

#ifndef GPU_JIT_GEN12HP_SYSTOLIC_GEMM_HPP
#define GPU_JIT_GEN12HP_SYSTOLIC_GEMM_HPP

#include <assert.h>
#include <memory>
#include <tuple>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_gemm_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct gen12hp_systolic_gemm_t : public gpu_gemm_t {
    struct pd_t : public gpu_gemm_pd_t {
        using hint_class = void;

        pd_t(engine_t *engine, const gemm_desc_t *adesc,
                const primitive_attr_t *attr, const hint_class *)
            : gpu_gemm_pd_t(engine, adesc, attr, nullptr) {}

        DECLARE_COMMON_PD_T("ngen:gen12hp:gemm:any", gen12hp_systolic_gemm_t);

        status_t init();

        dim_t m_aligned() const;
        dim_t n_aligned() const;
        dim_t k_aligned() const;

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;

        float alpha() const { return attr()->output_scales_.scales_[0]; }

        float beta() const {
            using namespace primitive_kind;
            const auto &p = attr()->post_ops_;
            return p.contain(sum, 0) ? p.entry_[0].sum.scale : 0.f;
        }
    };

    status_t init() override;

public:
    gen12hp_systolic_gemm_t(const pd_t *apd) : gpu_gemm_t(apd) {}

    virtual status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    int64_t default_block_k(data_type_t dt) const;
    std::tuple<int64_t, int64_t, int64_t> get_blocking() const;

    status_t launch_clear_sum(compute::compute_stream_t *compute_stream,
            int64_t r, int64_t c, const memory_storage_t &dst,
            int32_t offset_dst, int32_t ld_dst, bool copyb) const;
    status_t launch_copy(compute::compute_stream_t *compute_stream, int64_t r,
            int64_t c, const memory_storage_t &src, int64_t offset_src,
            int64_t ld_src, const memory_storage_t &dst, int32_t offset_dst,
            int32_t ld_dst, bool copyb) const;
    status_t launch_compute(compute::compute_stream_t *compute_stream,
            int32_t m, int32_t n, int32_t k, const memory_storage_t &ap,
            int64_t offset_a, int32_t lda, const memory_storage_t &bp,
            int64_t offset_b, int32_t ldb, const memory_storage_t &c,
            int64_t offset_c, int32_t ldc, float alpha, float beta, int16_t ao,
            int16_t bo, const memory_storage_t &co, int32_t offset_co,
            bool first_k_block, bool last_k_block) const;

    compute::kernel_t kernel_[2][2]; // [first_k_block][last_k_block]
    compute::kernel_t copy_kernel_[2][2]; // [trans][clear_sum]

    std::unique_ptr<memory_storage_t> a_packed_;
    std::unique_ptr<memory_storage_t> b_packed_;

    char co_type_;
    bool ab_zero_points_;

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

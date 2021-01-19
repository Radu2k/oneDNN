/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct gen12hp_systolic_gemm_t : public gpu_gemm_t {
    struct pd_t : public gpu_gemm_pd_t {
        using hint_class = void;

        pd_t(const gemm_desc_t *adesc, const primitive_attr_t *attr,
                const hint_class *)
            : gpu_gemm_pd_t(adesc, attr, nullptr) {}

        DECLARE_COMMON_PD_T("ngen:gen12hp:gemm:any", gen12hp_systolic_gemm_t);

        status_t init(engine_t *engine);

        bool set_default_formats(data_type_t dt) {
            using namespace format_tag;

            auto sz = types::data_type_size(dt);

            // Packed not implemented for int8 yet.
            if (sz == 2) {
                memory_desc_wrapper a_mdw(&desc_.b_desc);
                memory_desc_wrapper b_mdw(&desc_.a_desc);
                memory_desc_wrapper c_mdw(&desc_.c_desc);

                bool a_any = a_mdw.format_any();
                bool b_any = b_mdw.format_any();
                bool c_any = c_mdw.format_any();
                bool batch = desc()->is_batched();

                format_tag_t a_packed_tag = batch
                        ? ((sz == 2) ? aCB4c8b8c2b : aCB4c8b8c4b)
                        : ((sz == 2) ? BA4b8a8b2a : BA4b8a8b4a);
                format_tag_t b_packed_tag = batch
                        ? ((sz == 2) ? aBC48b16c : aBC48b32c)
                        : ((sz == 2) ? AB48a16b : AB48a32b);

                if (a_any)
                    CHECK(memory_desc_init_by_tag(desc_.b_desc, a_packed_tag));
                else if (a_mdw.matches_one_of_tag(
                                 a_packed_tag, ab, ba, abc, acb)
                        == undef)
                    return false;
                if (b_any)
                    CHECK(memory_desc_init_by_tag(desc_.a_desc, b_packed_tag));
                else if (b_mdw.matches_one_of_tag(
                                 b_packed_tag, ab, ba, abc, acb)
                        == undef)
                    return false;
                if (c_any)
                    CHECK(memory_desc_init_by_tag(desc_.c_desc, b_packed_tag));
                else if (c_mdw.matches_one_of_tag(b_packed_tag, ab, abc)
                        == undef)
                    return false;

                packed_a_ = a_mdw.matches_tag(a_packed_tag);
                packed_b_ = b_mdw.matches_tag(b_packed_tag);
                packed_c_ = c_mdw.matches_tag(b_packed_tag);
            }

            return gpu_gemm_pd_t::set_default_formats();
        }

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;

        float alpha() const { return attr()->output_scales_.scales_[0]; }

        float beta() const {
            using namespace primitive_kind;
            const auto &p = attr()->post_ops_;
            return p.contain(sum, 0) ? p.entry_[0].sum.scale : 0.f;
        }

        bool with_bias() const {
            return desc()->bias_type() != data_type::undef;
        }

        int bias_cmask() const {
            unsigned char to_cmask[4] = {0, 2, 1, 3};
            return with_bias() ? to_cmask[(desc()->bias_mask() >> 1) & 3] : -1;
        }

        const attr_info_t *attr_info() const { return &attr_info_; }

        bool packed_a() const { return packed_a_; }
        bool packed_b() const { return packed_b_; }
        bool packed_c() const { return packed_c_; }

        dim_t lda_packed() const {
            return packed_a() ? desc()->b_desc.padded_dims[with_batch() ? 1 : 0]
                              : 0;
        }
        dim_t ldb_packed() const {
            return packed_b() ? desc()->a_desc.padded_dims[with_batch() ? 2 : 1]
                              : 0;
        }
        dim_t ldc_packed() const {
            return packed_c() ? desc()->c_desc.padded_dims[with_batch() ? 2 : 1]
                              : 0;
        }

        bool with_batch() const { return desc()->is_batched(); }

    private:
        attr_info_t attr_info_ = {};
        bool packed_a_ = false, packed_b_ = false, packed_c_ = false;
    };

    status_t init(engine_t *engine) override;
    status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const override;

public:
    gen12hp_systolic_gemm_t(const pd_t *apd) : gpu_gemm_t(apd) {}

    virtual status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    bool enable_mn_blocking() const;
    int64_t default_block_m() const;
    int64_t default_block_n() const;
    int64_t default_block_k(data_type_t dt) const;
    std::tuple<int64_t, int64_t, int64_t> get_blocking() const;

    status_t launch_clear_sum(const gemm_exec_ctx_t &ctx, int64_t r, int64_t c,
            const memory_storage_t &dst, int32_t offset_dst, int32_t ld_dst,
            bool copyb) const;
    status_t launch_copy(const gemm_exec_ctx_t &ctx, int64_t r, int64_t c,
            const memory_storage_t &src, int64_t offset_src, int64_t ld_src,
            const memory_storage_t &dst, int32_t offset_dst, int32_t ld_dst,
            bool copyb) const;
    status_t launch_compute(const gemm_exec_ctx_t &ctx, int32_t m, int32_t n,
            int32_t k, const memory_storage_t &ap, int64_t offset_a,
            int32_t lda, const memory_storage_t &bp, int64_t offset_b,
            int32_t ldb, const memory_storage_t &c, int64_t offset_c,
            int32_t ldc, float alpha, float beta, int16_t ao, int16_t bo,
            const memory_storage_t &co, int32_t offset_co, bool first_k_block,
            bool last_k_block, int32_t batch, int32_t stride_a,
            int32_t stride_b, int32_t stride_c) const;

    static const int A_PACKED_ = 0;
    static const int B_PACKED_ = 1;

    compute::kernel_t kernel_[2][2]; // [first_k_block][last_k_block]
    compute::kernel_t copy_kernel_[2][2]; // [trans][clear_sum]

    compute::gpu_arch_t arch_;
    int eu_count_;

    int unroll_m_ = 0;
    int unroll_n_ = 0;

    char co_kind_;
    bool ab_zero_points_;
    bool walk_n_first_;

    const pd_t *pd() const { return (const pd_t *)gpu_primitive_t::pd().get(); }
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

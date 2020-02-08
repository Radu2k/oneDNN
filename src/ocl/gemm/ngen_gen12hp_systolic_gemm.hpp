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

#ifndef NGEN_GEN12HP_SYSTOLIC_GEMM_HPP
#define NGEN_GEN12HP_SYSTOLIC_GEMM_HPP

#include <assert.h>
#include <memory>
#include <tuple>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"
#include "compute/compute.hpp"
#include "ocl/gemm/jit_gen12hp_systolic_gemm_copy_kernel.hpp"
#include "ocl/gemm/ngen_gen12hp_systolic_gemm_kernel.hpp"
#include "ocl/gemm/ocl_gemm.hpp"
#include "ocl/gemm/ocl_gemm_pd.hpp"
#include "ocl/ngen_type_bridge.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct ngen_gen12hp_systolic_gemm_t : public ocl_gemm_t {
    struct pd_t : public ocl_gemm_pd_t {
        using hint_class = void;

        pd_t(engine_t *engine, const gemm_desc_t *adesc,
                const primitive_attr_t *attr, const hint_class *)
            : ocl_gemm_pd_t(engine, adesc, attr, nullptr) {}

        DECLARE_COMMON_PD_T(
                "ngen:gen12hp:gemm:any", ngen_gen12hp_systolic_gemm_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;
            using namespace primitive_kind;

            assert(this->engine()->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());

            const auto d = desc();

            // LIMITATIONS:
            // - batch is not supported
            // - runtime dims are not supported
            // - bias is not supported
            bool limits_ok = d->batch == 1
                    && !utils::one_of(DNNL_RUNTIME_DIM_VAL, d->m, d->n, d->k,
                            d->lda, d->ldb, d->ldc)
                    && d->bias_type == data_type::undef;

            bool ok = true && limits_ok && d->a_type == d->b_type
                    && utils::one_of(d->a_type, bf16, f16)
                    && utils::one_of(d->c_type, f32, d->a_type)
                    && compute_engine->mayiuse(compute::device_ext_t::
                                    intel_subgroup_split_matrix_multiply_accumulate)
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }

        dim_t m_aligned() const {
            using kernel_t = ngen_gen12hp_systolic_gemm_kernel_t;
            return utils::rnd_up(
                    desc()->m, kernel_t::unroll_m * kernel_t::thread_group_m);
        }

        dim_t n_aligned() const {
            using kernel_t = ngen_gen12hp_systolic_gemm_kernel_t;
            return utils::rnd_up(
                    desc()->n, kernel_t::unroll_n * kernel_t::thread_group_n);
        }

        dim_t k_aligned() const {
            return utils::rnd_up(desc()->k,
                    ngen_gen12hp_systolic_gemm_kernel_t::unroll_k(
                            desc()->a_type));
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
    };

    status_t init() override {
        using namespace data_type;

        auto *gpu_engine = utils::downcast<ocl_gpu_engine_t *>(engine());
        if (!gpu_engine) return status::out_of_memory;

        auto a_type = pd()->desc()->a_type;
        auto b_type = pd()->desc()->b_type;
        auto c_type = pd()->desc()->c_type;
        auto acc_type = pd()->desc()->acc_type;

        if (utils::one_of(acc_type, f16, bf16)) acc_type = f32;

        int64_t block_m = 0, block_n = 0, block_k = 0;
        std::tie(block_m, block_n, block_k) = get_blocking();

        memory_storage_t *a_packed_ptr, *b_packed_ptr;
        this->engine()->create_memory_storage(&a_packed_ptr,
                block_m * block_k * types::data_type_size(a_type));
        this->engine()->create_memory_storage(&b_packed_ptr,
                block_n * block_k * types::data_type_size(b_type));
        if (!a_packed_ptr || !b_packed_ptr) return status::runtime_error;
        a_packed_.reset(a_packed_ptr);
        b_packed_.reset(b_packed_ptr);

        // Initialize compute kernels (assembly)
        ngen_gen12hp_systolic_gemm_kernel_t::config_t cfg;

        cfg.a_type = convert_dnnl_type_to_ngen(a_type);
        cfg.b_type = convert_dnnl_type_to_ngen(b_type);
        cfg.c_type = convert_dnnl_type_to_ngen(c_type);
        cfg.acc_type = convert_dnnl_type_to_ngen(acc_type);
        cfg.alpha1 = (pd()->alpha() == 1.0f);
        cfg.beta0 = (pd()->beta() == 0.0f);
        cfg.beta1 = (pd()->beta() == 1.0f);

        if (!cfg.beta1) {
            auto ngen_kernel = ngen_gen12hp_systolic_gemm_kernel_t(cfg);
            kernel_ = compute::kernel_t(
                    new ocl_gpu_kernel_t(ngen_kernel.getKernel(
                            gpu_engine->context(), gpu_engine->device())));
        }

        cfg.beta0 = false;
        cfg.beta1 = true;
        auto ngen_kernel_b1 = ngen_gen12hp_systolic_gemm_kernel_t(cfg);
        kernel_b1_ = compute::kernel_t(
                new ocl_gpu_kernel_t(ngen_kernel_b1.getKernel(
                        gpu_engine->context(), gpu_engine->device())));

        // Initialize copy kernels (OpenCL)
        for (bool copy_b : {false, true}) {
            compute::kernel_ctx_t kernel_ctx;

            auto trans = !copy_b ? pd()->desc()->transa : pd()->desc()->transb;
            auto status = jit_gen12hp_systolic_gemm_copy_kernel::init_const_def(
                    kernel_ctx, copy_b, trans);
            if (status != status::success) return status;

            gpu_engine->create_kernel(&copy_kernel_[copy_b],
                    "gen12hp_systolic_gemm_copy", kernel_ctx);
            if (!copy_kernel_[copy_b]) return status::runtime_error;
        }

        return status::success;
    }

public:
    ngen_gen12hp_systolic_gemm_t(const pd_t *apd) : ocl_gemm_t(apd) {}

    virtual status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    std::tuple<int64_t, int64_t, int64_t> get_blocking() const;

    status_t launch_copy(compute::compute_stream_t *compute_stream, int64_t r,
            int64_t c, const memory_storage_t &src, int64_t offset_src,
            int64_t ld_src, const memory_storage_t &dst, int32_t offset_dst,
            int32_t ld_dst, bool copyb) const;
    status_t launch_compute(compute::compute_stream_t *compute_stream,
            int32_t m, int32_t n, int32_t k, const memory_storage_t &ap,
            int64_t offset_a, int32_t lda, const memory_storage_t &bp,
            int64_t offset_b, int32_t ldb, const memory_storage_t &c,
            int64_t offset_c, int32_t ldc, float alpha, float beta) const;

    compute::kernel_t kernel_;
    compute::kernel_t kernel_b1_;
    compute::kernel_t copy_kernel_[2]; // [trans]

    std::unique_ptr<memory_storage_t> a_packed_;
    std::unique_ptr<memory_storage_t> b_packed_;

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

} // namespace ocl
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

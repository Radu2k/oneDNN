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

#ifndef JIT_GEN12LP_GEMM_HPP
#define JIT_GEN12LP_GEMM_HPP

#include <assert.h>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_gen12lp_gemm_kernel.hpp"
#include "ocl/ocl_gemm_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

template <impl::data_type_t a_type, impl::data_type_t b_type,
        impl::data_type_t c_type, typename ao_type, typename bo_type>
struct jit_gen12lp_gemm_t : public primitive_t {
    using c_t = typename prec_traits<c_type>::type;

    enum class type {
        no_copy
    };

    struct pd_t : public ocl_gemm_pd_t {
        using hint_class = void;

        pd_t(engine_t *engine, const gemm_desc_t *adesc,
                const primitive_attr_t *attr, const hint_class *)
            : ocl_gemm_pd_t(engine, adesc, attr) {}

        DECLARE_COMMON_PD_T("ocl:gemm:any", jit_gen12lp_gemm_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;

            assert(this->engine()->kind() == engine_kind::gpu);
            auto *compute_engine 
                    = utils::downcast<compute::compute_engine_t *>(engine());
            
            bool ok = true && desc()->a_type == a_type
                    && desc()->b_type == b_type && desc()->c_type == c_type
                    && compute_engine->mayiuse(compute::device_ext_t::intel_subgroups)
                    && IMPLICATION(c_type == s32,
                               true
                                       && compute_engine->mayiuse(compute::device_ext_t::
                                                          intel_subgroups_short));
            if (!ok)
                return status::unimplemented;

            return status::success;
        }

        bool with_eltwise() const {
            return attr()->post_ops_.find(primitive_kind::eltwise) != -1;
        }

        float eltwise_alpha() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise()
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha
                    : 1.0f;
        }

        float eltwise_beta() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise()
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.beta
                    : 0.0f;
        }

        alg_kind_t eltwise_alg_kind() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise()
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alg
                    : mkldnn_alg_kind_undef;
        }

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;
        size_t dyn_offset_co = 0;
    };

    status_t init() override {
        auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine());

        eu_count_ = compute_engine->get_eu_count();
        hw_threads_ = compute_engine->get_hw_threads();

        gemm_type_ = get_gemm_type();

        switch (gemm_type_) {
        case type::no_copy: return init_nocopy();
        }

        return status::invalid_arguments;
    }

    status_t init_nocopy() {
        const char *kernel_name = nullptr;

        //compute kernel
        switch (c_type) {
        case data_type::s32: kernel_name = "gen12lp_gemm_compute_x8x8s32_kernel"; break;
        default: return status::unimplemented;
        }

        auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;
        
        memory_storage_t *temp_buf_ptr;
        //this->engine()->create_memory_storage(&temp_buf_ptr, 128 << 20); 
        this->engine()->create_memory_storage(&temp_buf_ptr, pd()->desc()->m * pd()->desc()->n * sizeof(int)); 
        temp_buf_.reset(temp_buf_ptr);
            
        bool fixed_c = (pd()->desc()->offsetc == mkldnn_fixed);
        bool column_c = (pd()->desc()->offsetc == mkldnn_column);
        bool row_c = (pd()->desc()->offsetc == mkldnn_row);

        auto status = jit_gen12lp_gemm_x8x8s32_kernel<c_type, ao_type, bo_type>::init_const_def(kernel_ctx, 
                pd()->desc()->transa, pd()->desc()->transb, fixed_c, column_c, row_c,
                pd()->with_eltwise(), pd()->eltwise_alg_kind());
        if (status != status::success)
            return status;

        compute_engine->create_kernel(&compute_x8x8s32_kernel_, kernel_name, kernel_ctx);
        if (!compute_x8x8s32_kernel_)
            return status::runtime_error;

        
        //scale kernel
        kernel_name = "gen12lp_gemm_scale_x8x8s32_kernel";

        status = jit_gen12lp_gemm_scale_x8x8s32_kernel<c_type, ao_type, bo_type>::init_const_def(kernel_ctx, 
                pd()->with_eltwise(), pd()->eltwise_alg_kind());
        if (status != status::success)
            return status;

        compute_engine->create_kernel(&scale_x8x8s32_kernel_, kernel_name, kernel_ctx);
        if (!scale_x8x8s32_kernel_)
            return status::runtime_error;

        return status::success;
    }


    jit_gen12lp_gemm_t(const pd_t *apd) : primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override;

private:

    status_t launch_x8x8s32(compute::compute_stream_t *s, const memory_storage_t &a,
            const memory_storage_t &b, const memory_storage_t &c,
            int64_t offset_a, int64_t offset_b, int64_t offset_c, int32_t lda,
            int32_t ldb, int32_t ldc, int32_t m, int32_t n, int32_t k, 
            int32_t beta, ao_type ao, bo_type bo, const memory_storage_t &co, 
            int32_t offset_co, bool apply_co, bool apply_eltwise, 
            c_t eltwise_alpha, c_t eltwise_beta) const;

    status_t launch_scale_x8x8s32(compute::compute_stream_t *s, const memory_storage_t &c_temp, 
            const memory_storage_t &c, char offsetc, int64_t 
            offset_c, int32_t m, int32_t n, int32_t ldc, float alpha,
            float beta, const memory_storage_t &co, int32_t offset_co,
            bool alpha_is_zero, bool apply_eltwise,
            c_t eltwise_alpha, c_t eltwise_beta) const;


    virtual status_t execute_standard(const exec_ctx_t &ctx) const;

    compute::kernel_t compute_x8x8s32_kernel_;
    compute::kernel_t scale_x8x8s32_kernel_;

    std::unique_ptr<memory_storage_t> temp_buf_;

    type gemm_type_ = type::no_copy;
    int hw_threads_ = 0;
    int eu_count_ = 0;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    
    type get_gemm_type() const {
        return type::no_copy;
    }
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GPU_OCL_GEN12HP_1ST_BWD_CONVOLUTION_HPP
#define GPU_OCL_GEN12HP_1ST_BWD_CONVOLUTION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/gpu_eltwise_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen12hp_1st_convolution_bwd_weights_t : public gpu_primitive_t {
    struct pd_t : public gpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &rhs)
            : gpu_convolution_bwd_weights_pd_t(rhs), conf(rhs.conf) {
            if (rhs.rpd_wei_) rpd_wei_.reset(rhs.rpd_wei_->clone());
            if (rhs.rpd_bia_) rpd_bia_.reset(rhs.rpd_bia_->clone());
        }

        DECLARE_COMMON_PD_T(
                "ocl:gen12hp:1st", gen12hp_1st_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            bool ok = set_default_alg_kind(alg_kind::convolution_direct)
                    && this->desc()->prop_kind == backward_weights
                    && this->desc()->alg_kind == alg_kind::convolution_direct
                    && utils::one_of(this->desc()->diff_weights_desc.data_type,
                            bf16, f32)
                    && utils::one_of(this->desc()->src_desc.data_type, bf16)
                    && utils::one_of(
                            this->desc()->diff_dst_desc.data_type, bf16)
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && compute_engine->mayiuse(
                            compute::device_ext_t::khr_int64_base_atomics)
                    && !has_zero_dim_memory() && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            status_t status = init_conf(engine);
            if (status != status::success) return status;
            if (!IMPLICATION(utils::one_of(bf16,
                                     this->desc()->diff_weights_desc.data_type,
                                     this->desc()->src_desc.data_type,
                                     this->desc()->diff_dst_desc.data_type),
                        conf.ver == ver_1stconv))
                return status::unimplemented;

            init_scratchpad();
            return status::success;
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conv_conf_t conf;
        std::unique_ptr<primitive_desc_t> rpd_wei_;
        std::unique_ptr<primitive_desc_t> rpd_bia_;

    private:
        status_t init_scratchpad();
    };

    gen12hp_1st_convolution_bwd_weights_t(const pd_t *apd)
        : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        const char *kernel_name;

        kernel_name = "gen12hp_1st_conv_bwd_weights";

        if (pd()->conf.reorder_wei) {
            CHECK(pd()->rpd_wei_->create_primitive(wei_reorder_, engine));
        }
        if (pd()->conf.reorder_bias) {
            CHECK(pd()->rpd_bia_->create_primitive(bia_reorder_, engine));
        }
        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        create_kernel(engine, &kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;
        return status::success;
    }

    primitive_list_t nested_primitives() const override {
        primitive_list_t prim_list;
        if (pd()->conf.reorder_wei)
            prim_list.emplace(prim_list.begin(), wei_reorder_.get());
        if (pd()->conf.reorder_bias)
            prim_list.emplace(prim_list.begin(), bia_reorder_.get());

        return prim_list;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
    std::shared_ptr<primitive_t> wei_reorder_;
    std::shared_ptr<primitive_t> bia_reorder_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

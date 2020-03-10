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

#ifndef GPU_GEN12LP_X8S8S32X_CONVOLUTION_HPP
#define GPU_GEN12LP_X8S8S32X_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen12lp_x8s8s32x_convolution_fwd_t : public primitive_impl_t {
    struct pd_t : public gpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:gen12lp", gen12lp_x8s8s32x_convolution_fwd_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;
            assert(this->engine()->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops;

            bool ok = true
                    && utils::one_of(this->desc()->prop_kind, forward_training,
                            forward_inference)
                    && this->desc()->alg_kind == alg_kind::convolution_direct
                    && utils::one_of(desc()->src_desc.data_type, u8, s8)
                    && utils::one_of(desc()->dst_desc.data_type, u8, s8)
                    && expect_data_types(desc()->src_desc.data_type, s8, f32,
                            desc()->dst_desc.data_type, s32)
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_ok(attr())
                    && IMPLICATION(!attr()->output_scales_.has_default_values(),
                            attr()->output_scales_.mask_ == 0);

            if (!ok) return status::unimplemented;

            status_t status = init_conf();
            if (status != status::success) return status;

            ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);
            return ok ? status::success : status::unimplemented;
        }

        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conv_conf_t conf;
    };

    status_t init() override {
        const char *kernel_name = nullptr;
        if (pd()->conf.is_depthwise) {
            if (pd()->conf.mb_block == 32)
                kernel_name = "conv_dw_fwd_mb_block_x8s8s32x";
            else
                kernel_name = "conv_dw_fwd_ow_block_x8s8s32x";
        } else {
            if (pd()->conf.ic > 4) {
                if (pd()->conf.mb_block == 32)
                    kernel_name = "conv_fwd_mb_block_x8s8s32x";
                else
                    kernel_name = "conv_fwd_ow_block_x8s8s32x";
            } else {
                kernel_name = "conv_fwd_first_x8s8s32x";
            }
        }
        compute::kernel_ctx_t kernel_ctx;
        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute_engine->create_kernel(&kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    gen12lp_x8s8s32x_convolution_fwd_t(const pd_t *apd)
        : primitive_impl_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

struct gen12lp_x8s8s32x_convolution_bwd_data_t : public primitive_impl_t {
    struct pd_t : public gpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                "ocl:gen12lp", gen12lp_x8s8s32x_convolution_bwd_data_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;
            assert(this->engine()->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            bool ok = true
                    && utils::one_of(desc()->diff_src_desc.data_type, s8, u8)
                    && utils::one_of(desc()->diff_dst_desc.data_type, s8, u8)
                    && expect_data_types(desc()->diff_src_desc.data_type, s8,
                            f32, desc()->diff_dst_desc.data_type, s32)
                    && desc()->prop_kind == prop_kind::backward_data
                    && desc()->alg_kind == alg_kind::convolution_direct
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_ok(attr());

            if (!ok) return status::unimplemented;

            status_t status = init_conf();
            if (status != status::success) return status;

            ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);
            return ok ? status::success : status::unimplemented;
        }

        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conv_conf_t conf;

        bool support_bias() const override { return true; }
    };

    status_t init() override {
        const char *kernel_name = "conv_bwd_data_x8s8s32x";
        compute::kernel_ctx_t kernel_ctx;
        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute_engine->create_kernel(&kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    gen12lp_x8s8s32x_convolution_bwd_data_t(const pd_t *apd)
        : primitive_impl_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

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

#ifndef GPU_OCL_GEN12HP_CONVOLUTION_HPP
#define GPU_OCL_GEN12HP_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen12hp_convolution_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:gen12hp", gen12hp_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops;

            bool ok = true
                    && utils::one_of(this->desc()->prop_kind, forward_training,
                            forward_inference)
                    && this->desc()->alg_kind == alg_kind::convolution_direct
                    && utils::one_of(true,
                            expect_data_types(f16, f16, f16, f16, f16),
                            expect_data_types(bf16, bf16, bf16, bf16, f32),
                            expect_data_types(bf16, bf16, f32, bf16, f32),
                            expect_data_types(bf16, bf16, bf16, f32, f32),
                            expect_data_types(bf16, bf16, f32, f32, f32),
                            expect_data_types(u8, s8, f32, u8, s32),
                            expect_data_types(u8, s8, f32, s8, s32),
                            expect_data_types(s8, s8, f32, u8, s32),
                            expect_data_types(s8, s8, f32, s8, s32))
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_ok(attr());
            if (!ok) return status::unimplemented;

            status_t status = init_conf();
            if (status != status::success) return status;

            ok = !conf.attr_info.with_per_oc_oscales
                    && set_default_formats_common(
                            conf.src_tag, conf.wei_tag, conf.dst_tag);

            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            auto *dev_info = compute_engine->device_info();
            is_gen12hp = dev_info->gpu_arch() == compute::gpu_arch_t::gen12hp;

            return ok ? status::success : status::unimplemented;
        }

        bool is_gen12hp = false;
        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conv_conf_t conf;
    };

    status_t init(engine_t *engine) override {
        const char *kernel_name = "gen12hp_conv_fwd";

        compute::kernel_ctx_t kernel_ctx;
        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        create_kernel(engine, &kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    gen12hp_convolution_fwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)gpu_primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

struct gen12hp_convolution_bwd_data_t : public gpu_primitive_t {
    struct pd_t : public gpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:gen12hp", gen12hp_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops;

            bool ok = utils::one_of(true,
                              expect_data_types(u8, s8, f32, u8, s32),
                              expect_data_types(u8, s8, f32, s8, s32),
                              expect_data_types(s8, s8, f32, u8, s32),
                              expect_data_types(s8, s8, f32, s8, s32),
                              expect_data_types(f16, f16, f16, f16, f16),
                              expect_data_types(bf16, bf16, bf16, bf16, f32))
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
            auto *dev_info = compute_engine->device_info();
            is_gen12hp = dev_info->gpu_arch() == compute::gpu_arch_t::gen12hp;

            return ok ? status::success : status::unimplemented;
        }

        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        bool is_gen12hp = false;
        conv_conf_t conf;

        bool support_bias() const override { return true; }
    };

    status_t init(engine_t *engine) override {
        const char *kernel_name = "gen12hp_conv_bwd_data";
        compute::kernel_ctx_t kernel_ctx;
        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        create_kernel(engine, &kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    gen12hp_convolution_bwd_data_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)gpu_primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

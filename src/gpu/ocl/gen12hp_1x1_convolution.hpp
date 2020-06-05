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

#ifndef GPU_OCL_GEN12HP_1X1_CONVOLUTION_HPP
#define GPU_OCL_GEN12HP_1X1_CONVOLUTION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen12hp_1x1_convolution_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:gen12hp:1x1", gen12hp_1x1_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops;

            bool ok = utils::one_of(desc()->prop_kind, forward_training,
                              forward_inference)
                    && desc()->alg_kind == alg_kind::convolution_direct
                    && utils::one_of(
                            desc()->dst_desc.data_type, bf16, f16, s8, u8)
                    && utils::one_of(true,
                            expect_data_types(u8, s8, f32,
                                    desc()->dst_desc.data_type, s32),
                            expect_data_types(bf16, bf16,
                                    desc()->dst_desc.data_type,
                                    desc()->dst_desc.data_type, f32),
                            expect_data_types(f16, f16, f16,
                                    desc()->dst_desc.data_type, f16))
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_ok(attr())
                    && IMPLICATION(!attr()->output_scales_.has_default_values(),
                            utils::one_of(src_md_.data_type, s8, u8)
                                    && attr()->output_scales_.mask_ == 0);

            if (!ok) return status::unimplemented;

            status_t status = init_conf();
            if (status != status::success) return status;

            ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);

            auto *compute_engine = utils::downcast<ocl_gpu_engine_t *>(engine);
            if (engine->runtime_kind() == dnnl_runtime_ocl) {
                auto *dev_info = utils::downcast<const ocl_gpu_device_info_t *>(
                        compute_engine->device_info());
                is_gen12hp = dev_info->gpu_arch() == gpu_arch_t::gen12hp;
            }

            return ok ? status::success : status::unimplemented;
        }

        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conv_conf_t conf;
        bool is_gen12hp = false;
    };

    status_t init(engine_t *engine) {
        const char *kernel_name = nullptr;
        if (pd()->desc()->src_desc.data_type == data_type::f16
                || pd()->desc()->src_desc.data_type == data_type::bf16)
            kernel_name = "gen12hp_1x1_conv_fwd_x16";
        else if (pd()->desc()->src_desc.data_type == data_type::u8)
            kernel_name = "gen12hp_1x1_conv_fwd_u8s8s32x";
        else
            assert(!"not expected");

        compute::kernel_ctx_t kernel_ctx;
        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        create_kernel(engine, &kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    gen12hp_1x1_convolution_fwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)gpu_primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

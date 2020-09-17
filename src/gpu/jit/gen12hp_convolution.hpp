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

#ifndef GPU_JIT_GEN12HP_CONVOLUTION_HPP
#define GPU_JIT_GEN12HP_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/jit/jit_eltwise_injector.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct gen12hp_convolution_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_convolution_fwd_pd_t {
        using gpu_convolution_fwd_pd_t::gpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("ngen:gen12hp", gen12hp_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;

            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            if (!compute_engine->is_jit_gen12hp()) return status::unimplemented;
            if (!compute_engine->mayiuse_ngen_kernels())
                return status::unimplemented;

            const auto attr_skip_mask
                    = primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::post_ops
                    | primitive_attr_t::skip_mask_t::sum_dt;
            bool ok = set_default_alg_kind(alg_kind::convolution_direct)
                    && utils::one_of(desc()->prop_kind, forward_training,
                            forward_inference)
                    && data_types_ok()
                    && attr()->has_default_values(
                            attr_skip_mask, desc()->dst_desc.data_type)
                    && post_ops_ok(attr())
                    && IMPLICATION(!attr()->output_scales_.has_default_values(),
                            utils::one_of(src_md_.data_type, s8, u8)
                                    && utils::one_of(
                                            attr()->output_scales_.mask_, 0,
                                            1 << 1));
            if (!ok) return status::unimplemented;

            CHECK(init_conf(engine));

            ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);

            return ok ? status::success : status::unimplemented;
        }

        bool data_types_ok() const {
            using namespace data_type;

            auto src_dt = invariant_src_md()->data_type;
            auto wei_dt = invariant_wei_md()->data_type;
            auto dst_dt = invariant_dst_md()->data_type;
            auto acc_dt = desc_.accum_data_type;

            bool is_int8 = (acc_dt == s32);
            is_int8 &= utils::one_of(src_dt, u8, s8);
            is_int8 &= utils::one_of(wei_dt, u8, s8);
            is_int8 &= utils::one_of(dst_dt, u8, s8, s32, f32);
            if (is_int8) return true;

            // Ignore accumulator type set to f16 and use f32.
            bool is_f16 = (acc_dt == f16 || acc_dt == f32);
            is_f16 &= (src_dt == f16);
            is_f16 &= (wei_dt == f16);
            is_f16 &= utils::one_of(dst_dt, f16, f32);
            if (is_f16) return true;

            bool is_bf16 = (acc_dt == f32);
            is_bf16 &= (src_dt == bf16);
            is_bf16 &= (wei_dt == bf16);
            is_bf16 &= utils::one_of(dst_dt, bf16, f32);
            if (is_bf16) return true;

            // Not supported.
            return false;
        }

        bool post_ops_ok(const primitive_attr_t *attr) const override {
            if (!gpu_convolution_fwd_pd_t::post_ops_ok(attr)) return false;

            auto &po = attr->post_ops_;

            // Binary is not supported.
            if (po.find(primitive_kind::binary) != -1) return false;

            for (int i = 0; i < po.len(); i++) {
                if (po.entry_[i].is_eltwise()) {
                    if (!jit_eltwise_injector_f32<ngen::HW::Gen12HP>::
                                    is_supported(po.entry_[i].eltwise.alg))
                        return false;
                }
            }
            return true;
        }

        status_t init_conf(engine_t *engine);

        conv_conf_t conf;
    };

    using gpu_primitive_t::gpu_primitive_t;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

protected:
    status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const override {
        return init_output_scales_res_storage(engine, r, OSCALES_);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;

    const int OSCALES_ = 0;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

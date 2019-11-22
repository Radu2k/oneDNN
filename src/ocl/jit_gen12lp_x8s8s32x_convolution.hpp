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

#ifndef JIT_GEN12LP_X8S8S32X_CONVOLUTION_HPP
#define JIT_GEN12LP_X8S8S32X_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_gen12lp_x8s8s32x_conv_kernel.hpp"
#include "ocl/ocl_convolution_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

extern const char *gen12lp_conv_fwd_data_x8s8s32x_kernel[];
extern const char *gen12lp_conv_bwd_data_x8s8s32x_kernel[];
extern const char *gen12lp_conv_dw_fwd_data_x8s8s32x_kernel[];

namespace dnnl {
namespace impl {
namespace ocl {

struct jit_gen12lp_x8s8s32x_convolution_fwd_t : public primitive_impl_t {
    struct pd_t : public ocl_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : ocl_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                "ocl:gen12lp", jit_gen12lp_x8s8s32x_convolution_fwd_t);

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

            status_t status = jit_gen12lp_x8s8s32x_conv_fwd_kernel::init_conf(
                    jcp_, *this->desc(), *this->src_md(), *this->weights_md(),
                    *this->dst_md(), *this->weights_md(1), *this->attr());
            if (status != status::success) return status;

            ok = set_default_formats_common(
                    jcp_.src_tag, jcp_.wei_tag, jcp_.dst_tag);
            return ok ? status::success : status::unimplemented;
        }
        jit_conv_conf_t jcp_;
    };

    status_t init() override {
        const char *kernel_name = nullptr;
        if (pd()->jcp_.is_depthwise) {
            kernel_name = "conv_dw_fwd_x8s8s32x_kernel";
        } else {
            if (pd()->jcp_.ic > 4) {
                if (pd()->jcp_.mb_block == 32)
                    kernel_name = "conv_fwd_mb_block_x8s8s32x_kernel";
                else
                    kernel_name = "conv_fwd_ow_block_x8s8s32x_kernel";
            } else {
                kernel_name = "conv_fwd_first_x8s8s32x_kernel";
            }
        }
        compute::kernel_ctx_t kernel_ctx;
        auto status = jit_gen12lp_x8s8s32x_conv_fwd_kernel::init_const_def(
                kernel_ctx, pd()->jcp_);
        if (status != status::success) return status;

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute_engine->create_kernel(&kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    jit_gen12lp_x8s8s32x_convolution_fwd_t(const pd_t *apd)
        : primitive_impl_t(apd) {
        ker_ = new jit_gen12lp_x8s8s32x_conv_fwd_kernel(pd()->jcp_);
    }

    ~jit_gen12lp_x8s8s32x_convolution_fwd_t() { delete ker_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    jit_gen12lp_x8s8s32x_conv_fwd_kernel *ker_;
    compute::kernel_t kernel_;
};

struct jit_gen12lp_x8s8s32x_convolution_bwd_data_t : public primitive_impl_t {
    struct pd_t : public ocl_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : ocl_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                "ocl:gen12lp", jit_gen12lp_x8s8s32x_convolution_bwd_data_t);

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

            status_t status
                    = jit_gen12lp_x8s8s32x_conv_bwd_data_kernel::init_conf(jcp_,
                            *this->desc(), *this->diff_src_md(),
                            *this->weights_md(), *this->diff_dst_md(),
                            *this->weights_md(1), *this->attr());
            if (status != status::success) return status;

            ok = set_default_formats_common(
                    jcp_.src_tag, jcp_.wei_tag, jcp_.dst_tag);
            return ok ? status::success : status::unimplemented;
        }
        jit_conv_conf_t jcp_;

        bool support_bias() const override { return true; }
    };

    status_t init() override {
        const char *kernel_name = "conv_bwd_data_x8s8s32x_kernel";
        compute::kernel_ctx_t kernel_ctx;
        auto status = jit_gen12lp_x8s8s32x_conv_bwd_data_kernel::init_const_def(
                kernel_ctx, pd()->jcp_);
        if (status != status::success) return status;

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute_engine->create_kernel(&kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    jit_gen12lp_x8s8s32x_convolution_bwd_data_t(const pd_t *apd)
        : primitive_impl_t(apd) {
        ker_ = new jit_gen12lp_x8s8s32x_conv_bwd_data_kernel(pd()->jcp_);
    }

    ~jit_gen12lp_x8s8s32x_convolution_bwd_data_t() { delete ker_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    jit_gen12lp_x8s8s32x_conv_bwd_data_kernel *ker_;
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif

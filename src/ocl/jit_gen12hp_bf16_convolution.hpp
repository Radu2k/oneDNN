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

#ifndef JIT_GEN12HP_BF16_CONVOLUTION_HPP
#define JIT_GEN12HP_BF16_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_gen12hp_bf16_conv_kernel.hpp"
#include "ocl/ocl_convolution_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

extern const char gen12hp_conv_bwd_wht_bf16_kernel;

namespace dnnl {
namespace impl {
namespace ocl {

struct jit_gen12hp_bf16_convolution_bwd_weights_t : public primitive_impl_t {
    struct pd_t : public ocl_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : ocl_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                "ocl:gen12hp", jit_gen12hp_bf16_convolution_bwd_weights_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine()->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());

            bool ok = true && desc()->prop_kind == backward_weights
                    && desc()->alg_kind == alg_kind::convolution_direct
                    && (expect_data_types(bf16, bf16, bf16, bf16, f32)
                            || expect_data_types(bf16, bf16, f32, bf16,
                                    f32) //bf16 wei, f32 bias
                            || expect_data_types(bf16, f32, bf16, bf16,
                                    f32) //f32 wei, bf16 bias
                            || expect_data_types(bf16, f32, f32, bf16,
                                    f32)) //f32 wei, f32 bias
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && attr()->has_default_values();

            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            status_t status
                    = jit_gen12hp_bf16_conv_bwd_weights_kernel::init_conf(jcp_,
                            *desc(), *src_md(), *diff_weights_md(),
                            *diff_dst_md(), *diff_weights_md(1), *attr(),
                            scratchpad);
            if (status != status::success) return status;

            ok = set_default_formats_common(
                    jcp_.src_tag, jcp_.wei_tag, jcp_.dst_tag);
            set_offsets(src_md(), off_.src_off);
            set_offsets(diff_weights_md(), off_.wht_off);
            set_offsets(diff_dst_md(), off_.dst_off);
            return ok ? status::success : status::unimplemented;
        }
        jit_conv_conf_t jcp_;
        jit_offsets off_;
    };

    status_t init() override {
        std::vector<const char *> kernel_names {
                "gen12hp_conv_bwd_wht_kernel_bf16",
                "gen12hp_wht_f32_zero_init_kernel"};
        if (pd()->jcp_.weights_data_type == data_type::bf16)
            kernel_names.push_back("gen12hp_wht_convert_f32_to_bf16_kernel");

        compute::kernel_ctx_t kernel_ctx;
        auto status = jit_gen12hp_bf16_conv_bwd_weights_kernel::init_const_def(
                kernel_ctx, pd()->jcp_, pd()->off_);
        if (status != status::success) return status;

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        std::vector<compute::kernel_t> kernels;
        CHECK(compute_engine->create_kernels(
                &kernels, kernel_names, kernel_ctx));
        conv_kernel_ = kernels[0];
        zero_init_kernel_ = kernels[1];

        if (pd()->jcp_.weights_data_type == data_type::bf16)
            convert_f32_to_bf16_kernel_ = kernels[2];

        return status::success;
    }

    jit_gen12hp_bf16_convolution_bwd_weights_t(const pd_t *apd)
        : primitive_impl_t(apd) {
        ker_ = new jit_gen12hp_bf16_conv_bwd_weights_kernel(pd()->jcp_);
    }

    ~jit_gen12hp_bf16_convolution_bwd_weights_t() { delete ker_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    jit_gen12hp_bf16_conv_bwd_weights_kernel *ker_;
    compute::kernel_t conv_kernel_;
    compute::kernel_t zero_init_kernel_;
    compute::kernel_t convert_f32_to_bf16_kernel_;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif

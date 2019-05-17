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

#ifndef JIT_GEN12LP_U8S8S32U8_1x1_CONVOLUTION_HPP
#define JIT_GEN12LP_U8S8S32U8_1x1_CONVOLUTION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "ocl/cl_engine.hpp"
#include "ocl/ocl_convolution_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"
#include "ocl/jit_gen12lp_u8s8s32u8_1x1_conv_kernel.hpp"

extern const char *gen12lp_1x1_conv_fwd_data_u8s8s32x_kernel;

namespace mkldnn {
namespace impl {
namespace ocl {

template <impl::data_type_t dst_type>
struct jit_gen12lp_u8s8s32u8_1x1_convolution_fwd_t: public primitive_t {
    struct pd_t: public ocl_convolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : ocl_convolution_fwd_pd_t(engine, adesc, attr,
                    hint_fwd_pd), jcp_()
        {}

        DECLARE_COMMON_PD_T("ocl:gen12lp",
            jit_gen12lp_u8s8s32u8_1x1_convolution_fwd_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;
            assert(this->engine()->kind() == engine_kind::gpu);

            bool ok = true
                && utils::one_of(this->desc()->prop_kind, forward_training,
                    forward_inference)
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && this->desc()->src_desc.data_type == data_type::u8
                && this->desc()->weights_desc.data_type == data_type::s8
                && this->desc()->accum_data_type == data_type::s32
                && this->desc()->dst_desc.data_type == dst_type
                && IMPLICATION(this->with_bias(),
                    true && this->desc()->bias_desc.data_type == f32);
            if (!ok)
                return status::unimplemented;

            status_t status
                = jit_gen12lp_u8s8s32u8_1x1_conv_fwd_kernel::init_conf(jcp_,
                    *this->desc(), *this->src_md(), *this->weights_md(),
                    *this->dst_md(), *this->weights_md(1),
                    *this->attr());
            if (status != status::success)
                return status;

            ok = set_default_formats_common(
                jcp_.src_tag, jcp_.wei_tag, jcp_.dst_tag);
            return ok ? status::success : status::unimplemented;
        }

        jit_conv_conf_t jcp_;
    };

    status_t init() override {
        auto jit = ocl_jit_t(gen12lp_1x1_conv_fwd_data_u8s8s32x_kernel);
        auto status = jit_gen12lp_u8s8s32u8_1x1_conv_fwd_kernel::init_const_def(
            jit, pd()->jcp_);
        if (status != status::success) return status;

        status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("gen12lp_1x1_conv_fwd_kernel");
        if (!kernel_)
            return status::runtime_error;

        return status::success;
    }

    jit_gen12lp_u8s8s32u8_1x1_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd) {
        ker_ = new jit_gen12lp_u8s8s32u8_1x1_conv_fwd_kernel(pd()->jcp_);
    }

    ~jit_gen12lp_u8s8s32u8_1x1_convolution_fwd_t() {
        delete ker_;
    }

    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_gen12lp_u8s8s32u8_1x1_conv_fwd_kernel *ker_;
    ocl_kernel_t kernel_;
};

}
}
}

#endif

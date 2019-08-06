/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CPU_REF_POOLING_HPP
#define CPU_REF_POOLING_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_isa_traits.hpp"
#include "cpu_pooling_pd.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type, impl::data_type_t acc_type = data_type>
struct ref_pooling_fwd_t : public primitive_impl_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_pooling_fwd_t);

        status_t init() {
            bool ok = true
                    && IMPLICATION(
                            data_type == data_type::bf16, mayiuse(avx512_core))
                    && set_default_params() == status::success && is_fwd()
                    && utils::everyone_is(
                            data_type, src_md()->data_type, dst_md()->data_type)
                    && desc()->accum_data_type == acc_type
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == prop_kind::forward_training;
            if (desc()->alg_kind == alg_kind::pooling_max && is_training)
                init_default_ws();

            return status::success;
        }
    };

    ref_pooling_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    typedef typename prec_traits<data_type>::type data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

template <impl::data_type_t data_type>
struct ref_pooling_bwd_t : public primitive_impl_t {
    struct pd_t : public cpu_pooling_bwd_pd_t {
        using cpu_pooling_bwd_pd_t::cpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_pooling_bwd_t);

        status_t init() {
            bool ok = true
                    && IMPLICATION(
                            data_type == data_type::bf16, mayiuse(avx512_core))
                    && set_default_params() == status::success && !is_fwd()
                    && utils::everyone_is(data_type, diff_dst_md()->data_type,
                            diff_src_md()->data_type)
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            if (desc()->alg_kind == alg_kind::pooling_max) {
                init_default_ws();
                if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            }

            return status::success;
        }
    };

    ref_pooling_bwd_t(const pd_t *apd) : primitive_impl_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward(ctx);
        return status::success;
    }

private:
    void execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

} // namespace cpu
} // namespace impl
} // namespace mkldnn

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

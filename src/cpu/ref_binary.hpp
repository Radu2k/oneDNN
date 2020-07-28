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

#ifndef CPU_REF_BINARY_HPP
#define CPU_REF_BINARY_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/ref_eltwise.hpp"

#include "cpu/cpu_binary_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

float compute_binary_scalar(alg_kind_t alg, float x, float y);

struct ref_binary_scalar_t {
public:
    ref_binary_scalar_t(alg_kind_t alg) : alg_(alg) {
        assert(utils::one_of(alg_, alg_kind::binary_add, alg_kind::binary_max,
                alg_kind::binary_min, alg_kind::binary_mul));
    }

    ref_binary_scalar_t(const post_ops_t::entry_t::binary_t &binary)
        : ref_binary_scalar_t(binary.alg) {}

    template <typename src0_data_t = float, typename src1_data_t = src0_data_t,
            typename dst_data_t = src0_data_t>
    dst_data_t compute_scalar(src0_data_t src0, src1_data_t src1) {
        return (dst_data_t)compute_binary_scalar(
                alg_, (float)src0, (float)src1);
    }

    const alg_kind_t alg_;
};

template <data_type_t src0_type, data_type_t src1_type = src0_type,
        data_type_t dst_type = src0_type>
struct ref_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_binary_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            bool ok = src0_type == src_md(0)->data_type
                    && src1_type == src_md(1)->data_type
                    && dst_type == dst_md()->data_type
                    && platform::has_data_type_support(src0_type)
                    && platform::has_data_type_support(src1_type)
                    && platform::has_data_type_support(dst_type)
                    && set_default_params() == status::success
                    && IMPLICATION(utils::one_of(src0_type, f32, bf16),
                            attr()->has_default_values(sm::post_ops))
                    && IMPLICATION(utils::one_of(src0_type, s8, u8),
                            attr()->has_default_values(
                                    sm::post_ops | sm::scales))
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            check_scales_mask());
            if (!ok) return status::unimplemented;

            return status::success;
        }

    private:
        bool check_scales_mask() const {
            for (const auto &s : attr()->scales_.scales_) {
                if (s.second.mask_ != 0) return false;
            }
            return true;
        }
    };

    ref_binary_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        using namespace primitive_kind;
        const auto &po = pd()->attr()->post_ops_;
        for (auto idx = 0; idx < po.len(); ++idx) {
            if (po.contain(eltwise, idx))
                eltwise_ker_.push_back(
                        utils::make_unique<ref_eltwise_scalar_fwd_t>(
                                po.entry_[idx].eltwise));
            else if (po.contain(binary, idx))
                binary_ker_.push_back(utils::make_unique<ref_binary_scalar_t>(
                        po.entry_[idx].binary));
        }
        return status::success;
    }

    using src0_data_t = typename prec_traits<src0_type>::type;
    using src1_data_t = typename prec_traits<src1_type>::type;
    using dst_data_t = typename prec_traits<dst_type>::type;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    status_t execute_ref(const exec_ctx_t &ctx) const;
    std::vector<std::unique_ptr<ref_eltwise_scalar_fwd_t>> eltwise_ker_;
    std::vector<std::unique_ptr<ref_binary_scalar_t>> binary_ker_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

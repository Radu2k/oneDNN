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

#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/ref_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace alg_kind;

float compute_binary_scalar(alg_kind_t alg, float x, float y) {
    switch (alg) {
        case binary_add: return x + y;
        case binary_max: return nstl::max(x, y);
        case binary_min: return nstl::min(x, y);
        case binary_mul: return x * y;
        default: assert(!"not supported operation!"); return NAN;
    }
}

inline float cast_to_dt(data_type_t dt, const void *ptr, dim_t idx) {
#define CASE(dt) \
    case dt: return (float)(((typename prec_traits<dt>::type *)ptr)[idx]);

    using namespace data_type;
    switch (dt) {
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        default: assert(!"bad data_type");
    }

#undef CASE
    return 0;
}

template <data_type_t src0_type, data_type_t src1_type, data_type_t dst_type>
void ref_binary_t<src0_type, src1_type, dst_type>::execute_ref(
        const exec_ctx_t &ctx) const {
    const auto src0 = CTX_IN_MEM(const src0_data_t *, DNNL_ARG_SRC_0);
    const auto src1 = CTX_IN_MEM(const src1_data_t *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto alg = pd()->desc()->alg_kind;

    // 0:src0 1:src1
    constexpr int nargs = 2;
    scales_t scales[nargs];
    int args[nargs] = {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1};

    if (nstl::is_integral<src0_data_t>::value)
        scales[0] = pd()->attr()->scales_.get(args[0]);

    if (nstl::is_integral<src0_data_t>::value)
        scales[1] = pd()->attr()->scales_.get(args[1]);

    bool do_scale_src0 = !scales[0].has_default_values();
    bool do_scale_src1 = !scales[1].has_default_values();

    const auto nelems_A = src0_d.nelems();

    const auto &po = pd()->attr()->post_ops_;
    const auto sum_idx = po.find(primitive_kind::sum);
    const bool do_sum = sum_idx != -1 && po.entry_[sum_idx].sum.scale != 0.f;
    const float sum_scale = do_sum ? po.entry_[sum_idx].sum.scale : 0.f;

    auto get_mask = [&](const dims_t &src1_dims) {
        const auto &src0_dims = src0_d.dims();

        int broadcast_mask = 0;
        for (int d = 0; d < src0_d.ndims(); ++d)
            broadcast_mask += src0_dims[d] == src1_dims[d] ? (1 << d) : 0;
        return broadcast_mask;
    };

    parallel_nd(nelems_A, [&](dim_t i) {
        auto off_A = src0_d.off_l(i);
        auto off_B = src0_d.off_m(src1_d, i, get_mask(src1_d.dims()));
        auto off_C = dst_d.off_l(i);

        float x_f = (float)src0[off_A];
        float y_f = (float)src1[off_B];
        float dst_f = (float)dst[off_C];

        if (do_scale_src0) x_f *= scales[0].scales_[0];
        if (do_scale_src1) y_f *= scales[1].scales_[0];

        float acc = compute_binary_scalar(alg, x_f, y_f);

        for (auto idx = 0; idx < po.len_; ++idx) {
            using namespace primitive_kind;
            const auto &e = po.entry_[idx];
            switch (e.kind) {
                case sum: acc += sum_scale * dst_f; break;
                case eltwise:
                    acc = eltwise_ker_[idx]->compute_scalar(acc);
                    break;
                case binary: {
                    const auto &b = e.binary;
                    const memory_desc_wrapper po_src1_d(b.src1_desc);
                    auto off_po = src0_d.off_m(
                            po_src1_d, i, get_mask(po_src1_d.dims()));
                    const auto attr_po_b = CTX_IN_MEM(
                            const void *, DNNL_ARG_ATTR_POST_OP_0 + idx);
                    auto val_po = cast_to_dt(
                            po_src1_d.data_type(), attr_po_b, off_po);
                    acc = binary_ker_[idx]->compute_scalar(acc, val_po);
                } break;
                default: assert("unsupported post op primitive kind!"); break;
            }
        }
        dst[off_C] = qz_a1b0<float, dst_data_t>()(acc);
    });
}

using namespace data_type;

template struct ref_binary_t<f32>;
template struct ref_binary_t<bf16>;
template struct ref_binary_t<s8, u8, s8>;
template struct ref_binary_t<s8, s8, s8>;
template struct ref_binary_t<u8, s8, u8>;
template struct ref_binary_t<u8, u8, u8>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

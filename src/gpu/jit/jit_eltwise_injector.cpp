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

#include "gpu/jit/jit_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

template <gpu_gen_t hw>
int jit_eltwise_injector_f32<hw>::min_scratch_regs() {
    using namespace alg_kind;
    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu: return (alpha_ == 0.f) ? 0 : 1;
            case eltwise_abs: return 0;
            case eltwise_square: return 0;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu: return 1;
            case eltwise_abs: return 1;
            case eltwise_square: return 0;
            default: assert(!"unsupported eltwise algorithm");
        }
    }
    return 0;
}

template <gpu_gen_t hw>
int jit_eltwise_injector_f32<hw>::preferred_scratch_regs() {
    using namespace alg_kind;
    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu: return (alpha_ == 0.f) ? 0 : 8;
            default: break;
        }
    }
    return min_scratch_regs();
}

template <gpu_gen_t hw>
int jit_eltwise_injector_f32<hw>::max_batch_size() {
    using namespace alg_kind;
    auto ss = scratch_.getLen();

    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu:
                if (alpha_ == 0.)
                    break;
                else
                    return ss;
            default: break;
        }
    }

    return 128;
}

template <gpu_gen_t hw>
int jit_eltwise_injector_f32<hw>::phase_count() {
    using namespace alg_kind;

    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu: return (alpha_ == 0) ? 1 : 2;
            default: break;
        }
    } else {
        switch (alg_) {
            case eltwise_abs: return 2;
            default: break;
        }
    }

    return 1;
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::relu_zero_ns_compute_fwd(
        int simd, const ngen::GRF &r) {
    h->max_(simd, r, r, 0.f);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::relu_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    auto temp = scratch_[off].f();
    switch (phase) {
        case 0: h->mul(simd, temp, r, alpha_); break;
        case 1: h->csel(simd | le | f0[0], r, temp, r, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::abs_compute_fwd(
        int simd, const ngen::GRF &r) {
    h->mov(simd, r, abs(r));
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::square_compute_fwd(
        int simd, const ngen::GRF &r) {
    h->mul(simd, r, r, r);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::relu_prepare_bwd() {
    auto neg_slope = scratch_[0].f(0);
    auto pos_slope = scratch_[0].f(4);
    h->mov(1, neg_slope, alpha_);
    h->mov(1, pos_slope, 1.f);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::relu_compute_bwd(
        int simd, const ngen::GRF &r) {
    auto neg_slope = scratch_[0].f(0);
    auto pos_slope = scratch_[0].f(4);
    h->csel(simd | le | f0[0], r, neg_slope, pos_slope, r);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::abs_prepare_bwd() {
    auto neg_one = scratch_[0].f(0);
    auto pos_one = scratch_[0].f(4);
    h->mov(1, neg_one, -1.f);
    h->mov(1, pos_one, 1.f);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::abs_compute_bwd(
        int simd, const ngen::GRF &r, int phase) {
    auto neg_one = scratch_[0].f(0);
    auto pos_one = scratch_[0].f(4);
    switch (phase) {
        case 0: h->csel(simd | lt | f0[0], r, neg_one, r, r); break;
        case 1: h->csel(simd | gt | f0[0], r, pos_one, r, r); break;
        default: break;
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::square_compute_bwd(
        int simd, const ngen::GRF &r) {
    h->add(simd, r, r, r);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::compute(const ngen::GRFRange &regs) {
    using namespace alg_kind;

    auto bmax = max_batch_size();
    auto phases = phase_count();

    for (int idx0 = 0; idx0 < regs.getLen(); idx0 += bmax) {
        auto batch = nstl::min(regs.getLen() - idx0, bmax);

        for (int phase = 0; phase < phases; phase++) {
            for (int ii = 0; ii < batch; ii += 2) {
                int nreg = nstl::min(2, batch - ii);
                int simd = nreg * 8;
                auto base = regs[idx0 + ii].f();

                if (is_fwd_) {
                    switch (alg_) {
                        case eltwise_relu:
                            if (alpha_ == 0.f)
                                relu_zero_ns_compute_fwd(simd, base);
                            else
                                relu_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_abs: abs_compute_fwd(simd, base); break;
                        case eltwise_square:
                            square_compute_fwd(simd, base);
                            break;
                        default: assert(!"unsupported eltwise algorithm");
                    }
                } else {
                    switch (alg_) {
                        case eltwise_relu: relu_compute_bwd(simd, base); break;
                        case eltwise_abs:
                            abs_compute_bwd(simd, base, phase);
                            break;
                        case eltwise_square:
                            square_compute_bwd(simd, base);
                            break;
                        default: assert(!"unsupported eltwise algorithm");
                    }
                }
            }
        }
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::prepare() {
    using namespace alg_kind;

    assert(scratch_.getLen() >= min_scratch_regs());

    if (is_fwd_) {
        /* nothing to do */
    } else {
        switch (alg_) {
            case eltwise_relu: relu_prepare_bwd(); break;
            case eltwise_abs: abs_prepare_bwd(); break;
            default: break;
        }
    }
}

template struct jit_eltwise_injector_f32<gpu_gen12hp>;

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

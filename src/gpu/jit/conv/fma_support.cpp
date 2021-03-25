/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "gpu/jit/conv/fma_support.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

type_t multiply_desc_t::_c_type(
        const type_t &a, const type_t &b, bool force_c_upconvert) {
    if (utils::one_of(a, type_t::s8(), type_t::u8())
            && utils::one_of(b, type_t::s8(), type_t::u8()))
        return type_t::s32();

    if (a == type_t::bf16() && b == type_t::bf16()) return type_t::f32();
    if (a == type_t::f16() && b == type_t::f16()) {
        if (force_c_upconvert) return type_t::f32();
        return type_t::f16();
    }
    if (a == type_t::f32() && b == type_t::f32()) return type_t::f32();

    ir_error_not_expected()
            << "Can't deduce C type. A type: " << a << " B type: " << b;
    return type_t::undef();
}

layout_t dpas_t::a_layout() const {
    if (src1_type.size() == 1) return layout_t(src1_type, 0, "8b8a4b");
    if (src1_type.size() == 2) return layout_t(src1_type, 0, "8b8a2b");
    ir_error_not_expected();
    return layout_t();
}

layout_t dpas_t::b_layout() const {
    if (src2_type.size() == 1) return layout_t(src2_type, 0, "8b32a");
    if (src2_type.size() == 2) return layout_t(src2_type, 0, "8b16a");
    ir_error_not_expected();
    return layout_t();
}

layout_t dpas_t::c_layout() const {
    return layout_t(dst_type, 0, "8b8a");
}

bool dpas_t::matches(const multiply_desc_t &desc) const {
    int m_blk = 8;
    int n_blk = rcount;
    int k_blk = sdepth * 4 / src1_type.size();

    if (!dpas_t::matches_types(desc.a_type(), desc.b_type(), desc.c_type()))
        return false;
    if (desc.k() != k_blk) return false;
    if (desc.m() % m_blk != 0 || desc.n() % n_blk != 0) return false;

    auto a_blk_layout = desc.a_layout().map(tensor_t({m_blk, desc.k()}));
    auto b_blk_layout = desc.b_layout().map(tensor_t({desc.k(), n_blk}));

    if (a_blk_layout != a_layout()) return false;
    if (b_blk_layout != b_layout()) return false;

    return true;
}

bool dpas_t::matches_types(const type_t &a, const type_t &b, const type_t &c) {
    if (a.is_x8() && b.is_x8() && c.is_s32()) return true;
    if (a.is_f16() && b.is_f16() && c.is_f32()) return true;
    if (a.is_bf16() && b.is_bf16() && c.is_f32()) return true;

    return false;
}

    return false;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

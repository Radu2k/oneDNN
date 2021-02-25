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

#ifndef GPU_JIT_CONV_FMA_SUPPORT_HPP
#define GPU_JIT_CONV_FMA_SUPPORT_HPP

#include <sstream>
#include <string>

#include "gpu/jit/conv/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class multiply_desc_t {
public:
    multiply_desc_t() = default;

    multiply_desc_t(const layout_t &a_layout, const layout_t &b_layout,
            bool force_c_upconvert)
        : a_layout_(a_layout), b_layout_(b_layout) {
        ir_assert(a_layout.ndims() == 2 && b_layout.ndims() == 2)
                << "Expected 2D layouts, A layout: " << a_layout
                << " B layout: " << b_layout;

        c_type_ = _c_type(a_type(), b_type(), force_c_upconvert);
    }

    const layout_t &a_layout() const { return a_layout_; }
    const layout_t &b_layout() const { return b_layout_; }

    const type_t &a_type() const { return a_layout_.type(); }
    const type_t &b_type() const { return b_layout_.type(); }
    const type_t &c_type() const { return c_type_; }

    int m() const { return a_layout_.dims()[0]; }
    int n() const { return b_layout_.dims()[1]; }
    int k() const { return a_layout_.dims()[1]; }

private:
    static type_t _c_type(
            const type_t &a, const type_t &b, bool force_c_upconvert);

    layout_t a_layout_;
    layout_t b_layout_;
    type_t c_type_;
};

// Function representing DPAS instruction.
class dpas_t : public func_impl_t {
public:
    IR_DECL_DERIVED_TYPE_ID(dpas_t, func_impl_t)

    static func_t make(bool is_dpasw, int sdepth, int rcount,
            const type_t &dst_type, const type_t &src1_type,
            const type_t &src2_type) {
        return func_t(new dpas_t(
                is_dpasw, sdepth, rcount, dst_type, src1_type, src2_type));
    }

    static func_t make_dpasw(const dpas_t &dpas) {
        return func_t(new dpas_t(true, dpas.sdepth, dpas.rcount, dpas.dst_type,
                dpas.src1_type, dpas.src2_type));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (is_dpasw == other.is_dpasw) && (sdepth == other.sdepth)
                && (rcount == other.rcount) && (dst_type == other.dst_type)
                && (src1_type == other.src1_type)
                && (src2_type == other.src2_type);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(
                is_dpasw, sdepth, rcount, dst_type, src1_type, src2_type);
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << (is_dpasw ? "dpasw" : "dpas");
        oss << "." << sdepth << "x" << rcount;
        return oss.str();
    }

    IR_DEFINE_ARG_GET(dst, 0)
    IR_DEFINE_ARG_GET(src0, 1)
    IR_DEFINE_ARG_GET(src1, 2)
    IR_DEFINE_ARG_GET(src2, 3)

    stmt_t operator()(const expr_t &dst, const expr_t &src0, const expr_t &src1,
            const expr_t &src2) const {
        return call({dst, src0, src1, src2});
    }

    int dst_size() const { return simd_size * rcount * sizeof(uint32_t); }
    int src0_size() const { return dst_size(); }
    int src1_size() const { return simd_size * sdepth * sizeof(uint32_t); }
    int src2_size() const {
        int dpas_size = sdepth * rcount * sizeof(uint32_t);
        return is_dpasw ? dpas_size / 2 : dpas_size;
    }

    layout_t a_layout() const;
    layout_t b_layout() const;
    layout_t c_layout() const;

    bool matches(const multiply_desc_t &desc) const;

    static bool matches_types(
            const type_t &a, const type_t &b, const type_t &c);

    static const int simd_size = 8;

    bool is_dpasw;

    int sdepth;
    int rcount;

    type_t dst_type; // src0 type is same as dst_type.
    type_t src1_type;
    type_t src2_type;

private:
    dpas_t(bool is_dpasw, int sdepth, int rcount, const type_t &dst_type,
            const type_t &src1_type, const type_t &src2_type)
        : is_dpasw(is_dpasw)
        , sdepth(sdepth)
        , rcount(rcount)
        , dst_type(dst_type)
        , src1_type(src1_type)
        , src2_type(src2_type) {}
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

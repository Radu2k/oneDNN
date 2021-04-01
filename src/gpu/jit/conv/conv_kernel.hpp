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

#ifndef GPU_JIT_CONV_CONV_KERNEL_HPP
#define GPU_JIT_CONV_CONV_KERNEL_HPP

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/fma_support.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/kernel_arg_info.hpp"
#include "gpu/jit/conv/kernel_builder.hpp"
#include "gpu/jit/conv/message_support.hpp"
#include "gpu/jit/conv/ngen_proxy.hpp"
#include "gpu/jit/conv/post_op_support.hpp"
#include "gpu/jit/jit_eltwise_injector.hpp"
#include "gpu/jit/jit_generator.hpp"
#include "gpu/jit/ngen/ngen_core.hpp"
#include "gpu/jit/ngen/ngen_register_allocator.hpp"

#include "gpu/jit/gemm/emulation.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

template <typename T>
T to_cpp(const ngen::Immediate &imm) {
    auto u64 = uint64_t(imm);
    switch (imm.getType()) {
        case ngen::DataType::w:
            return (T)utils::bit_cast<std::array<int16_t, 4>>(u64)[0];
        case ngen::DataType::uw:
            return (T)utils::bit_cast<std::array<uint16_t, 4>>(u64)[0];
        case ngen::DataType::d:
            return (T)utils::bit_cast<std::array<int32_t, 2>>(u64)[0];
        case ngen::DataType::ud:
            return (T)utils::bit_cast<std::array<uint32_t, 2>>(u64)[0];
        case ngen::DataType::q: return (T)utils::bit_cast<int64_t>(u64);
        case ngen::DataType::uq: return (T)utils::bit_cast<uint64_t>(u64);
        default: ir_error_not_expected();
    }
    return 0;
}

// type_t to ngen::DataType convertor.
ngen::DataType to_ngen(const type_t &type) {
    ir_assert(type.is_scalar()) << "Expected scalar type.";

#define CASE(_kind, ngen_enum) \
    if (type.kind() == type_kind_t::_kind) return ngen::DataType::ngen_enum

    CASE(bf16, bf);
    CASE(f16, hf);
    CASE(f32, f);
    CASE(s16, w);
    CASE(s32, d);
    CASE(s64, q);
    CASE(s8, b);
    CASE(u16, uw);
    CASE(u32, ud);
    CASE(u64, uq);
    CASE(u8, ub);

    if (type == type_t::byte_ptr()) return ngen::DataType::uq;

#undef CASE
    ir_error_not_expected();
    return ngen::DataType::invalid;
}

ngen::Immediate to_ngen(
        const expr_t &expr, const type_t &type = type_t::undef()) {
    ir_assert(expr.type().is_scalar()) << "Vector types are not supported.";
    if (expr.is<int_imm_t>()) {
        auto &imm = expr.as<int_imm_t>();
        // No conversion.
        if (utils::one_of(type, type_t::undef(), expr.type()))
            return ngen::Immediate(imm.value);
            // Do conversion.
#define CASE(cpp_type) \
    if (type.is_cpp<cpp_type>()) return ngen::Immediate(cpp_type(imm.value))

        CASE(int16_t);
        CASE(int32_t);
        CASE(int64_t);
        CASE(uint16_t);
        CASE(uint32_t);
        CASE(uint64_t);

#undef CASE
        ir_error_not_expected() << "Can't convert expression: " << expr;
    } else if (expr.is<float_imm_t>()) {
        ir_assert(utils::one_of(type, type_t::undef(), type_t::f32()))
                << "Conversion is not supported.";
        auto &imm = expr.as<float_imm_t>();
        return ngen::Immediate(imm.value);
    }
    ir_error_not_expected() << "Can't convert expression: " << expr;
    return ngen::Immediate();
}

ngen::Bundle to_ngen(const ngen_proxy::Bundle &bundle) {
    return ngen::Bundle(bundle.bank_id, bundle.bundle_id);
}

ngen::InstructionModifier to_ngen(
        const ngen_proxy::InstructionModifier &mod_proxy) {
    ngen::InstructionModifier mod;
    if (mod_proxy.is_atomic) mod |= ngen::ThreadCtrl::Atomic;
    if (!mod_proxy.sbid.is_empty()) mod |= ngen::SBID(mod_proxy.sbid.token).set;
    return mod;
}

ngen::ConditionModifier cmp_op_to_ngen(op_kind_t op_kind) {
    ir_assert(is_cmp_op(op_kind));
    switch (op_kind) {
        case op_kind_t::_eq: return ngen::ConditionModifier::eq;
        case op_kind_t::_ne: return ngen::ConditionModifier::ne;
        case op_kind_t::_ge: return ngen::ConditionModifier::ge;
        case op_kind_t::_gt: return ngen::ConditionModifier::gt;
        case op_kind_t::_le: return ngen::ConditionModifier::le;
        case op_kind_t::_lt: return ngen::ConditionModifier::lt;
        default: ir_error_not_expected();
    }
    return ngen::ConditionModifier::none;
}

ngen::RegData ngen_reg_data(const ngen::RegData &base, int off_bytes,
        ngen::DataType type, int width, int hstride = 1) {
    auto new_off = base.getByteOffset() + off_bytes;
    auto new_grf_off = (new_off % reg_bytes);
    auto type_size = ngen::getBytes(type);
    auto grf = ngen::GRF(base.getBase() + new_off / reg_bytes).retype(type);

    ir_assert(new_grf_off % type_size == 0);
    MAYBE_UNUSED(new_grf_off);

    if (width == 1) {
        hstride = 0;
    } else if (hstride == 0) {
        ir_assert(width == 1);
    } else {
        int max_width = 32 / type_size;
        width = std::min(width, max_width / hstride);
        width = std::min(width, 16);
    }
    int vstride = width * hstride;
    return grf[new_off / type_size](vstride, width, hstride);
}

ngen::Immediate ngen_negate(const ngen::Immediate &imm) {
    switch (imm.getType()) {
        case ngen::DataType::w: return ngen::Immediate(-to_cpp<int16_t>(imm));
        case ngen::DataType::d: return ngen::Immediate(-to_cpp<int32_t>(imm));
        case ngen::DataType::f: return ngen::Immediate(-to_cpp<float>(imm));
        default: ir_error_not_expected();
    }
    return ngen::Immediate();
}

bool ngen_is_qw(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::q, ngen::DataType::uq);
}

bool ngen_is_dw(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::d, ngen::DataType::ud);
}

bool ngen_is_w(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::w, ngen::DataType::uw);
}

bool ngen_is_xf(ngen::DataType type) {
    return utils::one_of(
            type, ngen::DataType::bf, ngen::DataType::hf, ngen::DataType::f);
}

enum class ngen_operand_kind_t { invalid, immediate, reg_data, flag_register };

// Wrapper to generalize ngen::FlagRegister, ngen::RegData and ngen::Immediate
// operands.
class ngen_operand_t {
public:
    ngen_operand_t() : kind_(ngen_operand_kind_t::invalid) {}

    ngen_operand_t(const ngen::FlagRegister &flag)
        : kind_(ngen_operand_kind_t::flag_register)
        , ptr_(new ngen::FlagRegister(flag),
                  destroy<ngen_operand_kind_t::flag_register>) {}

    ngen_operand_t(const ngen::RegData &reg_data)
        : kind_(ngen_operand_kind_t::reg_data)
        , ptr_(new ngen::RegData(reg_data),
                  destroy<ngen_operand_kind_t::reg_data>) {}

    ngen_operand_t(const ngen::Immediate &imm)
        : kind_(ngen_operand_kind_t::immediate)
        , ptr_(new ngen::Immediate(imm),
                  destroy<ngen_operand_kind_t::immediate>) {}

    template <typename T>
    ngen_operand_t(const T &other, const ngen::InstructionModifier &mod)
        : ngen_operand_t(other) {
        mod_ = mod;
    }

    const ngen::Immediate &immediate() const {
        ir_assert(is_immediate());
        return *(const ngen::Immediate *)ptr_.get();
    }

    const ngen::RegData &reg_data() const {
        ir_assert(is_reg_data());
        return *(const ngen::RegData *)ptr_.get();
    }

    const ngen::FlagRegister &flag_register() const {
        ir_assert(is_flag_register());
        return *(const ngen::FlagRegister *)ptr_.get();
    }

    ngen::InstructionModifier flag_register_mod() const {
        ngen::InstructionModifier mod;
        mod |= flag_register();
        return !is_negated() ? mod : ~mod;
    }

    const ngen::InstructionModifier &mod() const { return mod_; }

    bool is_invalid() const { return kind_ == ngen_operand_kind_t::invalid; }

    bool is_immediate() const {
        return kind_ == ngen_operand_kind_t::immediate;
    }

    bool is_reg_data() const { return kind_ == ngen_operand_kind_t::reg_data; }

    bool is_flag_register() const {
        return kind_ == ngen_operand_kind_t::flag_register;
    }

    bool is_negated() const { return is_negated_; }

    ngen::DataType type() const {
        if (is_immediate()) return immediate().getType();
        if (is_reg_data()) return reg_data().getType();
        ir_error_not_expected();
        return ngen::DataType::invalid;
    }

    ngen_operand_t operator-() const {
        if (is_immediate()) { return ngen_operand_t(ngen_negate(immediate())); }
        if (is_reg_data()) { return ngen_operand_t(-reg_data()); }
        if (is_flag_register()) {
            auto ret = *this;
            ret.is_negated_ = !ret.is_negated_;
            return ret;
        }
        ir_error_not_expected();
        return ngen_operand_t();
    }

    ngen_operand_t reinterpret(const type_t &new_type) const {
        ir_assert(is_reg_data());
        ir_assert(new_type.is_scalar());
        return ngen_reg_data(reg_data(), 0, to_ngen(new_type), 1);
    }

    // Creates an operand with the requested register region based on the
    // existing region. off - offset in elements of the region data type.
    ngen_operand_t sub_reg_data(int off, int exec_size) const {
        ir_assert(is_reg_data());
        auto rd = reg_data();
        int new_base = rd.getBase();
        int new_off = rd.getByteOffset() + off * rd.getBytes() * rd.getHS();
        new_base += (new_off / reg_bytes);
        new_off = (new_off % reg_bytes);

        rd.setBase(new_base);
        rd.setOffset(new_off / rd.getBytes());
        rd.setRegion(0, exec_size, rd.getHS());
        rd.fixup(exec_size, ngen::DataType::invalid, false, 1);
        return ngen_operand_t(rd, exec_size);
    }

private:
    template <ngen_operand_kind_t kind>
    static void destroy(void *ptr) {
        if (!ptr) return;

        switch (kind) {
            case ngen_operand_kind_t::immediate:
                delete (ngen::Immediate *)ptr;
                break;
            case ngen_operand_kind_t::reg_data:
                delete (ngen::RegData *)ptr;
                break;
            case ngen_operand_kind_t::flag_register:
                delete (ngen::FlagRegister *)ptr;
                break;
            default: ir_error_not_expected();
        }
    }

    ngen_operand_kind_t kind_;
    std::shared_ptr<void> ptr_;
    ngen::InstructionModifier mod_;

    // Whether the operand is negated. Applicable to flag registers only.
    // Negation of register data and immediate operands is directly supported
    // through nGEN API.
    bool is_negated_ = false;
};

template <typename T>
T to_cpp(const ngen_operand_t &op) {
    ir_assert(op.is_immediate());
    return to_cpp<T>(op.immediate());
}

// Maintains scoped allocations which are automatically released when the scope
// is destructed.
class ngen_register_scope_t {
public:
    ngen_register_scope_t(ngen::RegisterAllocator &ra) : ra_(&ra) {}

    ngen_register_scope_t(const ngen_register_scope_t &) = delete;

    ngen_register_scope_t(ngen_register_scope_t &&other)
        : ra_(other.ra_)
        , grf_ranges_(std::move(other.grf_ranges_))
        , subregisters_(std::move(other.subregisters_)) {
        other.ra_ = nullptr;
    }

    ~ngen_register_scope_t() {
        for (auto &r : grf_ranges_)
            ra_->safeRelease(r);

        for (auto &s : subregisters_)
            ra_->safeRelease(s);
        for (auto &f : flags_)
            ra_->safeRelease(f);
    }

    ngen::GRFRange alloc_range(
            int regs, ngen::Bundle base_bundle = ngen::Bundle()) {
        auto ret = ra_->alloc_range(regs, base_bundle);
        grf_ranges_.push_back(ret);
        return ret;
    }

    ngen::GRF alloc(ngen::Bundle bundle = ngen::Bundle()) {
        return alloc_range(1, bundle)[0];
    }

    ngen::Subregister alloc_sub(
            ngen::DataType type, ngen::Bundle bundle = ngen::Bundle()) {
        auto ret = ra_->alloc_sub(type, bundle);
        subregisters_.push_back(ret);
        return ret;
    }

    ngen::RegData alloc_reg_data(const type_t &type, int stride_bytes = -1,
            ngen::Bundle bundle = ngen::Bundle()) {
        if (type.is_scalar()) return alloc_sub(to_ngen(type), bundle);

        if (stride_bytes == -1) stride_bytes = type.scalar().size();

        ir_assert(stride_bytes > 0);
        ir_assert(stride_bytes % type.scalar().size() == 0);

        int regs = utils::div_up(type.elems() * stride_bytes, reg_bytes);
        auto sub = alloc_range(regs, bundle)[0].retype(
                to_ngen(type.scalar()))[0];
        auto ret = sub(stride_bytes / type.scalar().size());
        ret.fixup(type.elems(), ngen::DataType::invalid, false, 1);
        return ret;
    }

    ngen::FlagRegister alloc_flag() {
        auto ret = ra_->alloc_flag();
        flags_.push_back(ret);
        return ret;
    }

private:
    ngen::RegisterAllocator *ra_;

    std::vector<ngen::GRFRange> grf_ranges_;
    std::vector<ngen::Subregister> subregisters_;
    std::vector<ngen::FlagRegister> flags_;
};

class expr_binding_t {
public:
    void bind(const expr_t &expr, const ngen_operand_t &operand) {
        bind_unevaluated(expr, operand);
        mark_as_evaluated(expr);
    }

    void bind_unevaluated(const expr_t &expr, const ngen_operand_t &operand) {
        ir_assert(!expr.is_empty());
        auto ret = expr2operand_.insert({expr, operand});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
    }

    void unbind(const expr_t &expr) {
        ir_assert(!expr.is_empty());
        auto it = expr2operand_.find(expr);
        ir_assert(it != expr2operand_.end());
        expr2operand_.erase(it);
    }

    void mark_as_evaluated(const expr_t &expr) {
        ir_assert(!expr.is_empty());
        auto ret = evaluated_.insert(expr);
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
    }

    ngen_operand_t get(const expr_t &expr) const {
        if (expr.is_empty()) return ngen_operand_t();
        return expr2operand_.at(expr);
    }

    bool is_bound(const expr_t &expr) const {
        return expr2operand_.count(expr) == 1;
    }

    bool is_evaluated(const expr_t &expr) const {
        if (!is_bound(expr)) return false;
        return evaluated_.count(expr) == 1;
    }

private:
    object_map_t<expr_t, ngen_operand_t> expr2operand_;
    object_set_t<expr_t> evaluated_;
};

template <ngen::HW hw>
class expr_evaluator_t;

template <ngen::HW hw>
class ir_to_ngen_t;

template <ngen::HW hw>
class conv_kernel_t : public jit_generator<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    friend class expr_evaluator_t<hw>;
    friend class ir_to_ngen_t<hw>;
    friend class send_impl_t;

    conv_kernel_t(const conv_config_t &cfg, const convolution_pd_t *pd,
            kernel_arg_info_t &kernel_arg_info);

    void setup_interface(const stmt_t &kernel_body,
            const kernel_arg_info_t &kernel_arg_info) {
        externalName("gen_conv");
        requireLocalID(3);
        requireLocalSize();
        requireGRF(cfg_.regs);
        requireSIMD(cfg_.simd_size);
        requireBarrier();
        requireDPAS();

        for (int i = 0; i < kernel_arg_info.nargs(); i++) {
            auto &name = kernel_arg_info.arg_name(i);
            auto &type = kernel_arg_info.arg_type(i);
            if (type.is_ptr()) {
                newArgument(name, ngen::ExternalArgumentType::GlobalPtr);
            } else {
                newArgument(name, to_ngen(type));
            }
        }

        int slm_size
                = alloc_manager_t(kernel_body).total_size(alloc_kind_t::slm);
        requireSLM(slm_size);

        finalizeInterface();
    }

    void epilogue() {
        auto tmp = ra_.alloc();
        memfence(tmp);
        mov<uint32_t>(8, null, tmp);

        slmfence(tmp, r0);
        mov<int32_t>(8, null, tmp);

        mov<uint32_t>(8, r255, r0);
        threadend(r255);
        ra_.safeRelease(tmp);
    }

    // Kernel padding for instruction prefetch.
    void pad_kernel() {
        for (int rep = 0; rep < 8; rep++)
            nop();
    }

    void emov(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0) {
        if (dst.is_reg_data()) {
            if (src0.is_reg_data()) {
                emov(mod, dst.reg_data(), src0.reg_data());
            } else if (src0.is_immediate()) {
                emov(mod, dst.reg_data(), src0.immediate());
            } else {
                emov(mod | src0.flag_register_mod(), dst.reg_data(), 1);
                emov(mod | ~src0.flag_register_mod(), dst.reg_data(), 0);
            }
        } else {
            // dst is a flag register.
            ir_assert(!dst.is_negated());
            if (src0.is_reg_data()) {
                emov(mod, dst.flag_register(), src0.reg_data());
            } else {
                emov(mod, dst.flag_register(), src0.immediate());
            }
        }
    }

    void eadd(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            eadd(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            eadd(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void emul(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            emul(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            auto &src1_imm = src1.immediate();
            if (ngen_is_qw(dst.type()) || ngen_is_w(src1_imm.getType())) {
                emul(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
                return;
            }
            if (ngen_is_dw(src1_imm.getType())) {
                ir_assert(mod.getExecSize() == 1);
                auto tmp = ra_.alloc_sub<int64_t>();
                emul(mod, tmp.q(0), src0.reg_data(), src1_imm);
                emov(mod, dst.reg_data(), tmp.reinterpret(0, dst.type()));
                ra_.safeRelease(tmp);
                return;
            }
            emul(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void ediv(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        ir_assert(src1.is_immediate());
        auto &src1_imm = src1.immediate();
        int32_t src1_value = to_cpp<int32_t>(src1_imm);
        ir_assert(0 < src1_value && src1_value <= INT32_MAX) << src1_value;
        eidiv(dst.reg_data(), ngen::Subregister(), src0.reg_data(), src1_value);
    }

    void emod(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        ir_assert(src1.is_immediate());
        auto &src1_imm = src1.immediate();
        int32_t src1_value = to_cpp<int32_t>(src1_imm);
        ir_assert(0 < src1_value && src1_value <= INT32_MAX) << src1_value;
        eidiv(ngen::Subregister(), dst.reg_data(), src0.reg_data(), src1_value);
    }

    void eshl(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            shl(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            shl(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void eshr(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            shr(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            shr(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void emin(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            min_(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            min_(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void emax(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            max_(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            max_(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void ecmp(const ngen::InstructionModifier &mod, const ngen_operand_t &src0,
            const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            cmp(mod, src0.reg_data(), src1.reg_data());
        } else {
            cmp(mod, src0.reg_data(), src1.immediate());
        }
    }

    // Adapted version of magicgu function from Hacker's Delight 10-15.
    static void eidiv_magicgu(uint32_t d, uint32_t &m, uint32_t &p) {
        uint32_t s32_max = std::numeric_limits<int32_t>::max();
        ir_assert(d != 0 && d <= s32_max);
        uint64_t nc = (s32_max / d) * d - 1;
        for (p = 32; p < 64; p++) {
            uint64_t _2p = 1LL << p;
            if (_2p > nc * (d - 1 - (_2p - 1) % d)) {
                m = (_2p + d - 1 - (_2p - 1) % d) / d;
                return;
            }
        }
        ir_error_not_expected();
    }

    // Emulates integer division by a constant.
    // Requirements:
    //     0 <= x <= UINT32_MAX
    //     0 <  y <= INT32_MAX
    // Computes:
    //     qot = x / y
    //     rem = x % y
    void eidiv(const ngen::RegData &qot, const ngen::RegData &rem,
            const ngen::RegData &x, uint32_t y) {
        if (ngen::utils::is_zero_or_pow2(y)) {
            if (!qot.isInvalid()) shr(1, qot, x, ngen::utils::log2(y));
            if (!rem.isInvalid()) and_(1, rem, x, y - 1);
            return;
        }

        uint32_t m, p;
        eidiv_magicgu(y, m, p);

        auto _x = ra_.alloc().ud();
        auto _qot = ra_.alloc().ud();
        mov(1, _x, x);

        // qot = (x * m) >> p
        mul(1, acc0.ud(0), _x, m & 0xFFFF);
        mach(1, _qot, _x, m);
        shr<uint32_t>(1, _qot, _qot, p - 32);
        if (!qot.isInvalid()) mov(1, qot, _qot);

        if (!rem.isInvalid()) {
            // rem = x - qot * y
            bool y_is_16_bit = (y <= static_cast<uint32_t>(
                                        std::numeric_limits<int16_t>::max()));
            if (y_is_16_bit) {
                mad(1, rem, x, _qot, -int16_t(y));
            } else {
                auto tmp = ra_.alloc_sub<uint64_t>();
                mul(1, tmp, _qot, y);
                add(1, rem, x, -tmp.ud(0));
                ra_.safeRelease(tmp);
            }
        }

        ra_.safeRelease(_x);
        ra_.safeRelease(_qot);
    }

    friend struct dnnl::impl::gpu::jit::EmulationImplementation;
    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0) {
        EmulationImplementation::emov<DT>(*this, mod, dst, src0, emu_strategy);
    }
    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::Immediate src0) {
        EmulationImplementation::emov<DT>(*this, mod, dst, src0, emu_strategy);
    }
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        if (ngen_is_xf(dst.getType())) {
            mul(mod, dst, src0, src1);
            return;
        }
        EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1) {
        if (ngen_is_xf(dst.getType())) {
            mul(mod, dst, src0, src1);
            return;
        }
        EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void eshl(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1) {
        EmulationImplementation::eshl<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void eshr(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1) {
        EmulationImplementation::eshr<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }

private:
    void init_kernel_arg_info(kernel_arg_info_t &kernel_arg_info) {
        // TODO: Update for backward.
        ir_assert(cfg_.is_fwd);
        kernel_arg_info.register_user_arg(
                make_buffer("src"), DNNL_ARG_SRC, /*is_input=*/true);
        kernel_arg_info.register_user_arg(
                make_buffer("wei"), DNNL_ARG_WEIGHTS, /*is_input=*/true);
        kernel_arg_info.register_user_arg(
                make_buffer("dst"), DNNL_ARG_DST, /*is_input=*/false);
    }

    const conv_config_t &cfg_;
    ngen::RegisterAllocator ra_;
    ngen::GRF signal_header_;

    EmulationStrategy emu_strategy = EmulationStrategy(hw);
    EmulationState emu_state;
};

// Evaluates expression by emitting instructions with nGEN.
template <ngen::HW hw>
class expr_evaluator_t : public ir_visitor_t {
public:
    expr_evaluator_t(conv_kernel_t<hw> *host,
            const expr_binding_t &expr_binding, ngen_register_scope_t &scope)
        : host_(host), expr_binding_(expr_binding), scope_(scope) {}

    // If `out_operand` is not empty, use its pre-allocated location for the
    // result.
    ngen_operand_t eval(const expr_t &e,
            const ngen_operand_t &out_operand = ngen_operand_t()) {
        if (expr_binding_.is_evaluated(e)) {
            if (!out_operand.is_invalid()) {
                host_->emov(
                        out_operand.mod(), out_operand, expr_binding_.get(e));
            }
        } else {
            if (!out_operand.is_invalid())
                expr_binding_.bind_unevaluated(e, out_operand);
            visit(e);
        }

        return expr_binding_.get(e);
    }

    std::vector<ngen_operand_t> eval(const std::vector<expr_t> &exprs) {
        std::vector<ngen_operand_t> ret;
        for (auto &e : exprs) {
            if (!expr_binding_.is_evaluated(e)) visit(e);
            ret.push_back(expr_binding_.get(e));
        }
        return ret;
    }

    void _visit(const binary_op_t *obj) override {
        auto dst_op = alloc_op(obj);
        auto mod = dst_op.mod();

        switch (obj->op_kind) {
            case op_kind_t::_and: {
                auto src0_op = eval(obj->a, dst_op);
                eval(obj->b,
                        ngen_operand_t(
                                dst_op, mod | src0_op.flag_register_mod()));
                break;
            }
            default: {
                // Some cases require pre-allocated register regions with
                // special strides for a/b.
                auto a_out_op = maybe_alloc_strided_op(obj->type, obj->a);
                auto b_out_op = maybe_alloc_strided_op(obj->type, obj->b);
                auto src0_op = eval(obj->a, a_out_op);
                auto src1_op = eval(obj->b, b_out_op);
                ebinary(obj, mod, dst_op, src0_op, src1_op);
                break;
            }
        }

        expr_binding_.mark_as_evaluated(obj);
    }

    void _visit(const bool_imm_t *obj) override { ir_error_not_implemented(); }

    void _visit(const cast_t *obj) override {
        auto &from_type = obj->expr.type();
        auto &to_type = obj->type;

        ir_assert(from_type != to_type) << "Equal types are not expected.";

        if (is_const(obj->expr)) {
            bind(obj, to_ngen(obj->expr, to_type));
            return;
        }

        auto expr_op = eval(obj->expr);

        // Handle ptr -> u64 and u64 -> ptr casts.
        if (utils::one_of(obj->type, type_t::u64(), type_t::byte_ptr())
                && utils::one_of(
                        obj->expr.type(), type_t::u64(), type_t::byte_ptr())) {
            bind(obj, expr_op);
            return;
        }

        // Handle integer down-conversion preserving signedness.
        if (from_type.is_signed(1) && to_type.is_signed(1)
                && (from_type.size() > to_type.size())) {
            bind(obj, expr_op.reinterpret(to_type));
            return;
        }

        auto dst_op = alloc_op(obj);
        auto mod = dst_op.mod();
        if (obj->saturate) mod |= host_->sat;
        host_->emov(mod, dst_op, expr_op);
    }

    void _visit(const float_imm_t *obj) override { bind(obj, to_ngen(obj)); }

    void _visit(const int_imm_t *obj) override { bind(obj, to_ngen(obj)); }

    void _visit(const load_t *obj) override {
        auto &type = obj->type;
        auto buf_op = eval(obj->buf);
        auto off_op = eval(obj->off);
        int stride;
        if (obj->has_default_stride()) {
            stride = 1;
        } else {
            ir_assert(obj->stride % type.scalar().size() == 0);
            stride = obj->stride / type.scalar().size();
        }
        auto load_reg_data = ngen_reg_data(buf_op.reg_data(),
                to_cpp<int>(off_op.immediate()), to_ngen(type.scalar()),
                type.elems(), stride);
        bind(obj, load_reg_data);
    }

    void _visit(const ptr_t *obj) override {
        auto base_op = eval(obj->base);

        if (is_zero(obj->off)) {
            bind(obj, base_op);
            return;
        }

        ir_assert(base_op.is_reg_data());

        int off = to_cpp<int>(obj->off);
        int base = base_op.reg_data().getBase();
        auto grf = ngen::GRF(base + off / reg_bytes).retype(ngen::DataType::ub);
        if (off % reg_bytes == 0)
            bind(obj, grf);
        else
            bind(obj, grf[off % reg_bytes]);
    }

    void _visit(const shuffle_t *obj) override {
        if (obj->is_broadcast() && !obj->type.is_bool()) {
            auto scalar_op = eval(obj->vec[0]);
            bind(obj, scalar_op);
            return;
        }

        int elems = obj->elems();
        auto dst_op = alloc_op(obj);

        if (obj->type.is_bool()) {
            auto e_shuffle = expr_t(obj);
            ir_assert(dst_op.is_flag_register()) << e_shuffle;
            ir_assert(!dst_op.is_negated()) << e_shuffle;
            ir_assert(is_shuffle_const(obj)) << e_shuffle;
            uint16_t flag_mask = 0;
            for (int i = obj->elems() - 1; i >= 0; i--) {
                flag_mask <<= 1;
                flag_mask |= (to_cpp<bool>(e_shuffle[i]) ? 1 : 0);
            }
            if (dst_op.mod().getPredCtrl() == ngen::PredCtrl::None) {
                host_->emov(1, dst_op, ngen::Immediate(flag_mask));
            } else {
                ir_assert(dst_op.mod().getFlagReg() == dst_op.flag_register());
                host_->and_(1, dst_op.flag_register(), dst_op.flag_register(),
                        ngen::Immediate(flag_mask));
            }
            expr_binding_.mark_as_evaluated(obj);
            return;
        }

        // tuples: <offset, length, idx>
        std::vector<std::tuple<int, int, int>> chunks;
        for (int i = 0; i < elems; i++) {
            int idx = obj->idx[i];
            if (chunks.empty() || std::get<2>(chunks.back()) != idx) {
                chunks.emplace_back(i, 1, idx);
            } else {
                std::get<1>(chunks.back())++;
            }
        }

        for (auto &chunk : chunks) {
            int off = std::get<0>(chunk);
            int exec_size = std::get<1>(chunk);
            int idx = std::get<2>(chunk);
            auto chunk_op = dst_op.sub_reg_data(off, exec_size);
            eval(obj->vec[idx], chunk_op);
        }
        expr_binding_.mark_as_evaluated(obj);
    }

    void _visit(const unary_op_t *obj) override {
        ir_assert(obj->op_kind == op_kind_t::_minus);
        auto a_op = eval(obj->a);
        bind(obj, -a_op);
    }

    void _visit(const var_t *obj) override {
        ir_assert(expr_binding_.is_evaluated(obj)) << expr_t(obj);
    }

private:
    ngen_operand_t alloc_op(const expr_t &e) {
        if (expr_binding_.is_bound(e)) return expr_binding_.get(e);

        // Expression is not bound yet, allocate new storage and bind.
        ngen_operand_t op;
        if (e.type().is_bool()) {
            op = ngen_operand_t(scope_.alloc_flag(), e.type().elems());
        } else {
            op = ngen_operand_t(
                    scope_.alloc_reg_data(e.type()), e.type().elems());
        }
        expr_binding_.bind_unevaluated(e, op);
        return op;
    }

    // Pre-allocates a strided register region for expression `e` if needed.
    ngen_operand_t maybe_alloc_strided_op(
            const type_t &res_type, const expr_t &e) {
        // Need q-strided region for `e` if res_type is q/uq and `e` is of a
        // sub-q data type and not a scalar.
        if (e.type().is_scalar()) return ngen_operand_t();
        if (!utils::one_of(res_type.scalar(), type_t::s64(), type_t::u64()))
            return ngen_operand_t();
        if (utils::one_of(e.type().scalar(), type_t::s64(), type_t::u64()))
            return ngen_operand_t();

        auto *shuffle = e.as_ptr<shuffle_t>();
        if (shuffle && shuffle->is_broadcast()) return ngen_operand_t();

        return ngen_operand_t(
                scope_.alloc_reg_data(e.type(), res_type.scalar().size()),
                e.type().elems());
    }

    void bind(const expr_t &e, const ngen_operand_t &op) {
        if (!expr_binding_.is_bound(e)) {
            expr_binding_.bind(e, op);
            return;
        }

        // Expression is already bound, move to the location it was bound to.
        auto bound_op = expr_binding_.get(e);
        host_->emov(bound_op.mod(), bound_op, op);
        expr_binding_.mark_as_evaluated(e);
    }

    void ebinary(const binary_op_t *obj, const ngen::InstructionModifier &mod,
            const ngen_operand_t &dst, const ngen_operand_t &src0,
            const ngen_operand_t &src1) {
        switch (obj->op_kind) {
            case op_kind_t::_add: host_->eadd(mod, dst, src0, src1); break;
            case op_kind_t::_sub: host_->eadd(mod, dst, src0, -src1); break;
            case op_kind_t::_mul: host_->emul(mod, dst, src0, src1); break;
            case op_kind_t::_div: host_->ediv(mod, dst, src0, src1); break;
            case op_kind_t::_mod: host_->emod(mod, dst, src0, src1); break;
            case op_kind_t::_shl: host_->eshl(mod, dst, src0, src1); break;
            case op_kind_t::_shr: host_->eshr(mod, dst, src0, src1); break;
            case op_kind_t::_min: host_->emin(mod, dst, src0, src1); break;
            case op_kind_t::_max: host_->emax(mod, dst, src0, src1); break;
            case op_kind_t::_ge:
            case op_kind_t::_gt:
            case op_kind_t::_le:
            case op_kind_t::_lt:
            case op_kind_t::_eq:
            case op_kind_t::_ne: {
                ir_assert(!dst.is_negated()) << "Destination can't be negated.";
                ngen::InstructionModifier cmp_mod = mod;
                cmp_mod |= cmp_op_to_ngen(obj->op_kind);
                cmp_mod |= dst.flag_register();
                host_->ecmp(cmp_mod, src0, src1);
                break;
            }
            default:
                ir_error_not_expected()
                        << "Unknown kind: " << to_string(obj->op_kind);
        }
    }

    conv_kernel_t<hw> *host_;
    expr_binding_t expr_binding_;
    ngen_register_scope_t &scope_;
};

// Helper to emit send instructions.
class send_impl_t {
public:
    send_impl_t(const send_t &send) : send_(send) {}

    template <typename GeneratorT, typename T>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const ngen::InstructionModifier &mod,
            const ngen::RegData &surf_base_addr, int surf_bti,
            const ngen::RegData &header, const T &data) {

        auto access_type = send_.access_type;
        auto data_type = send_.data_type;
        auto data_elems = send_.data_elems;
        auto address_model = send_.address_model;

        bool is_read = (access_type == ngen_proxy::Access::Read);
        ngen::AddressBase address_base;
        if (address_model == ngen_proxy::AddressModel::ModelBTS) {
            address_base = ngen::AddressBase::createBTS(surf_bti);
        } else if (address_model == ngen_proxy::AddressModel::ModelA64) {
            address_base = ngen::AddressBase::createA64(true);
        } else if (address_model == ngen_proxy::AddressModel::ModelSLM) {
            address_base = ngen::AddressBase::createSLM();
        } else {
            ir_error_not_expected();
        }

        if (data_type == type_t::byte()) {
            emit_load_or_store(is_read, host, mod,
                    ngen::scattered_byte(data_elems), address_base, header,
                    data);
        } else if (data_type == type_t::dword()) {
            emit_load_or_store(is_read, host, mod,
                    ngen::scattered_dword(data_elems), address_base, header,
                    data);
        } else if (data_type == type_t::qword()) {
            emit_load_or_store(is_read, host, mod,
                    ngen::scattered_qword(data_elems), address_base, header,
                    data);
        } else if (data_type == type_t::oword()) {
            emit_load_or_store(is_read, host, mod,
                    ngen::block_oword(data_elems), address_base, header, data);
        } else if (data_type == type_t::hword()) {
            emit_load_or_store(is_read, host, mod,
                    ngen::block_hword(data_elems), address_base, header, data);
        } else {
            ir_error_not_expected();
        }
    }

private:
    template <typename GeneratorT, typename DataSpecT>
    void emit_load_or_store(bool is_read, GeneratorT *host,
            const ngen::InstructionModifier &mod, const DataSpecT &spec,
            ngen::AddressBase base, const ngen::RegData &addr,
            const ngen::RegData &data) {
        if (is_read) {
            host->load(mod, data, spec, base, addr);
        } else {
            host->store(mod, spec, base, addr, data);
        }
    }

    const send_t &send_;
};

// Lowers IR to nGEN.
template <ngen::HW hw>
class ir_to_ngen_t : public ir_visitor_t {
public:
    ir_to_ngen_t(conv_kernel_t<hw> *host, const expr_binding_t &expr_binding)
        : host_(host)
        , expr_binding_(expr_binding)
        , simd_size_(host->cfg_.simd_size) {}

    void _visit(const alloc_t *obj) override {
        auto scope = register_scope();
        bool do_alloc = (obj->kind == alloc_kind_t::grf);
        if (do_alloc) {
            int regs = utils::div_up(obj->size, reg_bytes);
            ngen::Bundle bundle;
            auto *grf_attr = obj->attr.as_ptr<grf_alloc_attr_t>();
            if (grf_attr) bundle = to_ngen(grf_attr->bundle);
            auto reg_range = scope.alloc_range(regs, bundle);
            expr_binding_.bind(obj->buf, reg_range[0]);
        }
        visit(obj->body);
        if (do_alloc) expr_binding_.unbind(obj->buf);
    }

    void _visit(const for_t *obj) override {
        auto scope = register_scope();
        auto var_op = scope.alloc_sub(to_ngen(obj->var.type()));
        auto init_op = eval(obj->init, scope);
        auto bound_op = eval(obj->bound, scope);
        ngen::Label loop_label;
        host_->emov(1, var_op, init_op);
        expr_binding_.bind(obj->var, var_op);
        host_->mark(loop_label);
        visit(obj->body);
        host_->eadd(1, var_op, var_op, ngen::Immediate(1));
        host_->ecmp(1 | host_->lt | host_->f0[0], var_op, bound_op);
        host_->jmpi(1 | host_->f0[0], loop_label);
        expr_binding_.unbind(obj->var);
    }

    void _visit(const func_call_t *obj) override {
        auto scope = register_scope();
        auto &func = obj->func;
        if (func.is<dpas_t>()) {
            auto arg_ops = eval(obj->args, scope);
            dpas(func.as<dpas_t>(), arg_ops, obj->attr);
        } else if (func.is<mad_t>()) {
            auto arg_ops = eval(obj->args, scope);
            mad(func.as<mad_t>(), arg_ops, obj->attr);
        } else if (func.is<send_t>()) {
            auto &send_func = func.as<send_t>();
            auto args = obj->args;
            auto &mask = send_t::arg_mask(args);
            // If all channels are disabled for writing, quick return.
            if (is_const_broadcast(mask, expr_t(false)) && send_func.is_write())
                return;
            // If all channels are enabled, do not use mask.
            if (is_const_broadcast(mask, expr_t(true))) { mask = expr_t(); }
            auto arg_ops = eval(args, scope);
            send(scope, func.as<send_t>(), arg_ops, obj->attr);
        } else if (func.is<eltwise_t>()) {
            auto &eltwise_func = func.as<eltwise_t>();
            auto arg_ops = eval(obj->args, scope);
            eltwise(scope, eltwise_func, arg_ops);
        } else if (func.is_equal(funcs::barrier_func())) {
            barrier(obj->attr);
        } else if (func.is_equal(funcs::barrier_wait_func())) {
            barrier_wait();
        } else if (func.is_equal(funcs::signal_func())) {
            signal(obj->attr);
        } else if (func.is_equal(funcs::slm_fence_func())) {
            slm_fence(obj->attr);
        } else {
            ir_error_not_expected() << object_t(obj);
        }
    }

    void _visit(const let_t *obj) override {
        // External variable.
        if (obj->value.is_empty()) {
            ir_assert(expr_binding_.is_bound(obj->var))
                    << "Unknown external variable: " << obj->var;
            visit(obj->body);
            return;
        }

        auto scope = register_scope();
        if (is_const(obj->value) || obj->var.type() != obj->value.type()) {
            auto var_op = scope.alloc_reg_data(obj->var.type());
            eval(obj->value, scope, var_op);
            expr_binding_.bind(obj->var, var_op);
        } else {
            auto value_op = eval(obj->value, scope);
            expr_binding_.bind(obj->var, value_op);
        }
        visit(obj->body);
        expr_binding_.unbind(obj->var);
    }

    void _visit(const store_t *obj) override {
        auto scope = register_scope();
        auto buf_op = eval(obj->buf, scope);
        auto off = to_cpp<int>(obj->off);
        auto mask_op = eval(obj->mask, scope);

        auto &type = obj->value.type();

        int stride;
        if (obj->has_default_stride()) {
            stride = 1;
        } else {
            ir_assert(obj->stride % type.scalar().size() == 0);
            stride = obj->stride / type.scalar().size();
        }

        ngen::InstructionModifier mod = type.elems();
        if (!mask_op.is_invalid()) mod |= mask_op.flag_register_mod();
        auto dst_rd = ngen_reg_data(buf_op.reg_data(), off,
                to_ngen(type.scalar()), type.elems(), stride);
        ngen_operand_t dst(dst_rd, mod);
        eval(obj->value, scope, dst);
    }

private:
    ngen_register_scope_t register_scope() {
        return ngen_register_scope_t(host_->ra_);
    }

    void signal(const func_call_attr_t &attr) {
        ngen::InstructionModifier mod;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);
        host_->barriermsg(mod, host_->signal_header_);
    }

    void barrier_wait() { host_->barrierwait(); }

    void slm_fence(const func_call_attr_t &attr) {
        auto scope = register_scope();
        auto tmp = scope.alloc();
        ngen::InstructionModifier mod;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);
        host_->slmfence(mod, tmp, host_->r0);
        host_->template mov<int32_t>(mod | 8, host_->null, tmp);
    }

    void barrier(const func_call_attr_t &attr) {
        auto scope = register_scope();
        auto tmp = scope.alloc();
        ngen::InstructionModifier mod;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);
        host_->slmfence(mod, tmp, host_->r0);
        host_->template mov<int32_t>(mod | 8, host_->null, tmp);
        host_->barriermsg(mod, host_->signal_header_);
        host_->barrierwait();
    }

    void dpas(const dpas_t &dpas_func, const std::vector<ngen_operand_t> &args,
            const func_call_attr_t &attr) {
        auto dst = dpas_t::arg_dst(args).reg_data();
        auto src1 = dpas_t::arg_src1(args).reg_data();
        auto src2 = dpas_t::arg_src2(args).reg_data();

        ngen::RegData src0;
        auto &src0_op = dpas_t::arg_src0(args);
        if (src0_op.is_reg_data()) {
            src0 = src0_op.reg_data();
        } else {
            ir_assert(src0_op.is_immediate());
            ir_assert(to_cpp<int32_t>(src0_op.immediate()) == 0);
            src0 = host_->null;
        }

        dst.setType(to_ngen(dpas_func.dst_type));
        src0.setType(to_ngen(dpas_func.dst_type));
        src1.setType(to_ngen(dpas_func.src1_type));
        src2.setType(to_ngen(dpas_func.src2_type));
        ngen::InstructionModifier mod = simd_size_;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);
        if (dpas_func.is_dpasw) {
            host_->dpasw(mod, dpas_func.sdepth, dpas_func.rcount, dst, src0,
                    src1, src2);
        } else {
            host_->dpas(mod, dpas_func.sdepth, dpas_func.rcount, dst, src0,
                    src1, src2);
        }
    }

    void mad(const mad_t &mad_func, const std::vector<ngen_operand_t> &args,
            const func_call_attr_t &attr) {
        auto dst = mad_t::arg_dst(args).reg_data();
        auto src1 = mad_t::arg_src1(args).reg_data();
        auto src2 = mad_t::arg_src2(args).reg_data();

        ngen::RegData src0;
        auto &src0_op = mad_t::arg_src0(args);
        if (src0_op.is_reg_data()) {
            src0 = ngen_reg_data(src0_op.reg_data(), 0,
                    to_ngen(mad_func.dst_type), mad_func.dst_simd_size);
        } else {
            ir_assert(src0_op.is_immediate());
            ir_assert(to_cpp<int32_t>(src0_op.immediate()) == 0);
            src0 = host_->null;
            src0.setType(to_ngen(mad_func.dst_type));
        }

        dst = ngen_reg_data(
                dst, 0, to_ngen(mad_func.dst_type), mad_func.dst_simd_size);
        src1 = ngen_reg_data(
                src1, 0, to_ngen(mad_func.src1_type), mad_func.src1_simd_size);
        src2 = ngen_reg_data(
                src2, 0, to_ngen(mad_func.src2_type), mad_func.src2_simd_size);

        ngen::InstructionModifier mod = simd_size_;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);

        // Force scalar register parameter to be scalar since when the registers
        // offset = 0 the register is interpreted as a vector by default
        if (mad_func.src1_simd_size == 1)
            src1.setRegion(0, mad_func.src1_simd_size, 0);
        if (mad_func.src2_simd_size == 1)
            src2.setRegion(0, mad_func.src2_simd_size, 0);

        host_->mad(mod, dst, src0, src1, src2);
    }

    void send(ngen_register_scope_t &scope, const send_t &send_func,
            const std::vector<ngen_operand_t> &args,
            const func_call_attr_t &attr) {
        send_impl_t spec_impl(send_func);
        auto &mem_buf_op = send_t::arg_mem_buf(args);
        auto &mem_off_op = send_t::arg_mem_off(args);
        auto &reg_buf_op = send_t::arg_reg_buf(args);
        auto &mask_op = send_t::arg_mask(args);

        ngen::RegData mem_buf;
        if (send_func.address_model != ngen_proxy::AddressModel::ModelSLM) {
            mem_buf = mem_buf_op.reg_data();
        }
        ngen::InstructionModifier mod = send_func.eff_mask_count;
        ir_assert(math::is_pow2(mod.getExecSize()));
        if (!attr.is_empty())
            mod |= to_ngen(attr.as<instruction_modifier_attr_t>().mod);
        if (!mask_op.is_invalid()) mod |= mask_op.flag_register_mod();
        auto rd = reg_buf_op.reg_data();

        // Zero-out inactive channels.
        if (send_func.is_read() && mod.getPredCtrl() != ngen::PredCtrl::None) {
            auto rd_mov = rd;
            rd_mov.setType(ngen::DataType::f);
            auto mod_mov = ~mod;
            mod_mov.setSWSB({});
            int step = send_func.mask_count() * sizeof(uint32_t);
            for (int i = 0; i < send_func.register_size(); i += step) {
                auto sub_rd_mov = ngen_reg_data(
                        rd_mov, i, ngen::DataType::f, send_func.eff_mask_count);
                host_->emov(mod_mov, sub_rd_mov, ngen::Immediate(0.0f));
            }
        }

        // Emit send instruction.
        spec_impl.emit(
                host_, scope, mod, mem_buf, 0, mem_off_op.reg_data(), rd);
    }

    void eltwise(ngen_register_scope_t &scope, const eltwise_t &func,
            const std::vector<ngen_operand_t> &args) {
        int elems = to_cpp<int>(eltwise_t::arg_elems(args));
        auto &data_op = eltwise_t::arg_data(args);
        auto &data_rd = data_op.reg_data();

        ir_assert(elems * sizeof(float) % reg_bytes == 0)
                << "Partial GRF updates are not supported.";
        ir_assert(data_rd.getOffset() == 0)
                << "Data must be aligned to GRF boundary.";

        jit_eltwise_injector_f32<hw> inj(
                host_, func.alg_kind, func.alpha, func.beta, func.scale);
        auto scratch = scope.alloc_range(inj.preferred_scratch_regs());
        inj.set_scratch(scratch);
        inj.prepare();
        inj.compute(ngen::GRFRange(
                data_rd.getBase(), elems * sizeof(float) / reg_bytes));
    }

    ngen_operand_t eval(const expr_t &e, ngen_register_scope_t &scope,
            const ngen_operand_t &out_operand = ngen_operand_t()) {
        expr_evaluator_t<hw> expr_evaluator(host_, expr_binding_, scope);
        return expr_evaluator.eval(e, out_operand);
    }

    std::vector<ngen_operand_t> eval(
            const std::vector<expr_t> &exprs, ngen_register_scope_t &scope) {
        expr_evaluator_t<hw> expr_evaluator(host_, expr_binding_, scope);
        return expr_evaluator.eval(exprs);
    }

    conv_kernel_t<hw> *host_;
    expr_binding_t expr_binding_;
    int simd_size_;
};

template <ngen::HW hw>
conv_kernel_t<hw>::conv_kernel_t(const conv_config_t &cfg,
        const convolution_pd_t *pd, kernel_arg_info_t &kernel_arg_info)
    : cfg_(cfg), ra_(hw) {

    init_kernel_arg_info(kernel_arg_info);

    // Build IR for the kernel.
    kernel_builder_t builder(cfg, pd, kernel_arg_info);
    stmt_t body = builder.stmt();

    alloc_manager_t alloc_mgr(body);

    setup_interface(body, kernel_arg_info);

    setDefaultNoMask();
    setDefaultAutoSWSB(true);

    prologue();

    // Claim registers.
    ra_.claim(r0);
    for (int i = 0; i < 3; i++)
        ra_.claim(getLocalID(i));

    for (int i = 0; i < kernel_arg_info.nargs(); i++) {
        ra_.claim(getArgument(kernel_arg_info.arg_name(i)));
    }

    if (emu_strategy.emulate64) {
        emu_state.temp[0] = ra_.alloc();
        emu_state.temp[1] = ra_.alloc();
    }
    // Enable IEEE f32 -> s32 rounding and f32/f16 denormals.
    or_(1, cr0, cr0, uint16_t(0x1480));

    // Allocate and initialize signal header for future use.
    signal_header_ = ra_.alloc();
    barrierheader(signal_header_);

    // Bind "external" variables.
    expr_binding_t expr_binding;

    // Bind grid indices.
    expr_binding.bind(builder.kernel_grid_idx(0), r0.ud(1));
    expr_binding.bind(builder.kernel_grid_idx(1), r0.ud(6));
    expr_binding.bind(builder.kernel_grid_idx(2), r0.ud(7));

    // Bind local IDs.
    for (int i = 0; i < 3; i++) {
        expr_binding.bind(builder.local_id(i), getLocalID(i).uw(0));
    }

    // Bind arguments.
    for (int i = 0; i < kernel_arg_info.nargs(); i++) {
        auto &arg_var = kernel_arg_info.arg_var(i);
        auto &name = kernel_arg_info.arg_name(i);
        if (arg_var.type().is_ptr()) {
            auto alloc_buf = alloc_mgr.find_buffer(name);
            ir_assert(alloc_buf.is_same(arg_var));
        }
        expr_binding.bind(arg_var, getArgument(name));
    }

    // Bind SLM buffer (SLM loads/stores use 0-based offsets).
    auto slm_buf = alloc_mgr.find_buffer("slm", /*allow_empty=*/true);
    if (!slm_buf.is_empty()) { expr_binding.bind(slm_buf, to_ngen(expr_t(0))); }

    // Generate assembly from IR.
    ir_to_ngen_t<hw> visitor(this, expr_binding);
    visitor.visit(body);

    epilogue();
    pad_kernel();
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

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

#ifndef NGEN_ASM_HPP
#define NGEN_ASM_HPP

#include <array>
#include <cstdint>
#include <sstream>

#define NGEN_ASM
#include "ngen.hpp"


namespace ngen {


inline void RegData::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
#ifdef NGEN_SAFE
    if (isInvalid()) throw invalid_object_exception();
#endif

    if (getNeg()) str << '-';
    if (getAbs()) str << "(abs)";

    if (isARF()) {
        str << getARFType();
        switch (getARFType()) {
            case ARFType::null:
            case ARFType::sp:
            case ARFType::ip:
                break;
            default:
                str << getARFBase();
        }
    } else if (isIndirect()) {
        str << "r[a" << getIndirectBase() << '.' << getIndirectOff();
        if (getOffset())
            str << ',' << getOffset();
        str << ']';
    } else
        str << 'r' << base;

    if (detail <= PrintDetail::base) return;

    if (!isIndirect() && !isNull())
        str << '.' << getOffset();

    if (detail <= PrintDetail::sub_no_type) return;

    if (detail >= PrintDetail::hs && !isNull()) {
        str << '<';
        if (detail >= PrintDetail::vs_hs && !isVxIndirect())
            str << getVS() << ';';
        if (detail == PrintDetail::full)
            str << getWidth() << ',';
        str << getHS();
        str << '>';
    }

    str << ':' << getType();
}

static inline std::ostream& operator<<(std::ostream &str, const RegData &r)
{
    LabelManager man;
    r.outputText(str, PrintDetail::full, man);
    return str;
}

inline void Immediate::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
    uint64_t nbytes = getBytes(getType());
    uint64_t val;

    if (nbytes == 8)
        val = payload;
    else
        val = payload & ((uint64_t(1) << (nbytes * 8)) - 1);

    str << "0x" << std::hex << val << std::dec;
    if (!hiddenType && detail >= PrintDetail::sub)
        str << ':' << type;
}

inline void ExtendedReg::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
#ifdef NGEN_SAFE
    if (isInvalid()) throw invalid_object_exception();
#endif

    if (base.getNeg()) str << '-';
    if (base.getAbs()) str << "(abs)";

    str << 'r' << base.getBase() << '.';
    if (mmeNum == 8)
        str << "nomme";
    else
        str << "mme" << int(mmeNum);

    if (detail >= PrintDetail::sub)
        str << ':' << base.getType();
}

inline void Align16Operand::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
#ifdef NGEN_SAFE
    if (isInvalid()) throw invalid_object_exception();
    throw iga_align16_exception();
#else
    str << "<unsupported Align16 operand>";
#endif
}

inline void Label::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) {
    str << 'L' << getID(man);
}




struct NoOperand {
    static const bool emptyOp = true;
    void fixup(int esize, DataType defaultType, bool isDest) const {}
    constexpr bool isScalar() const { return false; }

    void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const {}
};

#if defined(NGEN_GLOBAL_REGS) && !defined(NGEN_GLOBAL_REGS_DEFINED)
#include "ngen_registers.hpp"
#endif

class AsmCodeGenerator {
protected:
    class InstructionStream {
        friend class AsmCodeGenerator;

        std::ostream *buffer;
        bool root;

        InstructionStream() : buffer(new std::ostringstream), root(false) {}
        InstructionStream(std::ostream &buffer_) : buffer(&buffer_), root(true) {}

        void append(InstructionStream &other) {
            if (!other.root) {
                auto &sbuffer = *reinterpret_cast<std::ostringstream*>(other.buffer);
                *buffer << sbuffer.str();
            }
        }

    public:
        ~InstructionStream() { if (buffer && !root) delete buffer; }

        std::ostream *getBuffer() const { return buffer; }
    };

    HW hardware;
    bool isGen12;

    LabelManager labelManager;
    std::ostream *outStream;
    std::vector<InstructionStream*> streamStack;

public:
    AsmCodeGenerator(HW hardware_, std::ostream &outStream_) : hardware(hardware_), isGen12(hardware_ >= HW::Gen12LP), labelManager{}, outStream(&outStream_), defaultModifier{} {
        streamStack.push_back(new InstructionStream(outStream_));
    }

    ~AsmCodeGenerator() {
        for (auto &s : streamStack)
            delete s;
    }

private:
    InstructionModifier defaultModifier;

    enum class ModPlacementType {Pre, Mid, Post};
    inline void outputMods(const InstructionModifier &mod, Opcode op, ModPlacementType location);

    template <typename D, typename S0, typename S1, typename S2, typename Ext>
    inline void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, Ext ext);

protected:
    // Common output functions.
    template <typename D, typename S0, typename S1, typename S2> void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2) {
        opX(op, defaultType, mod, dst, src0, src1, src2, [](std::ostream &str) {});
    }
    template <typename D, typename S0, typename S1> void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1) {
        opX(op, defaultType, mod, dst, src0, src1, NoOperand());
    }
    template <typename D, typename S0, typename S1> void opX(Opcode op, const InstructionModifier &mod, D dst, S0 src0, S1 src1) {
        opX(op, DataType::invalid, mod, dst, src0, src1);
    }
    template <typename D, typename S0> void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0) {
        opX(op, defaultType, mod, dst, src0, NoOperand());
    }
    template <typename D, typename S0> void opX(Opcode op, const InstructionModifier &mod, D dst, S0 src0) {
        opX(op, DataType::invalid, mod, dst, src0);
    }
    template <typename D> void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst) {
        opX(op, defaultType, mod, dst, NoOperand());
    }
    template <typename D> void opX(Opcode op, const InstructionModifier &mod, D dst) {
        opX(op, DataType::invalid, mod, dst);
    }
    void opX(Opcode op) {
        opX(op, InstructionModifier(), NoOperand());
    }
    void opX(Opcode op, const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opX(op, DataType::invalid, mod, jip, NoOperand());
    }
    void opX(Opcode op, const InstructionModifier &mod, Label &jip, Label &uip) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(op, DataType::invalid, mod, NoOperand(), jip, uip, NoOperand());
    }

    template <typename S1, typename ED, typename D>
    inline void opSend(Opcode op, const InstructionModifier &mod, SharedFunction sf, RegData dst, RegData src0, S1 src1, ED exdesc, D desc);

    inline void opDpas(Opcode op, const InstructionModifier &mod, int sdepth, int rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2);

    template <typename D, typename S0>
    inline void opCall(Opcode op, const InstructionModifier &mod, D dst, S0 src0);

    template <typename S1>
    inline void opJmpi(Opcode op, const InstructionModifier &mod, S1 src1);

    template <typename S0>
    inline void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, S0 src0);

    void unsupported();

    // Configuration
    void setDefaultNoMask(bool def = true) { defaultModifier = def ? NoMask : InstructionModifier(); }

    // Stream handling.
    void pushStream()                               { pushStream(new InstructionStream()); }
    void pushStream(InstructionStream &s)           { pushStream(&s); }
    void pushStream(InstructionStream *s)           { streamStack.push_back(s); outStream = s->buffer; }

    InstructionStream *popStream();

    void appendStream(InstructionStream *s)         { appendStream(*s); }
    void appendStream(InstructionStream &s)         { streamStack.back()->append(s); }
    void appendCurrentStream()                      { InstructionStream *s = popStream(); appendStream(s); delete s; }

    void discardStream()                            { delete popStream(); }

    // Instructions
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::addc, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::addc, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void and(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        and_<DT>(mod, dst, src0, src1);
    }
    template <typename DT = void>
    void and(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        and_<DT>(mod, dst, src0, src1);
    }
#endif
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfi1(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::bfi1_gen12 : Opcode::bfi1, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void bfi1(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::bfi1_gen12 : Opcode::bfi1, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfrev(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(isGen12 ? Opcode::bfrev_gen12 : Opcode::bfrev, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void bfrev(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(isGen12 ? Opcode::bfrev_gen12 : Opcode::bfrev, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::bfn, getDataType<DT>(), mod, dst, src0, src1, src2, [=](std::ostream &str) { str << ".0x" << std::hex << uint32_t(ctrl) << std::dec; });
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::bfn, getDataType<DT>(), mod, dst, src0, src1, src2, [=](std::ostream &str) { str << ".0x" << std::hex << uint32_t(ctrl) << std::dec; });
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        opX(Opcode::bfn, getDataType<DT>(), mod, dst, src0, src1, src2, [=](std::ostream &str) { str << ".0x" << std::hex << uint32_t(ctrl) << std::dec; });
    }
    void brc(const InstructionModifier &mod, Label &jip, Label &uip) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::brc, mod, jip, uip);
    }
    void brc(const InstructionModifier &mod, const RegData &src0) {
        opCall(Opcode::brc, mod, NoOperand(), src0);
    }
    void brd(const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opX(Opcode::brd, mod, jip);
    }
    void brd(const InstructionModifier &mod, const RegData &src0) {
        opCall(Opcode::brd, mod, NoOperand(), src0);
    }
    void break_(const InstructionModifier &mod, Label &jip, Label &uip) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::break_, mod, jip, uip);
    }
    void call(const InstructionModifier &mod, const RegData &dst, Label &jip) {
        (void) jip.getID(labelManager);
        opCall(Opcode::call, mod, dst, jip);
    }
    void call(const InstructionModifier &mod, const RegData &dst, const RegData &jip) {
        opCall(Opcode::call, mod, dst, jip);
    }
    void calla(const InstructionModifier &mod, const RegData &dst, int32_t jip) {
        opCall(Opcode::calla, mod, dst, Immediate(jip));
    }
    void calla(const InstructionModifier &mod, const RegData &dst, const RegData &jip) {
        opCall(Opcode::calla, mod, dst, jip);
    }
    template <typename DT = void>
    void cbit(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::cbit, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void cbit(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::cbit, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void cmpn(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::cmpn, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    void cont(const InstructionModifier &mod, Label &jip, Label &uip) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::cont, mod, jip, uip);
    }
    template <typename DT = void>
    void dp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::dp2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::dp2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::dp3, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::dp3, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::dp4, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::dp4, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    void dpas(const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opDpas(Opcode::dpas, mod, sdepth, rcount, dst, src0, src1, src2);
    }
    void dpasw(const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opDpas(Opcode::dpasw, mod, sdepth, rcount, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dph(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::dph, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dph(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::dph, getDataType<DT>(), mod, dst, src0, src1);
    }
    void else_(const InstructionModifier &mod, Label &jip, Label &uip, bool branchCtrl = false) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::else_, DataType::invalid, mod, NoOperand(), jip, uip, NoOperand(), [=](std::ostream &str) {
            if (branchCtrl) str << ".b";
        });
    }
    void else_(InstructionModifier mod, Label &jip) {
        else_(mod, jip, jip);
    }
    void endif(const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opX(Opcode::endif, mod, NoOperand(), jip);
    }
    template <typename DT = void>
    void fbh(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::fbh, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void fbh(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::fbh, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void fbl(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::fbl, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void fbl(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::fbl, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void frc(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::frc, getDataType<DT>(), mod, dst, src0);
    }
    void goto_(const InstructionModifier &mod, Label &jip, Label &uip, bool branchCtrl = false) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::goto_, DataType::invalid, mod, NoOperand(), jip, uip, NoOperand(), [=](std::ostream &str) {
            if (branchCtrl) str << ".b";
        });
    }
    void goto_(const InstructionModifier &mod, Label &jip) {
        goto_(mod, jip, jip);
    }
    void halt(const InstructionModifier &mod, Label &jip, Label &uip) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::halt, mod, jip, uip);
    }
    void halt(const InstructionModifier &mod, Label &jip) {
        halt(mod, jip, jip);
    }
    void if_(const InstructionModifier &mod, Label &jip, Label &uip, bool branchCtrl = false) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::if_, DataType::invalid, mod, NoOperand(), jip, uip, NoOperand(), [=](std::ostream &str) {
            if (branchCtrl) str << ".b";
        });
    }
    void if_(const InstructionModifier &mod, Label &jip) {
        if_(mod, jip, jip);
    }
    void illegal() {
        opX(Opcode::illegal);
    }
    void join(const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opX(Opcode::join, mod, jip);
    }
    void jmpi(const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opJmpi(Opcode::jmpi, mod, jip);
    }
    void jmpi(const InstructionModifier &mod, const RegData &jip) {
        opJmpi(Opcode::jmpi, mod, jip);
    }
    template <typename DT = void>
    void lzd(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::lzd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void lzd(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::lzd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void lrp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::lrp, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::mach, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::mach, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
#ifdef NGEN_SAFE
        if (hardware < HW::Gen10) unsupported();
#endif
        opX(Opcode::mach, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
#ifdef NGEN_SAFE
        if (hardware < HW::Gen10) unsupported();
#endif
        opX(Opcode::mach, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const Align16Operand &dst, const Align16Operand &src0, const Align16Operand &src1, const Align16Operand &src2) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void madm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, const ExtendedReg &src2) {
        opX(Opcode::madm, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0) {
#ifdef NGEN_SAFE
        if (mathArgCount(fc) != 1) throw invalid_operand_count_exception();
#endif
        if (fc == MathFunction::rsqtm)
            math<DT>(mod, fc, dst | nomme, src0 | nomme);
        else
            opX(Opcode::math, getDataType<DT>(), mod, dst, src0, NoOperand(), NoOperand(), [=](std::ostream &str) { str << '.' << fc; });
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const RegData &src1) {
#ifdef NGEN_SAFE
        if (mathArgCount(fc) != 2) throw invalid_operand_count_exception();
#endif
        if (fc == MathFunction::invm)
            math<DT>(mod, fc, dst | nomme, src0 | nomme, src1 | nomme);
        else
            opX(Opcode::math, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), [=](std::ostream &str) { str << '.' << fc; });
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const Immediate &src1) {
#ifdef NGEN_SAFE
        if (fc == MathFunction::invm || fc == MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        opX(Opcode::math, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), [=](std::ostream &str) { str << '.' << fc; });
    }
    template <typename DT = void>
    void math(InstructionModifier mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        mod.setCMod(ConditionModifier::eo);
        opX(Opcode::math, getDataType<DT>(), mod, dst, src0, NoOperand(), NoOperand(), [=](std::ostream &str) { str << '.' << fc; });
    }
    template <typename DT = void>
    void math(InstructionModifier mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::invm) throw invalid_operand_exception();
#endif
        mod.setCMod(ConditionModifier::eo);
        opX(Opcode::math, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), [=](std::ostream &str) { str << '.' << fc; });
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
#ifdef NGEN_SAFE
        if (hardware < HW::Gen10) throw unsupported_instruction();
#endif
        opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1);
    }
    void nop() {
        opX(isGen12 ? Opcode::nop_gen12 : Opcode::nop);
    }
    template <typename DT = void>
    void not_(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(isGen12 ? Opcode::not_gen12 : Opcode::not_, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void not_(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(isGen12 ? Opcode::not_gen12 : Opcode::not_, getDataType<DT>(), mod, dst, src0);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void not(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        not_<DT>(mod, dst, src0);
    }
    template <typename DT = void>
    void not(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        not_<DT>(mod, dst, src0);
    }
#endif
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::or_gen12 : Opcode:: or_ , getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::or_gen12 : Opcode:: or_ , getDataType<DT>(), mod, dst, src0, src1);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void or(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        or_<DT>(mod, dst, src0, src1);
    }
    template <typename DT = void>
    void or(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        or_<DT>(mod, dst, src0, src1);
    }
#endif
    template <typename DT = void>
    void pln(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::pln, getDataType<DT>(), mod, dst, src0, src1);
    }
    void ret(const InstructionModifier &mod, const RegData &src0) {
        opJmpi(Opcode::ret, mod, src0);
    }
    template <typename DT = void>
    void rndd(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::rndd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndd(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::rndd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rnde(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::rnde, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rnde(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::rnde, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndu(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::rndu, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndu(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::rndu, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndz(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::rndz, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndz(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::rndz, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::sel_gen12 : Opcode::sel, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::sel_gen12 : Opcode::sel, getDataType<DT>(), mod, dst, src0, src1);
    }

    /* Gen12-style sends */
    template <typename S1, typename T1, typename T2> void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, S1 src1, T1 exdesc, T2 desc) {
        opSend((isGen12 || S1::emptyOp) ? Opcode::send : Opcode::sends, mod, sf, dst, src0, src1, exdesc, desc);
    }
    template <typename S1, typename T1, typename T2> void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, S1 src1, T1 exdesc, T2 desc) {
        opSend((isGen12 || S1::emptyOp) ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, src1, exdesc, desc);
    }
    /* Pre-Gen12 style sends */
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc) {
        if (isGen12)
            send(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, null, Immediate(exdesc & ~0x2F), Immediate(desc));
        else
            send(mod, SharedFunction::null, dst, src0, NoOperand(), Immediate(exdesc), Immediate(desc));
    }
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc) {
        if (isGen12)
            send(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, null, Immediate(exdesc & ~0x2F), desc);
        else
            send(mod, SharedFunction::null, dst, src0, NoOperand(), Immediate(exdesc), desc);
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc) {
        if (isGen12)
            sendc(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, null, Immediate(exdesc & ~0x2F), Immediate(desc));
        else
            sendc(mod, SharedFunction::null, dst, src0, NoOperand(), Immediate(exdesc), Immediate(desc));
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc) {
        if (isGen12)
            sendc(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, null, Immediate(exdesc & ~0x2F), desc);
        else
            sendc(mod, SharedFunction::null, dst, src0, NoOperand(), Immediate(exdesc), desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        uint32_t modExDesc = isGen12 ? (exdesc & ~0x2F) : exdesc;
        send(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, Immediate(modExDesc), Immediate(desc));
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        uint32_t modExDesc = isGen12 ? (exdesc & ~0x2F) : exdesc;
        send(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, Immediate(modExDesc), desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
#ifdef NGEN_SAFE
        if (isGen12) throw sfid_needed_exception();
#endif
        send(mod, static_cast<SharedFunction>(0), dst, src0, src1, exdesc, Immediate(desc));
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
#ifdef NGEN_SAFE
        if (isGen12) throw sfid_needed_exception();
#endif
        send(mod, static_cast<SharedFunction>(0), dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        uint32_t modExDesc = isGen12 ? (exdesc & ~0x2F) : exdesc;
        sendc(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, Immediate(modExDesc), Immediate(desc));
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        uint32_t modExDesc = isGen12 ? (exdesc & ~0x2F) : exdesc;
        sendc(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, Immediate(modExDesc), desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
#ifdef NGEN_SAFE
        if (isGen12) throw sfid_needed_exception();
#endif
        sendc(mod, static_cast<SharedFunction>(0), dst, src0, src1, exdesc, Immediate(desc));
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
#ifdef NGEN_SAFE
        if (isGen12) throw sfid_needed_exception();
#endif
        sendc(mod, static_cast<SharedFunction>(0), dst, src0, src1, exdesc, desc);
    }

    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void smov(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::smov_gen12 : Opcode::smov, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::subb, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::subb, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    void sync(SyncFunction fc, const InstructionModifier &mod = InstructionModifier()) {
        opSync(Opcode::sync, fc, mod, null);
    }
    void sync(SyncFunction fc, const RegData &src0) {
        opSync(Opcode::sync, fc, InstructionModifier(), src0);
    }
    void sync(SyncFunction fc, const InstructionModifier &mod, const RegData &src0) {
        opSync(Opcode::sync, fc, mod, src0);
    }
    void sync(SyncFunction fc, int src0) {
        opSync(Opcode::sync, fc, InstructionModifier(), Immediate(src0));
    }
    void sync(SyncFunction fc, const Immediate &src0) {
        opSync(Opcode::sync, fc, InstructionModifier(), src0);
    }
    void sync(SyncFunction fc, const InstructionModifier &mod, const Immediate &src0) {
        opSync(Opcode::sync, fc, mod, src0);
    }
    void wait(const InstructionModifier &mod, const RegData &nreg) {
        opX(Opcode::wait, mod, NoOperand(), nreg);
    }
    void while_(const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opX(Opcode::while_, mod, jip);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void xor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        xor_<DT>(mod, dst, src0, src1);
    }
    template <typename DT = void>
    void xor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        xor_<DT>(mod, dst, src0, src1);
    }
#endif
    inline void mark(Label &label);

#include "ngen_pseudo.hpp"
#ifndef NGEN_GLOBAL_REGS
#include "ngen_registers.hpp"
#endif
};


inline void AsmCodeGenerator::unsupported()
{
#ifdef NGEN_SAFE
    throw unsupported_instruction();
#endif
}

inline AsmCodeGenerator::InstructionStream *AsmCodeGenerator::popStream()
{
#ifdef NGEN_SAFE
    if (streamStack.size() <= 1) throw stream_stack_underflow();
#endif

    InstructionStream *result = streamStack.back();
    streamStack.pop_back();
    outStream = streamStack.back()->buffer;
    return result;
}

inline void AsmCodeGenerator::mark(Label &label)
{
    label.outputText(*outStream, PrintDetail::full, labelManager);
    *outStream << ':' << std::endl;
}

static inline bool isVariableLatency(Opcode op)
{
    switch (op) {
        case Opcode::send:
        case Opcode::sendc:
        case Opcode::dpas:
        case Opcode::dpasw:
        case Opcode::math:
            return true;
        default:
            return false;
    }
}

enum class ModPlacementType { Pre, Mid, Post };
inline void AsmCodeGenerator::outputMods(const InstructionModifier &mod, Opcode op, AsmCodeGenerator::ModPlacementType location)
{
    ConditionModifier cmod = mod.getCMod();
    PredCtrl ctrl = mod.getPredCtrl();
    bool wrEn = mod.isWrEn();
    bool havePred = (ctrl != PredCtrl::None) && (cmod != ConditionModifier::eo);

    switch (location) {
        case ModPlacementType::Pre:
            if (wrEn || havePred) {
                *outStream << '(';
                if (wrEn) {
                    *outStream << 'W';
                    if (havePred) *outStream << '&';
                }
                if (havePred) {
                    if (mod.isPredInv()) *outStream << '~';
                    mod.getFlagReg().outputText(*outStream, PrintDetail::sub_no_type, labelManager);
                    if (ctrl != PredCtrl::Normal)
                        *outStream << '.' << toText(ctrl, mod.isAlign16());
                }
                *outStream << ')';
            }
            *outStream << '\t';
            break;
        case ModPlacementType::Mid:
            if (mod.getExecSize() > 0)
                *outStream << '(' << mod.getExecSize() << "|M" << mod.getChannelOffset() << ')' << '\t';

            if (cmod != ConditionModifier::none) {
                *outStream << '(' << cmod << ')';
                mod.getFlagReg().outputText(*outStream, PrintDetail::sub_no_type, labelManager);
                *outStream << '\t';
            }

            if (mod.isSaturate()) *outStream << "(sat)";
            break;
        case ModPlacementType::Post:
        {
            bool havePostMod = false;
            auto startPostMod = [&]() {
                *outStream << (havePostMod ? ',' : '{');
                havePostMod = true;
            };
            auto printPostMod = [&](const char *name) {
                startPostMod(); *outStream << name;
            };

            SWSBInfo swsb = mod.getSWSB();
            if (swsb.hasSB()) {
                SBInfo sb = swsb.sb();
                startPostMod(); *outStream << '$' << sb.getID();
                if (sb.isSrc()) *outStream << ".src";
                if (sb.isDst() || (!isVariableLatency(op) && (swsb.dist() > 0)))
                    *outStream << ".dst";
            }
            if (swsb.dist() > 0) {
                startPostMod(); *outStream << swsb.pipe() << '@' << swsb.dist();
            }

            if (mod.isAlign16())    printPostMod("Align16");
            if (mod.isNoDDClr())    printPostMod("NoDDClr");
            if (mod.isNoDDChk())    printPostMod("NoDDChk");
            if (mod.getThreadCtrl() == ThreadCtrl::Atomic)             printPostMod("Atomic");
            if (!isGen12 && mod.getThreadCtrl() == ThreadCtrl::Switch) printPostMod("Switch");
            if (mod.isAccWrEn())    printPostMod("AccWrEn");
            if (mod.isCompact())    printPostMod("Compact");
            if (mod.isBreakpoint()) printPostMod("Debug");
            if (mod.isEOT())        printPostMod("EOT");

            if (havePostMod) *outStream << '}';
        }
        break;
    }
}

template <typename D, typename S0, typename S1, typename S2, typename Ext>
inline void AsmCodeGenerator::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, Ext ext)
{
    bool is3Src = !S2::emptyOp;

    InstructionModifier emod = mod ^ defaultModifier;
    auto esize = emod.getExecSize();

    if (is3Src && hardware < HW::Gen10)
        esize = std::min<int>(esize, 8);        // WA for IGA Align16 emulation issue

#ifdef NGEN_SAFE
    if (esize > 1 && dst.isScalar())
        throw invalid_execution_size_exception();
#endif

    dst.fixup(esize, defaultType, true);
    src0.fixup(esize, defaultType, false);
    src1.fixup(esize, defaultType, false);
    src2.fixup(esize, defaultType, false);

    outputMods(emod, op, ModPlacementType::Pre);

    *outStream << getMnemonic(op, hardware);
    ext(*outStream);
    *outStream << '\t';

    outputMods(emod, op, ModPlacementType::Mid);

    dst.outputText(*outStream, PrintDetail::hs, labelManager);                                   *outStream << '\t';
    src0.outputText(*outStream, is3Src ? PrintDetail::vs_hs : PrintDetail::full, labelManager);  *outStream << '\t';
    src1.outputText(*outStream, is3Src ? PrintDetail::vs_hs : PrintDetail::full, labelManager);  *outStream << '\t';
    src2.outputText(*outStream, PrintDetail::hs, labelManager);                                  *outStream << '\t';

    outputMods(emod, op, ModPlacementType::Post);
    *outStream << std::endl;
}

template <typename S1, typename ED, typename D>
inline void AsmCodeGenerator::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sf, RegData dst, RegData src0, S1 src1, ED exdesc, D desc)
{
    InstructionModifier emod = mod ^ defaultModifier;

    outputMods(emod, op, ModPlacementType::Pre);

    *outStream << getMnemonic(op, hardware);
    if (isGen12)
        *outStream << '.' << sf;
    *outStream << '\t';

    outputMods(emod, op, ModPlacementType::Mid);

    dst.outputText(*outStream, PrintDetail::base, labelManager);     *outStream << '\t';
    src0.outputText(*outStream, PrintDetail::base, labelManager);    *outStream << '\t';
    src1.outputText(*outStream, PrintDetail::base, labelManager);    *outStream << '\t';
    exdesc.outputText(*outStream, PrintDetail::sub_no_type, labelManager);  *outStream << '\t';
    desc.outputText(*outStream, PrintDetail::sub_no_type, labelManager);    *outStream << '\t';

    outputMods(emod, op, ModPlacementType::Post);
    *outStream << std::endl;
}

inline void AsmCodeGenerator::opDpas(Opcode op, const InstructionModifier &mod, int sdepth, int rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2)
{
    InstructionModifier emod = mod ^ defaultModifier;

    outputMods(emod, op, ModPlacementType::Pre);

    *outStream << getMnemonic(op, hardware) << '.' << sdepth << 'x' << rcount << '\t';

    outputMods(emod, op, ModPlacementType::Mid);

    dst.outputText(*outStream, PrintDetail::sub, labelManager);      *outStream << '\t';
    src0.outputText(*outStream, PrintDetail::sub, labelManager);     *outStream << '\t';
    src1.outputText(*outStream, PrintDetail::sub, labelManager);     *outStream << '\t';
    src2.outputText(*outStream, PrintDetail::sub, labelManager);     *outStream << '\t';

    outputMods(emod, op, ModPlacementType::Post);
    *outStream << std::endl;
}

template <typename D, typename S0>
inline void AsmCodeGenerator::opCall(Opcode op, const InstructionModifier &mod, D dst, S0 src0)
{
    InstructionModifier emod = mod | NoMask;

    outputMods(emod, op, ModPlacementType::Pre);

    *outStream << getMnemonic(op, hardware) << '\t';

    outputMods(emod, op, ModPlacementType::Mid);

    dst.outputText(*outStream, PrintDetail::sub, labelManager);          *outStream << '\t';
    src0.outputText(*outStream, PrintDetail::sub_no_type, labelManager); *outStream << '\t';

    outputMods(emod, op, ModPlacementType::Post);
    *outStream << std::endl;
}

template <typename S1>
inline void AsmCodeGenerator::opJmpi(Opcode op, const InstructionModifier &mod, S1 src1)
{
    InstructionModifier emod = mod | NoMask;

    outputMods(emod, op, ModPlacementType::Pre);

    *outStream << getMnemonic(op, hardware) << '\t';

    outputMods(emod, op, ModPlacementType::Mid);

    src1.outputText(*outStream, PrintDetail::sub_no_type, labelManager);  *outStream << '\t';

    outputMods(emod, op, ModPlacementType::Post);
    *outStream << std::endl;
}

template <typename S0>
inline void AsmCodeGenerator::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, S0 src0)
{
    InstructionModifier emod = mod ^ defaultModifier;

    outputMods(emod, op, ModPlacementType::Pre);

    *outStream << getMnemonic(op, hardware) << '.' << fc << '\t';

    outputMods(emod, op, ModPlacementType::Mid);

    src0.outputText(*outStream, PrintDetail::sub_no_type, labelManager);  *outStream << '\t';

    outputMods(emod, op, ModPlacementType::Post);
    *outStream << std::endl;
}

} /* namespace ngen */

#endif

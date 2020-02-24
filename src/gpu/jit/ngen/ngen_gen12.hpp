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

/*
 * Do not #include this file directly; ngen uses it internally.
 */

// Gen12 binary encoding.

// 24 bits of data common between src0 and src1 (lower 16 bits common with dst)
union BinaryOperand12 {
    uint32_t bits;
    struct {
        unsigned hs : 2;
        unsigned regFile : 1;
        unsigned subRegNum : 5;
        unsigned regNum : 8;
        unsigned addrMode : 1;          // = 0 (direct)
        unsigned width : 3;
        unsigned vs : 4;
    } direct;
    struct {
        unsigned hs : 2;
        unsigned addrOff : 10;
        unsigned addrReg : 4;
        unsigned addrMode : 1;          // = 1 (indirect)
        unsigned width : 3;
        unsigned vs : 4;
    } indirect;
};

// 16 bits of data common between dst, src0/1/2 for 3-source instructions
union TernaryOperand12 {
    uint16_t bits;
    struct {
        unsigned hs : 2;                    // sdepth for dst with dpas
        unsigned regFile : 1;
        unsigned subRegNum : 5;             // mme# for math
        unsigned regNum : 8;
    } direct;
};

union Instruction12 {
    struct {                            // Lower 35 bits are essentially common.
        unsigned opcode : 8;
        unsigned swsb : 8;
        unsigned execSize : 3;
        unsigned execOffset : 3;
        unsigned flagReg : 2;
        unsigned predCtrl : 4;
        unsigned predInv : 1;
        unsigned cmptCtrl : 1;
        unsigned debugCtrl : 1;
        unsigned maskCtrl : 1;
        //
        unsigned atomicCtrl : 1;
        unsigned accWrCtrl : 1;
        unsigned saturate : 1;
        unsigned : 29;
        //
        unsigned : 32;
        unsigned : 32;
    } common;
    struct {
        unsigned : 32;
        //
        unsigned : 3;
        unsigned dstAddrMode : 1;
        unsigned dstType : 4;
        unsigned src0Type : 4;
        unsigned src0Mods : 2;
        unsigned src0Imm : 1;
        unsigned src1Imm : 1;
        unsigned dst : 16;              // first 16 bits of BinaryOperand12
        //
        unsigned src0 : 24;             // BinaryOperand12
        unsigned src1Type : 4;
        unsigned cmod : 4;
        //
        unsigned src1 : 24;             // BinaryOperand12
        unsigned src1Mods : 2;
        unsigned _ : 6;
    } binary;
    struct {
        uint64_t _;
        uint32_t __;
        uint32_t value;
    } imm32;
    struct {
        uint64_t _;
        uint64_t value;
    } imm64;
    struct {
        unsigned : 32;                  // common
        unsigned : 3;
        unsigned src0VS0 : 1;
        unsigned dstType : 3;
        unsigned execType : 1;
        unsigned src0Type : 3;
        unsigned src0VS1 : 1;
        unsigned src0Mods : 2;
        unsigned src0Imm : 1;
        unsigned src2Imm : 1;
        unsigned dst : 16;              // TernaryOperand12 or immediate
        //
        unsigned src0 : 16;
        unsigned src2Type : 3;
        unsigned src1VS0 : 1;
        unsigned src2Mods : 2;          // subBytePrecision (DPAS)
        unsigned src1Mods : 2;          // subBytePrecision (DPAS)
        unsigned src1Type : 3;
        unsigned src1VS1 : 1;
        unsigned cmod : 4;              // same location as binary
        //
        unsigned src1 : 16;             // TernaryOperand12
        unsigned src2 : 16;             // TernaryOperand12 or immediate
    } ternary;
    struct {
        unsigned : 32;
        unsigned : 32;
        unsigned : 20;
        unsigned bfnCtrl03 : 4;
        unsigned : 4;
        unsigned bfnCtrl47 : 4;
        unsigned : 32;
    } bfn;
    struct {
        unsigned : 32;
        //
        unsigned : 11;
        unsigned rcount : 3;
        unsigned : 2;
        unsigned sdepth : 2;
        unsigned : 14;
        //
        unsigned : 20;
        unsigned src2SubBytePrecision : 2;
        unsigned src1SubBytePrecision : 2;
        unsigned : 8;
        //
        unsigned : 32;
    } dpas;
    struct {
        unsigned : 32;
        //
        unsigned : 1;
        unsigned fusionCtrl : 1;
        unsigned eot : 1;
        unsigned exDesc11_23 : 13;
        unsigned descIsReg : 1;
        unsigned exDescIsReg : 1;
        unsigned dstRegFile : 1;
        unsigned desc20_24 : 5;
        unsigned dstReg : 8;
        //
        unsigned exDesc24_25 : 2;
        unsigned src0RegFile : 1;
        unsigned desc25_29 : 5;
        unsigned src0Reg : 8;
        unsigned : 1;
        unsigned desc0_10 : 11;
        unsigned sfid : 4;
        //
        unsigned exDesc26_27 : 2;
        unsigned src1RegFile : 1;
        unsigned exDesc6_10 : 5;
        unsigned src1Reg : 8;
        unsigned : 1;
        unsigned desc11_19 : 9;
        unsigned desc30_31 : 2;
        unsigned exDesc28_31 : 4;
    } send;
    struct {
        unsigned : 32;
        unsigned : 8;
        unsigned exDescReg : 3;
        unsigned : 21;
        unsigned : 32;
        unsigned : 32;
    } sendIndirect;
    struct {
        unsigned : 32;                  // common
        unsigned : 1;
        unsigned branchCtrl : 1;
        unsigned : 30;
        unsigned uip : 32;
        unsigned jip : 32;
    } branches;
    uint64_t qword[2];

    constexpr Instruction12() : qword{0,0} {};
};

static_assert(sizeof(Instruction12) == 16, "Internal error: Instruction12 has been padded by the compiler.");

// Encoding routines.

static inline unsigned getTypecode12(DataType type)
{
    // :bf = 0b1101 (not in BSpec)
    static const uint8_t conversionTable[16] = {2,6,1,5,0,4,11,10,3,7,9,13,2,0,4,8};
    return conversionTable[static_cast<unsigned>(type) & 0xF];
}

template <bool dest, bool encodeHS = true>
static inline constexpr14 BinaryOperand12 encodeBinaryOperand12(const RegData &rd)
{
    BinaryOperand12 op{0};

#ifdef NGEN_SAFE
    if (rd.isInvalid()) throw invalid_object_exception();
#endif

    if (rd.isIndirect()) {
        op.indirect.addrOff = rd.getOffset();
        op.indirect.addrReg = rd.getIndirectOff();
        op.indirect.addrMode = 1;
        if (!dest) {
            op.indirect.vs = (rd.isVxIndirect()) ? 0xFFFF :
                                   (rd.getVS() == 0) ? 0 :
                                                       (1 + utils::log2(rd.getVS()));
        }
    } else {
        op.direct.regFile = getRegFile(rd);
        op.direct.subRegNum = rd.getByteOffset();
        op.direct.regNum = rd.getBase();
        op.direct.addrMode = 0;
        if (!dest)
            op.direct.vs = (rd.getVS() == 0) ? 0 : (1 + utils::log2(rd.getVS()));
    }

    if (encodeHS)
        op.direct.hs = (rd.getHS() == 0) ? 0 : (1 + utils::log2(rd.getHS()));

    if (!dest) op.direct.width = utils::log2(rd.getWidth());

    return op;
}

template <bool dest>
static inline constexpr14 BinaryOperand12 encodeBinaryOperand12(const ExtendedReg &reg)
{
    auto op = encodeBinaryOperand12<dest>(reg.getBase());
    op.direct.subRegNum = reg.getMMENum();

    return op;
}

template <bool dest, bool encodeHS = true>
static inline constexpr14 TernaryOperand12 encodeTernaryOperand12(const RegData &rd)
{
#ifdef NGEN_SAFE
    if (rd.isInvalid()) throw invalid_object_exception();
    if (rd.isIndirect()) throw invalid_operand_exception();
#endif

    TernaryOperand12 op{0};

    if (encodeHS) {
        if (dest)
            op.direct.hs = utils::log2(rd.getHS());
        else
            op.direct.hs = (rd.getHS() == 0) ? 0 : (1 + utils::log2(rd.getHS()));
    }

    op.direct.regFile = getRegFile(rd);
    op.direct.subRegNum = rd.getByteOffset();
    op.direct.regNum = rd.getBase();

    return op;
}

template <bool dest>
static inline constexpr14 TernaryOperand12 encodeTernaryOperand12(const ExtendedReg &reg)
{
    auto op = encodeTernaryOperand12<dest>(reg.getBase());
    op.direct.subRegNum = reg.getMMENum();

    return op;
}

static inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod)
{
    i.common.opcode = static_cast<unsigned>(opcode);
    i.common.swsb = mod.parts.swsb;
    i.common.execSize = mod.parts.eSizeField;
    i.common.execOffset = mod.parts.chanOff;
    i.common.flagReg = (mod.parts.flagRegNum << 1) | mod.parts.flagSubRegNum;
    i.common.predCtrl = mod.parts.predCtrl;
    i.common.predInv = mod.parts.predInv;
    i.common.cmptCtrl = mod.parts.cmptCtrl;
    i.common.debugCtrl = mod.parts.debugCtrl;
    i.common.maskCtrl = mod.parts.maskCtrl;
    i.common.atomicCtrl = mod.parts.threadCtrl;
    i.common.accWrCtrl = mod.parts.accWrCtrl;
    i.common.saturate = mod.parts.saturate;
}

static inline unsigned encodeTernaryVS01(const RegData &rd)
{
    switch (rd.getVS()) {
        case 0: return 0;
        case 1: return 1;
        case 4: return 2;
        case 8: return 3;
        default:
#ifdef NGEN_SAFE
            if (rd.getHS() == 0)
                throw invalid_region_exception();
#endif
            return 3;
    }
}

static inline unsigned encodeTernaryVS01(const ExtendedReg &reg)
{
    return encodeTernaryVS01(reg.getBase());
}

template <typename D, typename S0, typename S1, typename S2>
static inline void encodeTernaryTypes(Instruction12 &i, D dst, S0 src0, S1 src1, S2 src2)
{
    auto dtype = getTypecode12(dst.getType());
    auto s0type = getTypecode12(src0.getType());
    auto s1type = getTypecode12(src1.getType());
    auto s2type = getTypecode12(src2.getType());

    i.ternary.execType = (dtype >> 3);
    i.ternary.dstType  = dtype;
    i.ternary.src0Type = s0type;
    i.ternary.src1Type = s1type;
    i.ternary.src2Type = s2type;

#ifdef NGEN_SAFE
    if (((dtype & s0type & s1type & s2type) ^ (dtype | s0type | s1type | s2type)) & 8)
        throw ngen::invalid_type_exception();
#endif
}

template <typename S0>
static inline void encodeTernarySrc0(Instruction12 &i, S0 src0)
{
    i.ternary.src0 = encodeTernaryOperand12<false>(src0).bits;
    i.ternary.src0Mods = src0.getMods();

    auto vs0 = encodeTernaryVS01(src0);

    i.ternary.src0VS0 = vs0;
    i.ternary.src0VS1 = vs0 >> 1;
}

static inline void encodeTernarySrc0(Instruction12 &i, const Immediate &src0)
{
    i.ternary.src0Imm = true;
    i.ternary.src0 = static_cast<uint64_t>(src0);
}

template <typename S1>
static inline void encodeTernarySrc1(Instruction12 &i, S1 src1)
{
    i.ternary.src1 = encodeTernaryOperand12<false>(src1).bits;

    i.ternary.src1Mods = src1.getMods();

    auto vs1 = encodeTernaryVS01(src1);
    i.ternary.src1VS0 = vs1;
    i.ternary.src1VS1 = vs1 >> 1;
}

template <typename S2>
static inline void encodeTernarySrc2(Instruction12 &i, S2 src2)
{
    i.ternary.src2 = encodeTernaryOperand12<false>(src2).bits;
    i.ternary.src2Mods = src2.getMods();
}

static inline void encodeTernarySrc2(Instruction12 &i, const Immediate &src2)
{
    i.ternary.src2Imm = true;
    i.ternary.src2 = static_cast<uint64_t>(src2);
}

static inline void encodeSendExDesc(Instruction12 &i, uint32_t exdesc)
{
    i.send.eot = (exdesc >> 5);
    i.send.exDesc6_10 = (exdesc >> 6);
    i.send.exDesc11_23 = (exdesc >> 11);
    i.send.exDesc24_25 = (exdesc >> 24);
    i.send.exDesc26_27 = (exdesc >> 26);
    i.send.exDesc28_31 = (exdesc >> 28);
}

static inline void encodeSendExDesc(Instruction12 &i, RegData exdesc)
{
#ifdef NGEN_SAFE
    // Only a0.x:ud is allowed for extended descriptor.
    if (!exdesc.isARF() || exdesc.getARFType() != ARFType::a || exdesc.getARFBase() != 0 || exdesc.getType() != DataType::ud)
        throw invalid_arf_exception();
#endif
    i.sendIndirect.exDescReg = exdesc.getOffset();
    i.send.exDescIsReg = true;
}

static inline void encodeSendDesc(Instruction12 &i, uint32_t desc)
{
    i.send.desc0_10 = (desc >> 0);
    i.send.desc11_19 = (desc >> 11);
    i.send.desc20_24 = (desc >> 20);
    i.send.desc25_29 = (desc >> 25);
    i.send.desc30_31 = (desc >> 30);
}

static inline void encodeSendDesc(Instruction12 &i, RegData desc)
{
#ifdef NGEN_SAFE
    // Only a0.0:ud is allowed for desc.
    if (!desc.isARF() || desc.getARFType() != ARFType::a || desc.getARFBase() != 0 || desc.getOffset() != 0)
        throw invalid_arf_exception();
#endif
    i.send.descIsReg = true;
}

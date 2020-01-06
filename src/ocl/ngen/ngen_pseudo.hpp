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


// Pseudo-instructions and macros.
template <typename DT = void>
void min(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    sel(mod | lt | f0[0], dst, src0, src1);
}
template <typename DT = void>
void min(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    sel(mod | lt | f0[0], dst, src0, src1);
}
template <typename DT = void>
void max(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    sel(mod | ge | f0[0], dst, src0, src1);
}
template <typename DT = void>
void max(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    sel(mod | ge | f0[0], dst, src0, src1);
}
template <typename DT = void>
void bfi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, const RegData &src3) {
    bfi1(mod, dst, src0, src1);
    bfi2(mod, dst, dst, src2, src3);
}

// Brief math instructions.
template <typename DT = void>
void cos(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::cos, dst, src0);
}
template <typename DT = void>
void exp(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::exp, dst, src0);
}
template <typename DT = void>
void fdiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    math<DT>(mod, MathFunction::fdiv, dst, src0, src1);
}
template <typename DT = void>
void fdiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    math<DT>(mod, MathFunction::fdiv, dst, src0, src1);
}
template <typename DT = void>
void idiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    math<DT>(mod, MathFunction::idiv, dst, src0, src1);
}
template <typename DT = void>
void idiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    math<DT>(mod, MathFunction::idiv, dst, src0, src1);
}
template <typename DT = void>
void inv(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::inv, dst, src0);
}
template <typename DT = void>
void invm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1) {
    math<DT>(mod, MathFunction::invm, dst, src0, src1);
}
template <typename DT = void>
void iqot(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    math<DT>(mod, MathFunction::iqot, dst, src0, src1);
}
template <typename DT = void>
void iqot(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    math<DT>(mod, MathFunction::iqot, dst, src0, src1);
}
template <typename DT = void>
void irem(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    math<DT>(mod, MathFunction::irem, dst, src0, src1);
}
template <typename DT = void>
void irem(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    math<DT>(mod, MathFunction::irem, dst, src0, src1);
}
template <typename DT = void>
void log(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::log, dst, src0);
}
template <typename DT = void>
void pow(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
    math<DT>(mod, MathFunction::pow, dst, src0, src1);
}
template <typename DT = void>
void pow(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
    math<DT>(mod, MathFunction::pow, dst, src0, src1);
}
template <typename DT = void>
void rsqt(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::rsqt, dst, src0);
}
template <typename DT = void>
void rsqtm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0) {
    math<DT>(mod, MathFunction::rsqtm, dst, src0);
}
template <typename DT = void>
void sin(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::sin, dst, src0);
}
template <typename DT = void>
void sqt(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
    math<DT>(mod, MathFunction::sqt, dst, src0);
}

// IEEE 754-compliant divide math macro sequence.
//   Requires GRFs initialized with 0.0 and 1.0, as well as temporary GRFs (4 for single precision, 5 for double precision).
//   dst, num, denom must be distinct GRFs.
template <typename DT = void, typename A>
void fdiv_ieee(const InstructionModifier &mod, FlagRegister flag, RegData dst, RegData num, RegData denom,
               RegData zero, RegData one, const A &tmp)
{
    DataType dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = dst.getType();

    Label labelSkip;

    switch (dt) {
        case DataType::hf:
            fdiv<DT>(mod, dst, num, denom);
            break;
        case DataType::f:
            invm<DT>(mod | eo | flag,         dst | mme0,      num | nomme,   denom | nomme);
            if_(mod | ~flag, labelSkip);

            madm<DT>(mod, tmp[0] | mme1,     zero | nomme,     num | nomme,     dst | mme0);
            madm<DT>(mod, tmp[1] | mme2,      one | nomme,  -denom | nomme,     dst | mme0);
            madm<DT>(mod, tmp[2] | mme3,      dst | mme0,   tmp[1] | mme2,      dst | mme0);
            madm<DT>(mod, tmp[3] | mme4,      num | nomme,  -denom | nomme,  tmp[0] | mme1);
            madm<DT>(mod, tmp[0] | mme5,   tmp[0] | mme1,   tmp[3] | mme4,   tmp[2] | mme3);
            madm<DT>(mod, tmp[1] | mme6,      num | nomme,  -denom | nomme,  tmp[0] | mme5);
            madm<DT>(mod,    dst | nomme,  tmp[0] | mme5,   tmp[1] | mme6,   tmp[2] | mme3);

            mark(labelSkip);
            endif(mod);
            break;
        case DataType::df:
            invm<DT>(mod | eo | flag,         dst | mme0,      num | nomme,   denom | nomme);
            if_(mod | ~flag, labelSkip);

            madm<DT>(mod, tmp[0] | mme1,     zero | nomme,     num | nomme,     dst | mme0);
            madm<DT>(mod, tmp[1] | mme2,      one | nomme,  -denom | nomme,     dst | mme0);
            madm<DT>(mod, tmp[2] | mme3,      num | nomme,  -denom | nomme,  tmp[0] | mme1);
            madm<DT>(mod, tmp[3] | mme4,      dst | mme0,   tmp[1] | mme2,      dst | mme0);
            madm<DT>(mod, tmp[4] | mme5,      one | nomme,  -denom | nomme,  tmp[3] | mme4);
            madm<DT>(mod,    dst | mme6,      dst | mme0,   tmp[1] | mme2,   tmp[3] | mme4);
            madm<DT>(mod, tmp[0] | mme7,   tmp[0] | mme1,   tmp[2] | mme3,   tmp[3] | mme4);
            madm<DT>(mod, tmp[3] | mme0,   tmp[3] | mme4,      dst | mme6,   tmp[4] | mme5);
            madm<DT>(mod, tmp[2] | mme1,      num | nomme,  -denom | nomme,  tmp[0] | mme7);
            madm<DT>(mod,    dst | nomme,  tmp[0] | mme7,   tmp[2] | mme1,   tmp[3] | mme0);

            mark(labelSkip);
            endif(mod);
            break;
        default:
#ifdef NGEN_SAFE
            throw invalid_type_exception();
#endif
            break;
    }
}

// IEEE 754-compliant reciprocal math macro sequence. Only needed for double precision (use math.inv for single/half precision).
//   Requires GRF initialized with 1.0, as well as 3 temporary GRFs.
//   dst and src must be distinct GRFs.
template <typename DT = void, typename A>
void inv_ieee(const InstructionModifier &mod, FlagRegister flag, RegData dst, RegData src, RegData one, const A &tmp)
{
    DataType dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = dst.getType();

    Label labelSkip;

    switch (dt) {
        case DataType::hf:
            inv<DT>(mod, dst, src);
            break;
        case DataType::f:
            invm<DT>(mod | eo | flag,         dst | mme0,      one | nomme,     src | nomme);
            if_(mod | ~flag, labelSkip);

            madm<DT>(mod, tmp[1] | mme2,      one | nomme,    -src | nomme,     dst | mme0);
            madm<DT>(mod, tmp[2] | mme3,      dst | mme0,   tmp[1] | mme2,      dst | mme0);
            madm<DT>(mod, tmp[0] | mme5,      dst | mme0,   tmp[1] | mme2,   tmp[2] | mme3);
            madm<DT>(mod, tmp[1] | mme6,      one | nomme,    -src | nomme,  tmp[0] | mme5);
            madm<DT>(mod,    dst | nomme,  tmp[0] | mme5,   tmp[1] | mme6,   tmp[2] | mme3);

            mark(labelSkip);
            endif(mod);
            break;
        case DataType::df:
            invm<DT>(mod | eo | flag,        dst | mme0,      one | nomme,     src | nomme);
            if_(mod | ~flag, labelSkip);

            madm<DT>(mod, tmp[0] | mme2,     one | nomme,    -src | nomme,     dst | mme0);
            madm<DT>(mod, tmp[1] | mme4,     dst | mme0,   tmp[0] | mme2,      dst | mme0);
            madm<DT>(mod, tmp[2] | mme5,     one | nomme,    -src | nomme,  tmp[1] | mme4);
            madm<DT>(mod,    dst | mme6,     dst | mme0,   tmp[0] | mme2,   tmp[1] | mme4);
            madm<DT>(mod, tmp[1] | mme0,  tmp[1] | mme4,      dst | mme6,   tmp[2] | mme5);
            madm<DT>(mod, tmp[0] | mme1,     one | nomme,    -src | nomme,     dst | mme6);
            madm<DT>(mod,    dst | nomme,    dst | mme6,   tmp[0] | mme1,   tmp[1] | mme0);

            mark(labelSkip);
            endif(mod);
            break;
        default:
#ifdef NGEN_SAFE
            throw invalid_type_exception();
#endif
            break;
    }
}

// IEEE 754-compliant square root macro sequence.
//   Requires GRFs initialized with 0.0 and 0.5 (also 1.0 for double precision),
//     and temporary GRFs (3 for single precision, 4 for double precision).
//   dst and src must be distinct GRFs.
template <typename DT = void, typename A>
void sqt_ieee(const InstructionModifier &mod, FlagRegister flag, RegData dst, RegData src,
               RegData zero, RegData oneHalf, RegData one, const A &tmp)
{
    DataType dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = dst.getType();

    Label labelSkip;

    switch (dt) {
        case DataType::hf:
            sqt<DT>(mod, dst, src);
            break;
        case DataType::f:
            rsqtm<DT>(mod | eo | flag,        dst | mme0,       src | nomme);
            if_(mod | ~flag, labelSkip);

            madm<DT>(mod, tmp[0] | mme1,     zero | nomme,  oneHalf | nomme,     dst | mme0);
            madm<DT>(mod, tmp[1] | mme2,     zero | nomme,      src | nomme,     dst | mme0);
            madm<DT>(mod, tmp[2] | mme3,  oneHalf | nomme,  -tmp[1] | mme2,   tmp[0] | mme1);
            madm<DT>(mod, tmp[0] | mme4,   tmp[0] | mme1,    tmp[2] | mme3,   tmp[0] | mme1);
            madm<DT>(mod,    dst | mme5,   tmp[1] | mme2,    tmp[2] | mme3,   tmp[1] | mme2);
            madm<DT>(mod, tmp[2] | mme6,      src | nomme,     -dst | mme5,      dst | mme5);
            madm<DT>(mod,    dst | nomme,     dst | mme5,    tmp[0] | mme4,   tmp[2] | mme6);

            mark(labelSkip);
            endif(mod);
            break;
        case DataType::df:
            rsqtm<DT>(mod | eo | flag,        dst | mme0,       src | nomme);
            if_(mod | ~flag, labelSkip);

            madm<DT>(mod, tmp[0] | mme1,     zero | mme0,   oneHalf | nomme,     dst | mme0);
            madm<DT>(mod, tmp[1] | mme2,     zero | mme0,       src | nomme,     dst | mme0);
            madm<DT>(mod, tmp[2] | mme3,  oneHalf | nomme,  -tmp[1] | mme2,   tmp[0] | mme1);
            madm<DT>(mod, tmp[3] | mme4,      one | nomme,  oneHalf | nomme,     dst | nomme);
            madm<DT>(mod, tmp[3] | mme5,      one | nomme,   tmp[3] | mme4,   tmp[2] | mme3);
            madm<DT>(mod,    dst | mme6,     zero | mme0,    tmp[2] | mme3,   tmp[1] | mme2);
            madm<DT>(mod, tmp[2] | mme7,     zero | mme0,    tmp[2] | mme3,   tmp[0] | mme1);
            madm<DT>(mod,    dst | mme6,   tmp[1] | mme2,    tmp[3] | mme5,      dst | mme6);
            madm<DT>(mod, tmp[3] | mme5,   tmp[0] | mme1,    tmp[3] | mme5,   tmp[2] | mme7);
            madm<DT>(mod, tmp[0] | mme1,      src | nomme,     -dst | mme6,      dst | mme6);
            madm<DT>(mod,    dst | nomme,     dst | mme6,    tmp[0] | mme1,   tmp[3] | mme5);

            mark(labelSkip);
            endif(mod);
            break;
        default:
#ifdef NGEN_SAFE
            throw invalid_type_exception();
#endif
            break;
    }
}

// Thread spawner messages.
void threadend(const InstructionModifier &mod, const RegData &r0_info) {
    auto dmSave = defaultModifier;
    defaultModifier = InstructionModifier();

    send(NoMask | 8 | EOT | mod, null, r0_info, 0x27, 0x2000010);

    defaultModifier = dmSave;
}

void threadend(const RegData &r0_info) { threadend(InstructionModifier(), r0_info); }

// Gateway messages.
void barriermsg(const InstructionModifier &mod, const GRF &header)
{
    auto dmSave = defaultModifier;
    defaultModifier = InstructionModifier();

    send(NoMask | 1 | mod, null, header, 0x3, 0x2000004);

    defaultModifier = dmSave;
}

void barriermsg(const GRF &header) { barriermsg(InstructionModifier(), header); }


void barriersignal(const InstructionModifier &mod, const GRF &temp, const GRF &r0_info = r0)
{
    auto dmSave = defaultModifier;
    defaultModifier = InstructionModifier();

    and_(NoMask | 8, temp.ud(), r0_info.ud(2), uint32_t((hardware >= HW::Gen11) ? 0x7F000000 : 0x8F000000));
    barriermsg(mod, temp);

    defaultModifier = dmSave;
}

void barriersignal(const GRF &temp, const GRF &r0_info = r0) { barriersignal(InstructionModifier(), temp, r0_info); }

void barrierwait()
{
    if (isGen12)
        sync(SyncFunction::bar, NoMask);
    else {
        auto dmSave = defaultModifier;
        defaultModifier = InstructionModifier();

        wait(NoMask, n0[0]);

        defaultModifier = dmSave;
    }
}

void barrier(const GRF &temp, const GRF &r0_info = r0)
{
    barriersignal(temp, r0_info);
    barrierwait();
}

// Data port messages.
template <typename DataSpec>
void load(const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const RegData &addr)
{
    MessageDescriptor desc;
    ExtendedMessageDescriptor exdesc;

    encodeLoadDescriptors(desc, exdesc, mod, spec, base);
    send(mod, dst, addr, exdesc.all, desc.all);
}

template <typename DataSpec>
void store(const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const RegData &addr, const RegData &data)
{
    MessageDescriptor desc;
    ExtendedMessageDescriptor exdesc;

    encodeStoreDescriptors(desc, exdesc, mod, spec, base);
    sends(mod, NullRegister(), addr, data, exdesc.all, desc.all);
}

// For write-only atomics, dest is null; for unary atomics, data is null.
template <typename DataSpec>
void atomic(AtomicOp op, const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const RegData &addr, const RegData &data = NullRegister())
{
    MessageDescriptor desc;
    ExtendedMessageDescriptor exdesc;

    encodeAtomicDescriptors(desc, exdesc, op, mod, dst, spec, base);
    if (data.isNull())
        send(mod, dst, addr, exdesc.all, desc.all);
    else
        sends(mod, dst, addr, data, exdesc.all, desc.all);
}

// Global memory fence.
void memfence(const InstructionModifier &mod, const RegData &dst, const RegData &header = GRF(0))
{
    auto dmSave = defaultModifier;
    defaultModifier = InstructionModifier();

    send(NoMask | 8 | mod, dst, header, 0xA, 0x219E000);

    defaultModifier = dmSave;
}

void memfence(const RegData &dst, const RegData &header = GRF(0)) { memfence(InstructionModifier(), dst, header); }

// SLM-only memory fence.
void slmfence(const InstructionModifier &mod, const RegData &dst, const RegData &header = GRF(0))
{
    auto dmSave = defaultModifier;
    defaultModifier = InstructionModifier();

    send(NoMask | 8 | mod, dst, header, 0xA, 0x219E0FE);

    defaultModifier = dmSave;
}

void slmfence(const RegData &dst, const RegData &header = GRF(0)) { slmfence(InstructionModifier(), dst, header); }

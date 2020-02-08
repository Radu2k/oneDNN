
/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the apache License, Version 2.0 (the "License");
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

#include "ocl/gemm/ngen_gen12hp_systolic_gemm_kernel.hpp"

using namespace ngen;

namespace dnnl {
namespace impl {
namespace ocl {

void ngen_gen12hp_systolic_gemm_kernel_t::zero_c() {
    // Use floats to avoid contentions with integer unit.
    for (int i = 0; i < 192; i += 2)
        mov<float>(16, c_regs[i], 0.0f);
}

void ngen_gen12hp_systolic_gemm_kernel_t::scattered_setup_c(
        int stride, bool load) {
    auto c = load ? c_ptr_mem : uc_base;

    // Set up SIMD16 scattered access pointers to emulate block access to C
    //   (2 columns x 8 regs/column).
    mov<uint16_t>(4, uheaders[15],
            Immediate::uv(0 * stride, 1 * stride, 2 * stride, 3 * stride, 0, 0,
                    0, 0));
    add<uint64_t>(4 | SWSB<AllPipes>(1), uheaders[0], c, uheaders[15].uw());
    add<uint64_t>(4 | SWSB(1), uheaders[1], uheaders[0], uint16_t(stride * 4));
    add<uint64_t>(8 | SWSB(1), uheaders[2], uheaders[0], uint16_t(stride * 8));
    add<uint64_t>(8 | SWSB(2), uheaders[4], uheaders[0], uint16_t(stride * 16));
    add<uint64_t>(8 | SWSB(3), uheaders[6], uheaders[0], uint16_t(stride * 24));
    for (int q = 8; q < 16; q += 2)
        add<uint64_t>(8 | SWSB(4), uheaders[q], uheaders[q - 8], uldc);
}

void ngen_gen12hp_systolic_gemm_kernel_t::block_setup_c(
        bool remainder, bool load) {
    auto c = load ? c_ptr_mem : uc_base;
    if (remainder) {
        // 8 blocks, each 16x1.
        mov<uint64_t>(1, uheaders[0][0], c);
        add<uint64_t>(1 | SWSB<int>(3), uheaders[1][0], c,
                uint16_t(getBytes(cfg.c_type) * 16));
        add<uint64_t>(8 | SWSB<AllPipes>(1), uheaders[2], uheaders[0], uldc);
        add<uint64_t>(8 | SWSB(2), uheaders[4], uheaders[0], uldc_x2);
        add<uint64_t>(8 | SWSB(2), uheaders[6], uheaders[2], uldc_x2);
        for (int q = 8; q < 16; q += 2)
            add<uint64_t>(8 | SWSB(4), uheaders[q], uheaders[q - 8], uldc_x4);
    } else {
        // 4 blocks, each 32x1.
        mov<uint64_t>(1, uheaders[0][0], c);
        add<uint64_t>(1 | SWSB<int>(3), uheaders[1][0], c, uldc);
        add<uint64_t>(8 | SWSB<AllPipes>(1), uheaders[2], uheaders[0], uldc_x2);
        add<uint64_t>(8 | SWSB(2), uheaders[4], uheaders[0], uldc_x4);
        add<uint64_t>(8 | SWSB(2), uheaders[6], uheaders[2], uldc_x4);
    }
}

int ngen_gen12hp_systolic_gemm_kernel_t::interleave(int j) {
    // Convert logical column index in C to the corresponding interleaved index.
    bool second = (j >= 24);
    if (second) j -= 24;
    return ((j & ~3) << 1) + (int(second) << 2) + (j & 3);
}

void ngen_gen12hp_systolic_gemm_kernel_t::load_c(
        bool remainder, bool c_align16) {
    Label done;

    // Get configuration options.
    bool alpha1 = cfg.alpha1;
    bool beta0 = cfg.beta0;
    bool beta1 = cfg.beta1;

    const auto c_elem_bytes = getBytes(cfg.c_type);
    bool c32 = (c_elem_bytes == 4);
    int loads_per_col = (c32 && c_align16 && !remainder) ? 1 : 2;

    if (beta0 && alpha1) {
        sync(SyncFunction::nop, SWSB<AllPipes>(1));
        return; // Nothing to do.
    }

    // Get the bank ID for a given register.
    auto bank = [](const RegData &r) { return (r.getBase() & 2) >> 1; };

    // Load a 32x4 block of C and increment pointers.
    auto c_load = [&](int j0) {
        Label skip;

        if (beta0) return;

        if (remainder) {
            // Check load j0 + 1.
            cmp(1 | gt | f1[1], null.ud(), un_rem, uint16_t(j0 + 1));
        }

        for (int j = j0; j < j0 + 4; j++) {
            auto jj = (j & 7);

            if (remainder) {
                // Skip this load if masked off. Otherwise, prepare next flag.
                jmpi(1 | ~f1[j & 1], skip);
                if (j + 2 < 48)
                    cmp(1 | gt | f1[j & 1], null.ud(), un_rem, uint16_t(j + 2));
            }

            if (c_align16) {
                if (remainder) {
                    // Block read with masks.
                    assert(c32);
                    load(16 | f0[0] | SWSB(SBID(2 * jj + 0), 7),
                            utemp[jj * 4 + 0], block_oword(4), A64,
                            uheaders[2 * jj + 0]);
                    load(16 | f0[1] | SWSB(SBID(2 * jj + 1), 7),
                            utemp[jj * 4 + 2], block_oword(4), A64,
                            uheaders[2 * jj + 1]);
                } else {
                    // Block read.
                    load(16 | SWSB(SBID(jj), 7),
                            utemp[jj * 4 + 4 - c_elem_bytes],
                            aligned_block_oword(c_elem_bytes * 2), A64,
                            uheaders[jj]);
                }
            } else {
                // Scattered byte or dword load, possibly masked.
                auto j1 = (j & 1);
                auto mod0 = 16 | SWSB(SBID(2 * jj + 0), 7);
                auto mod1 = 16 | SWSB(SBID(2 * jj + 1), 5);
                if (remainder) {
                    mod0 = mod0 | f0[0];
                    mod1 = mod1 | f0[1];
                }
                if (c32) {
                    load(mod0, utemp[jj * 4 + 0], scattered_dword(1), A64,
                            uheaders[8 * j1 + 0]);
                    load(mod1, utemp[jj * 4 + 2], scattered_dword(1), A64,
                            uheaders[8 * j1 + 4]);
                } else {
                    load(mod0, utemp[jj * 4 + 0], scattered_byte(c_elem_bytes),
                            A64, uheaders[8 * j1 + 0]);
                    load(mod1, utemp[jj * 4 + 2], scattered_byte(c_elem_bytes),
                            A64, uheaders[8 * j1 + 4]);
                }
                if (j + 2 < 48) {
                    add<uint64_t>(8 | SBID(2 * jj + 0).src,
                            uheaders[8 * j1 + 0], uheaders[8 * j1 + 0],
                            uldc_x2);
                    add<uint64_t>(8, uheaders[8 * j1 + 2], uheaders[8 * j1 + 2],
                            uldc_x2);
                    add<uint64_t>(8 | SBID(2 * jj + 1).src,
                            uheaders[8 * j1 + 4], uheaders[8 * j1 + 4],
                            uldc_x2);
                    add<uint64_t>(8, uheaders[8 * j1 + 6], uheaders[8 * j1 + 6],
                            uldc_x2);
                }
            }
        }

        if (c_align16 && (j0 + 8 < 48)) {
            // Increment pointers.
            for (int j = j0; j < j0 + 4; j++) {
                auto jj = (j & 7);
                if (remainder) {
                    add<uint64_t>(1 | SBID(2 * jj + 0).src,
                            uheaders[2 * jj + 0], uheaders[2 * jj + 0],
                            uldc_x8);
                    add<uint64_t>(1 | SBID(2 * jj + 1).src,
                            uheaders[2 * jj + 1], uheaders[2 * jj + 1],
                            uldc_x8);
                } else
                    add<uint64_t>(1 | SBID(jj).src, uheaders[jj], uheaders[jj],
                            uldc_x8);
            }
        }

        mark(skip);
    };

    if (remainder) {
        // Do the first n compare.
        cmp(1 | gt | f1[0] | SWSB(6), null.ud(), un_rem, uint32_t(0));
    }

    // Set up headers.
    if (c_align16)
        block_setup_c(remainder, true);
    else
        scattered_setup_c(c_elem_bytes, true);

    // Get first load ready.
    c_load(0);

    for (int j0 = 0; j0 < 48; j0 += 4) {
        int j0_4 = j0 & 4;

        auto acc = c_regs[interleave(j0)];
        auto acc_stride = 48;

        // Get (sub)register in loaded C submatrix at offset (ii*8, jj).
        auto get_load_reg = [&](DataType dt, int ii, int jj) {
            auto bytes = c_align16 ? getBytes(dt) : 4;
            auto stride = bytes / getBytes(dt);
            auto per_reg = 4 / bytes;
            auto reg = utemp[j0_4 * 4 + (4 - bytes) + (ii / per_reg) + jj * 4];
            auto off = (ii % per_reg) * 8;

            return reg.sub(off, dt)(stride);
        };

        // Load C block ahead of time for next loop, and check for loop exit.
        if ((j0 + 4) < 48) {
            c_load(j0 + 4);
            if (remainder)
                cmp(1 | gt | f1[1], null.ud(), un_rem, uint16_t(j0 + 4));
        }

        // Premultiply by alpha if both alpha is not 1, unless beta = 1 (use FMA later instead).
        InstructionModifier swsb_pa {};
        if (!alpha1 && !beta1) {
            for (int ii = 0; ii < 4; ii++)
                for (int jj = 0; jj < 4; jj += 2)
                    mul<float>(16, acc + (jj + ii * acc_stride),
                            acc + (jj + ii * acc_stride),
                            ualpha_regs[!bank(acc + jj)]);
            swsb_pa = SWSB<float>(7);
        }

        // Wait for loads. Use SBIDs instead once auto-SWSB implemented in nGEN.
        if (!beta0) {
            uint16_t sbid_mask = (loads_per_col == 2) ? (0xFF << (j0_4 * 2))
                                                      : (0xF << j0_4);
            sync(SyncFunction::allwr, sbid_mask);
        }

        // Half-precision C must be upconverted to single precision separately (no hf/f mixed mode support in Gen12HP)
        auto old_type = cfg.c_type;
        if (cfg.c_type == DataType::hf) {
            for (int jj = 0; jj < 4; jj++) {
                for (int ii = 0; ii < 4; ii += 2) {
                    auto old_hf = get_load_reg(DataType::hf, ii, jj);
                    auto old_f = get_load_reg(DataType::f, ii, jj);
                    mov(16, old_f, old_hf);
                }
            }
            old_type = DataType::f;
            sync(SyncFunction::nop,
                    SWSB<float>(1)); // Temporary until auto-SWSB supported.
        }

        // Main alpha/beta scaling.
        for (int ii = 0; ii < 4; ii++) {
            for (int jj = 0; jj < 4; jj++) {
                GRF a_reg = acc + (jj + ii * acc_stride);
                a_reg = a_reg.retype(cfg.acc_type);
                auto oreg = get_load_reg(old_type, ii, jj);
                int b = !bank(a_reg);

                if (beta0) {
                    /* no op */
                } else if (beta1) {
                    if (alpha1)
                        add(8, a_reg, a_reg, oreg);
                    else
                        mad(8 | swsb_pa, a_reg, oreg, a_reg, ualpha_regs[b]);
                } else
                    mad(8 | swsb_pa, a_reg, a_reg, oreg, ubeta_regs[b]);
            }
        }

        // Early exit if no more columns to load.
        if (remainder && (j0 + 4 < 48)) jmpi(1 | ~f1[1], done);
    }

    mark(done);
    sync(SyncFunction::allrd, SWSB<AllPipes>(1));
}

void ngen_gen12hp_systolic_gemm_kernel_t::store_c(
        bool remainder, bool c_align16) {
    Label done;

    const auto c_elem_bytes = getBytes(cfg.c_type);
    bool c32 = (c_elem_bytes == 4);

    if (remainder) {
        // Do the first two n compares.
        cmp(1 | gt | f1[0] | SWSB(6), null.ud(), un_rem, uint32_t(0));
        cmp(1 | gt | f1[1] | SWSB(7), null.ud(), un_rem, uint32_t(1));
    }

    // Set up headers. TODO: reuse headers from load where possible.
    if (c_align16)
        block_setup_c(remainder, false);
    else
        scattered_setup_c(c_elem_bytes, false);

    for (int j0 = 0; j0 < 48; j0 += 4) {
        int j0_4 = j0 & 4;

        auto acc = c_regs[interleave(j0)];
        auto acc_stride = 48;

        // Get (sub)register in stored C submatrix at offset (ii*8, jj).
        auto get_store_reg = [&](int ii, int jj) {
            auto bytes = c_align16 ? c_elem_bytes : 4;
            auto stride = bytes / c_elem_bytes;
            auto per_reg = 4 / bytes;
            auto reg = utemp[j0_4 * 4 + (ii / per_reg) + jj * 4];
            auto off = (ii % per_reg) * 8;

            return reg.sub(off, cfg.c_type)(stride);
        };

        // 4x4 transpose of 8x1 blocks, downconverting if necessary.
        for (int ii = 0; ii < 4; ii++) {
            for (int jj = 0; jj < 4; jj++) {
                GRF a_reg = acc + (jj + ii * acc_stride);
                a_reg = a_reg.retype(cfg.acc_type);
                auto dreg = get_store_reg(ii, jj);

                if (a_reg.getType() == dreg.getType()) {
                    // Use float moves for raw moves.
                    a_reg.setType(DataType::f);
                    dreg.setType(DataType::f);
                }

                mov<float>(8, dreg, a_reg);
            }
        }

        // Store C.
        for (int j = j0; j < j0 + 4; j++) {
            auto jj = (j & 7);

            if (remainder) {
                // Skip this load if masked off. Otherwise, prepare next flag.
                jmpi(1 | ~f1[j & 1], done);
                if (j + 2 < 48)
                    cmp(1 | gt | f1[j & 1], null.ud(), un_rem, uint16_t(j + 2));
            }

            if (c_align16) {
                if (remainder) {
                    // Block write with masks.
                    assert(c32);
                    store(16 | f0[0] | SWSB<AllPipes>(4), block_oword(4), A64,
                            uheaders[2 * jj + 0], utemp[jj * 4 + 0]);
                    store(16 | f0[1] | SWSB<AllPipes>(4), block_oword(4), A64,
                            uheaders[2 * jj + 1], utemp[jj * 4 + 2]);
                } else {
                    // Block write.
                    store(16 | SWSB<AllPipes>(4), block_oword(2 * c_elem_bytes),
                            A64, uheaders[jj], utemp[jj * 4]);
                }

                if ((jj == 7) && (j0 + 8 < 48)) {
                    // Increment all block write pointers.
                    sync(SyncFunction::allrd);
                    for (int q = 0; q < (remainder ? 16 : 8); q += 2)
                        add<uint64_t>(8, uheaders[q], uheaders[q], uldc_x8);
                }
            } else {
                // Scattered dword or byte store, possibly masked.
                auto j1 = (j & 1);
                auto mod0 = 16 | SWSB<AllPipes>(7);
                auto mod1 = 16 | SWSB<AllPipes>(7);
                if (remainder) {
                    mod0 = mod0 | f0[0];
                    mod1 = mod1 | f0[1];
                }
                if (c32) {
                    store(mod0, scattered_dword(1), A64, uheaders[8 * j1 + 0],
                            utemp[jj * 4 + 0]);
                    store(mod1, scattered_dword(1), A64, uheaders[8 * j1 + 4],
                            utemp[jj * 4 + 2]);
                } else {
                    store(mod0, scattered_byte(c_elem_bytes), A64,
                            uheaders[8 * j1 + 0], utemp[jj * 4 + 0]);
                    store(mod1, scattered_byte(c_elem_bytes), A64,
                            uheaders[8 * j1 + 4], utemp[jj * 4 + 2]);
                }
                if ((j1 == 1) && (j + 2 < 48)) {
                    // Increment all scattered pointers at once.
                    sync(SyncFunction::allrd);
                    for (int q = 0; q < 16; q += 2)
                        add<uint64_t>(8, uheaders[q], uheaders[q], uldc_x2);
                }
            }
        }
    }

    mark(done);
}

void ngen_gen12hp_systolic_gemm_kernel_t::update_c(bool remainder) {
    // C is arranged in 8x8 column major blocks organized in a row major 4x6 array, for a total size of 32x48.
    // Each 8x8 block is split in two 8x4 blocks (due to dpasw).
    // Rearrange into contiguous columns and use hword x4 stores, taking 4 columns at a time.
    //   (effectively a 4x4 whole-register transpose)
    // This burns through icache, consider rewriting with indirect accesses.

    bool c32 = (getBytes(cfg.c_type) == 4);

    if (remainder) {
        // Set up m mask in f0.0:ud. Notice t1 is QW to support
        //  shift = 32 (all-zeros mask) case.
        auto t0 = r25.ud(0);
        auto t1 = r27.uq(0);

        add(1 | sat | SWSB(2), t0, -um_rem, uint16_t(32));
        mov(1, t1, uint32_t(0xFFFFFFFF));
        shr(1 | SWSB(1), t1, t1, t0);
        mov(1 | SWSB<int64_t>(1), f0.ud(0), t1.ud());
    }

    // Set up headers and multiples of LDC (= ldc in bytes). TODO collapse into one instruction.
    shl(1 | SWSB(3), uldc_x2, uldc, uint16_t(1));
    shl(1, uldc_x4, uldc, uint16_t(2));
    shl(1, uldc_x8, uldc, uint16_t(3));

    // Check whether C pointer has given (power of 2) alignment. Result stored in f1.1.
    auto check_c_align = [&](int align,
                                 InstructionModifier swsb
                                 = InstructionModifier()) {
        // This should work, but doesn't (Fulsim bug?):
        // auto dummy = utemp[0].ud(0);
        // bfn(1 | swsb | ze | f1[1], getBFNCtrl([](uint8_t a, uint8_t b, uint8_t c) { return (a | b) & c; }),
        //    dummy, uldc, uc_base.ud(0), uint16_t(align - 1));
        auto uc_align = r18.ud(0);
        or_(1, uc_align, uldc, c_ptr_mem.ud(0));
        and_(1 | SWSB(1) | ze | f1[1], null.ud(), uc_align,
                uint16_t(align - 1));
    };

    Label unaligned_c;

    if (!cfg.c_align16_check) {
        // Assume 16-byte alignment.
        load_c(remainder, true);
        store_c(remainder, true);
    } else if (!c32 && remainder) {
        // C not 32-bit, remainder. Only one (unaligned) path.
        load_c(remainder, false);
        store_c(remainder, false);
    } else {
        // Two full paths, one with aligned C, one without.
        check_c_align(16);
        jmpi(1 | ~f1[1], unaligned_c);
        load_c(remainder, true);
        store_c(remainder, true);
        epilogue();
        mark(unaligned_c);
        load_c(remainder, false);
        store_c(remainder, false);
    }
}

// Update C, checking at runtime for remainder handling.
void ngen_gen12hp_systolic_gemm_kernel_t::update_c() {
    Label partial_c;

    // Move C pointer to safety.
    mov(2, uc_base, c_ptr_mem);

    // Pull saved data from accumulators. Note moves to/from accumulator don't support full swizzling, so
    //  the Subregisters for the following movs must match.
    assert(ldc_save.getByteOffset() == uldc.getByteOffset());
    mov(1, uldc, ldc_save);

    assert(mrem_save.getByteOffset() == um_rem.getByteOffset());
    mov(4, um_rem.ud()(1), mrem_save.ud()(1));
    mov(2, ualpha_regs[1].ud()(1), alpha_save.ud()(1));

    // Do remainder check.
    if (!cfg.c_remainder)
        update_c(false);
    else {
        jmpi(1 | f0[0] | anyv, partial_c);
        update_c(false);
        epilogue();

        mark(partial_c);
        update_c(true);
    }
}

// Scoreboard usage:
//   $0-2   B SLM loads
//   $3-4   A SLM loads
//   $5     Last DPASW in chain
//   $6-7   Load local IDs/kernel arguments
//   $8     A copy to SLM
//   $9-10  B copy to SLM
//   $11    Initial A copy to SLM using C register space
//   $12-13 Initial B copy to SLM using C register space
//   $14    EOT
//   $15    Barriers/SLM fences

void ngen_gen12hp_systolic_gemm_kernel_t::dpasw_typed(
        const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount,
        const GRF &c_reg, const GRF &a_reg, const GRF &b_reg) {
    dpasw(mod, sdepth, rcount, c_reg.retype(cfg.acc_type),
            c_reg.retype(cfg.acc_type), a_reg.retype(cfg.a_type),
            b_reg.retype(cfg.b_type));
}

void ngen_gen12hp_systolic_gemm_kernel_t::multiply_chunk(int ao, int i0,
        bool waitb, const InstructionModifier &swsb0,
        const InstructionModifier &swsb_end) {
    int co = i0 * 6;

    if (waitb) {
        dpasw_typed(
                8 | swsb0 | Atomic, 8, 8, c_regs[co], a_regs[ao], b_regs[0]);
        dpasw_typed(8, 8, 8, c_regs[co + 8], a_regs[ao], b_regs[4]);
        dpasw_typed(8 | sb1.dst | Atomic, 8, 8, c_regs[co + 16], a_regs[ao],
                b_regs[8]);
        dpasw_typed(8, 8, 8, c_regs[co + 24], a_regs[ao], b_regs[12]);
        dpasw_typed(8 | sb2.dst | Atomic, 8, 8, c_regs[co + 32], a_regs[ao],
                b_regs[16]);
        dpasw_typed(
                8 | swsb_end, 8, 8, c_regs[co + 40], a_regs[ao], b_regs[20]);
    } else {
        dpasw_typed(
                8 | swsb0 | Atomic, 8, 8, c_regs[co], a_regs[ao], b_regs[0]);
        dpasw_typed(8 | Atomic, 8, 8, c_regs[co + 8], a_regs[ao], b_regs[4]);
        dpasw_typed(8 | Atomic, 8, 8, c_regs[co + 16], a_regs[ao], b_regs[8]);
        dpasw_typed(8 | Atomic, 8, 8, c_regs[co + 24], a_regs[ao], b_regs[12]);
        dpasw_typed(8 | Atomic, 8, 8, c_regs[co + 32], a_regs[ao], b_regs[16]);
        dpasw_typed(
                8 | swsb_end, 8, 8, c_regs[co + 40], a_regs[ao], b_regs[20]);
    }
}

void ngen_gen12hp_systolic_gemm_kernel_t::multiply(
        int buffer, bool last_multiply) {
    // Load half of A (16x32) -- hopefully broadcast from SLM to this row -- and half of B (32x24).
    InstructionModifier swsb = last_multiply ? SWSB(1) : dep_addr0;

    mov(1 | swsb, addr0.ud(2), slm_a_offset_load);
    mov(1 | dep_addr1, addr1.ud(2), slm_b_offset_load);
    add(1 | dep_addr2, addr2.ud(2), slm_b_offset_load, uint16_t(8 * 32 / 16));
    add(1 | dep_addr3, addr3.ud(2), slm_b_offset_load, uint16_t(16 * 32 / 16));

    if (cfg.alt_barriers) barrierwait();

    if (cfg.fulsim) sync(SyncFunction::nop, SWSB<int64_t>(1));
    sync(SyncFunction::nop, SWSB(sb5.src));
    load(16 | SWSB(sb3, 4), a_regs[0], block_oword(16), SLM, addr0);
    load(16 | SWSB(sb0, 3), b_regs[0], block_oword(16), SLM, addr1);
    load(16 | SWSB(sb1, 2), b_regs[8], block_oword(16), SLM, addr2);
    load(16 | SWSB(sb2, 1), b_regs[16], block_oword(16), SLM, addr3);

    add(1 | sb3.src, addr0.ud(2), slm_a_offset_load, uint16_t(8 * 32 / 16));
    add(1 | sb0.src, addr1.ud(2), slm_a_offset_load, uint16_t(16 * 32 / 16));
    add(1 | sb1.src, addr2.ud(2), slm_a_offset_load, uint16_t(24 * 32 / 16));
    load(16 | SWSB(sb4, 3), a_regs[8], block_oword(16), SLM, addr0);

    // Wait for first A register to load.
    sync(SyncFunction::nop, sb3.dst);

    if (cfg.alt_barriers && !last_multiply) {
        and_<uint32_t>(1 | sb2.src, addr3[2], r0_save[2], uint32_t(0x7F000000));
        barriermsg(SWSB(sb15, 1), addr3);
    }

    // Rows 0-7
    multiply_chunk(0, 0, true, sb0.dst, sb3);

    // Load third quarter of A (8x32)
    load(16 | SWSB(sb3, 2), a_regs[0], block_oword(16), SLM, addr1);

    // Rows 8-15
    multiply_chunk(8, 8, false, sb4.dst, sb4);

    // Load last quarter of A (8x32)
    load(16 | SWSB(sb4, 1), a_regs[8], block_oword(16), SLM, addr2);

    // Increment A and B to next buffer.
    swsb = cfg.fulsim ? InstructionModifier(sb3.src) : InstructionModifier();
    if (buffer == 2)
        mov(2 | swsb, slm_a_offset_load(1), slm_a_offset_load_init(1));
    else
        add(2 | swsb, slm_a_offset_load(1), slm_a_offset_load(1),
                uint16_t(slm_buf_size() / 16));

    // Rows 16-23
    multiply_chunk(0, 16, false, sb3.dst);

    // Rows 24-32
    multiply_chunk(8, 24, false, sb4.dst, sb5);

    // Remember dependencies for address registers.
    dep_addr0 = InstructionModifier {};
    dep_addr1 = sb3.src;
    dep_addr2 = sb4.src;
    dep_addr3 = cfg.alt_barriers ? sb15.src : sb2.src;
}

void ngen_gen12hp_systolic_gemm_kernel_t::copy_load(
        int load_buffer, bool use_c) {
    // Load new A and B and increment load pointers
    sync(SyncFunction::nop,
            SWSB<uint64_t>(1)); // SWSB doesn't cover $.src + RegDist
    mov(1 | dep_addr0, addr0.uq(0), a_ptr_mem);
    mov(1 | dep_addr1, addr1.uq(0), b_ptr_mem);
    add(1 | dep_addr2, addr2.uq(0), b_ptr_mem, uint16_t(8 * 32));

    if (use_c) {
        load(16 | SWSB(sb11, 3), c_regs[0], block_hword(8), A64, addr0);
        load(16 | SWSB(sb12, 2), c_regs[8], block_hword(8), A64,
                addr1); // Fulsim 39092: @3
        load(16 | SWSB(sb13, 1), c_regs[16], block_hword(4), A64,
                addr2); // ditto
        dep_addr0 = sb11.src;
        dep_addr1 = sb12.src;
        dep_addr2 = sb13.src;
    } else {
        load(16 | SWSB(sb8, 3), a_copy[0], block_hword(8), A64,
                addr0); // Stronger than necessary dependencies... can load as soon as prev. store inputs are read.
        load(16 | SWSB(sb9, 2), b_copy[0], block_hword(8), A64, addr1);
        load(16 | SWSB(sb10, 1), b_copy[8], block_hword(4), A64, addr2);
        dep_addr0 = sb8.src;
        dep_addr1 = sb9.src;
        dep_addr2 = sb10.src;
    }

    if (cfg.fulsim)
        sync(SyncFunction::allrd,
                use_c ? 0x3000 : 0x600); // Unnecessary syncs to pacify Fulsim.

    add(1 | SWSB(3), a_ptr_mem, a_ptr_mem, uint16_t(32 * 32));
    add(1 | SWSB(3), b_ptr_mem, b_ptr_mem, uint16_t(48 * 32));
}

void ngen_gen12hp_systolic_gemm_kernel_t::copy_store(
        int store_buffer, bool first) {
    auto aoffset = first ? slm_a_offset_store_init : slm_a_offset_store;
    auto boffset = first ? slm_b_offset_store_init : slm_b_offset_store;

    // Store A and B and advance store pointers to next buffer.
    mov(1 | dep_addr0, addr0.ud(2), aoffset);
    mov(1 | dep_addr1, addr1.ud(2), boffset);
    add(1 | dep_addr2, addr2.ud(2), boffset, uint16_t(8 * 32 / 16));

    if (first) {
        store(16 | SWSB(sb11, 3), block_oword(16), SLM, addr0, c_regs[0]);
        store(16 | SWSB(sb12, 2), block_oword(16), SLM, addr1,
                c_regs[8]); // Fulsim 39092: @3
        store(16 | SWSB(sb13, 1), block_oword(8), SLM, addr2, c_regs[16]);
        dep_addr0 = sb11.src;
        dep_addr1 = sb12.src;
        dep_addr2 = sb13.src;
    } else {
        store(16 | SWSB(sb8, 3), block_oword(16), SLM, addr0, a_copy[0]);
        store(16 | SWSB(sb9, 2), block_oword(16), SLM, addr1, b_copy[0]);
        store(16 | SWSB(sb10, 1), block_oword(8), SLM, addr2, b_copy[8]);
        dep_addr0 = sb8.src;
        dep_addr1 = sb9.src;
        dep_addr2 = sb10.src;
    }

    if (cfg.fulsim) sync(SyncFunction::allrd, first ? 0x3000 : 0x600);

    if (store_buffer == 2)
        mov(2, slm_a_offset_store(1), slm_a_offset_store_init(1));
    else
        add(2, slm_a_offset_store(1), aoffset(1),
                uint16_t(slm_buf_size() / 16));
}

void ngen_gen12hp_systolic_gemm_kernel_t::store_signal() {
    if (cfg.use_slm_fence) {
        // Signal SLM data ready once memory fence returns, asynchronously
        sync(SyncFunction::nop, dep_addr0);
        and_<uint32_t>(
                1 | dep_addr3, addr3[2], r0_save[2], uint32_t(0x7F000000));

        slmfence(SWSB(sb15, 1), addr0);
        barriermsg(SWSB(sb15), addr3);
        dep_addr0 = dep_addr3 = sb15.src;
    } else {
        and_<uint32_t>(
                1 | dep_addr3, addr3[2], r0_save[2], uint32_t(0x7F000000));
        barriermsg(SWSB(sb15, 1), addr3);
        dep_addr3 = sb15.src;
    }
}

void ngen_gen12hp_systolic_gemm_kernel_t::body() {
    Label top, bottom;

    add(1 | le | f0[1], k_counter, k_counter, int16_t(-1));

    copy_load(0, true); // L0 -> C
    copy_load(1); // L1
    copy_store(0, true); // S0 <- C
    store_signal(); // Signal 0 ready
    zero_c();
    sync(SyncFunction::nop, SWSB<uint32_t>(1));
    copy_store(1); // S1
    if (!cfg.alt_barriers) barrierwait(); // Wait 0 ready
    store_signal(); // Signal 1 ready

    jmpi(1 | f0[1], bottom); // Zero-trip loop check

    mark(top);
    add(1 | gt | f0[1], k_counter, k_counter, int16_t(-1));

    copy_load(2); // L2
    multiply(0); // M0
    copy_store(2); // S2

    if (!cfg.alt_barriers) {
        barrierwait(); // Wait 1 ready
        store_signal(); // Signal 2 ready
    }

    copy_load(0); // L0
    multiply(1); // M1
    copy_store(0); // S0

    if (!cfg.alt_barriers) {
        barrierwait(); // Wait 2 ready
        store_signal(); // Signal 0 ready
    }

    copy_load(1); // L1
    multiply(2); // M2
    copy_store(1); // S1

    if (!cfg.alt_barriers) {
        barrierwait(); // Wait 0 ready
        store_signal(); // Signal 1 ready
    }

    jmpi(1 | f0[1], top);
    mark(bottom);

    copy_load(2); // L2
    multiply(0); // M0
    copy_store(2); // S2

    if (!cfg.alt_barriers) {
        barrierwait(); // Wait 1 ready
        store_signal(); // Signal 2 ready
    }

    multiply(1); // M1

    if (!cfg.alt_barriers) barrierwait(); // Wait 2 ready

    multiply(2, true); // M2

    sync(SyncFunction::allwr); // Wait for systolic ops to finish

    update_c();
}

void ngen_gen12hp_systolic_gemm_kernel_t::epilogue() {
    // Global memory fence and end of thread.
    memfence(SWSB(sb14), r16);
    mov<uint32_t>(8, r255, r0_save);
    threadend(SWSB(sb14, 1), r255);
}

ngen_gen12hp_systolic_gemm_kernel_t::ngen_gen12hp_systolic_gemm_kernel_t(
        config_t cfg_)
    : cfg(cfg_) {
    if (!cfg.valid()) assert(!"Invalid configuration");

    setDefaultNoMask();

    // Signature:
    //   kernel void gemm_kernel(global char *ap, global uchar *bp, global int *C,
    //                           int k, int ldc,
    //                           long offset_a, long offset_b, long offset_c,
    //                           int m, int n,
    //                           float alpha, float beta,
    //                           int lda, int ldb)

    interface.externalName("gen12hp_systolic_gemm_kernel");
    interface.newArgument("ap", ExternalArgumentType::GlobalPtr);
    interface.newArgument("bp", ExternalArgumentType::GlobalPtr);
    interface.newArgument("c", ExternalArgumentType::GlobalPtr);
    interface.newArgument("k", DataType::d);
    interface.newArgument("ldc", DataType::d);
    interface.newArgument("offset_a", DataType::q);
    interface.newArgument("offset_b", DataType::q);
    interface.newArgument("offset_c", DataType::q);
    interface.newArgument("m", DataType::d);
    interface.newArgument("n", DataType::d);
    interface.newArgument("alpha", DataType::f);
    interface.newArgument("beta", DataType::f);
    interface.newArgument("lda", DataType::d);
    interface.newArgument("ldb", DataType::d);
    interface.requireBarrier();
    interface.requireDPAS();
    interface.requireLocalID(2);
    interface.requireSIMD(8);
    interface.requireSLM(slm_buf_size() * 3);
    interface.finalize();

    // Inputs.
    auto global_id_x = r0.ud(1);
    auto global_id_y = r0.ud(6);
    auto local_id_x = r1.uw(0);
    auto local_id_y = r2.uw(0);
    auto ap = interface.getArgument("ap");
    auto bp = interface.getArgument("bp");
    auto c_ptr = interface.getArgument("c");
    auto in_offset_a = interface.getArgument("offset_a");
    auto in_offset_b = interface.getArgument("offset_b");
    auto in_offset_c = interface.getArgument("offset_c");
    auto k = interface.getArgument("k");
    auto ldc = interface.getArgument("ldc");
    auto m = interface.getArgument("m");
    auto n = interface.getArgument("n");
    auto alpha = interface.getArgument("alpha");
    auto beta = interface.getArgument("beta");
    auto lda = interface.getArgument("lda");
    auto ldb = interface.getArgument("ldb");

    // Temporaries
    auto n0 = r10.ud(0);
    auto m0 = r10.ud(1);
    auto offset_a = r12.uq(0);
    auto offset_b = r12.uq(1);
    auto offset_c = r12.uq(2);
    auto global_n0 = r18.ud(0);
    auto global_m0 = r18.ud(1);
    auto local_n0 = r20.ud(0);
    auto local_m0 = r20.ud(1);
    auto suboffset_a = r26.ud(0);
    auto suboffset_b = r26.ud(1);
    auto thd1_adjust = r27.ud(0);
    auto temp = r28.ud(0);
    auto k_copy = r29.ud(0);
    auto save_copy = r30.ud();
    auto ldc_copy = r32.ud(1);

    // Prologue w/ local ID load.
    GRF header = r61;

    mov<uint32_t>(8, header, uint32_t(0));
    and_<uint32_t>(1, header[2], r0[0], uint32_t(0xFFFFFFE0));
    and_<uint16_t>(1, header[0], r0[4], uint16_t(0xFF));
    add<uint32_t>(1 | SWSB(2), header[2], header[2], uint16_t(0x80));
    mad<uint32_t>(
            1 | SWSB(1), header[2], header[2], header.uw(0), uint16_t(0x60));
    load(16 | SWSB(sb7, 1), r1, aligned_block_oword(4), A32NC,
            header); // Read local IDs.
    sync(SyncFunction::nop, sb7.dst);
    sync(SyncFunction::nop);

    // Prologue w/o local ID load for HW-generated local IDs (offset 0x80).
    mov<uint32_t>(8, header, uint32_t(0));
    and_<uint32_t>(1, header[2], r0[0], uint32_t(0xFFFFFFE0));
    mov<uint32_t>(8, r0_save, r0);

    load(16 | SWSB(sb6, 2), r4, aligned_block_oword(8), A32NC, header);

    // Find our threadgroup's position within the matrix.
    shl(1, global_m0, global_id_x, uint16_t(7));
    mul(1, global_n0, global_id_y, uint16_t(192));

    // Find our position within the threadgroup. Fixed threadgroup size: 4x4.
    shl(1, local_m0, local_id_x, uint16_t(2));
    mul(1, local_n0, local_id_y, uint16_t(48));
    add(2 | SWSB(1), n0(1), local_n0(1), global_n0(1));

    // Compute starting addresses:
    //   - suboffset_a = local_id_Y * 8 * 32
    //   - suboffset_b = local_id_X/8 * 12 * 32
    //   - slm_a_offset_load_init = local_m0 * 32 [36 with padding]
    //   - slm_b_offset_load_init = 128 * 32 [36 w/ padding] + local_n0 * 32 + (24 * 32 if fused)
    //   - slm_a_offset_store_init = slm_a_offset_load_init + suboffset_a
    //   - slm_b_offset_store_init = slm_b_offset_load_init + suboffset_b
    //   - Ap += m0 * lda + suboffset_a
    //   - Bp += n0 * ldb + suboffset_b
    //   - C += m0 + n0 * ldc [save for later]
    uint16_t lg2_a_elem_bytes = ngen::utils::log2(getBytes(cfg.a_type));
    uint16_t lg2_c_elem_bytes = ngen::utils::log2(getBytes(cfg.c_type));

    sync(SyncFunction::nop, sb6.dst);
    shl(1, ldc_copy, ldc, lg2_c_elem_bytes);
    mov(1, k_copy, k);
    shl(1, suboffset_a, local_id_y, uint16_t(8));
    mul(1, suboffset_b, local_id_x, uint16_t(12 * 32 / 8));
    assert(ldc_save.getByteOffset() == ldc_copy.getByteOffset());
    mov(1 | SWSB(3), ldc_save.ud(), ldc_copy.ud());
    mul(1 | SWSB<int>(5), offset_c, n0, ldc);
    mul(1, offset_a, m0, lda);
    mul(1, offset_b, n0, ldb);
    add(1 | SWSB(3), offset_c, offset_c, m0);
    add(1 | SWSB(3), offset_a, offset_a, in_offset_a); // TODO: combine
    add(1 | SWSB(3), offset_b, offset_b, in_offset_b);
    add(1 | SWSB(3), offset_c, offset_c, in_offset_c);
    if (getBytes(cfg.a_type) > 1)
        shl(2 | SWSB(2), offset_a(1), offset_a(1), lg2_a_elem_bytes); // A, B
    // add(2 | SWSB<AllPipes>(1), offset_a(1), offset_a(1), suboffset_a(1));    // unclear if allowed in HW
    add(1 | SWSB<AllPipes>(1), offset_a, offset_a, suboffset_a);
    add(1 | SWSB<AllPipes>(2), offset_b, offset_b, suboffset_b);
    shl(1 | SWSB(2), offset_c, offset_c, lg2_c_elem_bytes);
    add(1 | SWSB(2), a_ptr_mem, ap, offset_a);
    add(1 | SWSB(3), b_ptr_mem, bp, offset_b);
    add(1 | SWSB(3), c_ptr_mem, c_ptr, offset_c);

    and_(1, temp, local_id_x, uint16_t(8));
    shr(2 | SWSB(7), suboffset_a(1), suboffset_a(1), uint16_t(4));

    if (cfg.pad_a) {
        shl(1, local_n0, local_n0, uint16_t(5 - 4));
        mul(1, local_m0, local_id_x, uint16_t(9));
        mul(1 | SWSB(2), thd1_adjust, temp, uint16_t(24 * 32 / (8 * 16)));
        add(1 | SWSB(3), local_n0, local_n0, uint16_t((128 * 36) / 16));
    } else {
        shl(2, local_n0(1), local_n0(1), uint32_t(5 - 4));
        mul(1 | SWSB(2), thd1_adjust, temp, uint16_t(24 * 32 / (8 * 16)));
        add(1 | SWSB(2), local_n0, local_n0, uint16_t(128 * 32 / 16));
    }

    mov(1 | SWSB(2), slm_a_offset_load_init.uw(), local_m0.uw());
    add(1 | SWSB(2), slm_b_offset_load_init.uw(), local_n0.uw(),
            thd1_adjust.uw());
    assert(k_counter.getByteOffset() == k_copy.getByteOffset());
    mov(1, k_counter, k_copy);
    add(1 | SWSB(6), slm_a_offset_store_init.uw(), local_m0.uw(),
            suboffset_a.uw());
    add(1 | SWSB(5), slm_b_offset_store_init.uw(), local_n0.uw(),
            suboffset_b.uw());
    mov(2 | SWSB(4), slm_a_offset_load(1), slm_a_offset_load_init(1));

    // Compute m, n remainders and save alpha/beta for C update.
    // Also compute threshold for m remainder: 64 for thread 0 of fused pair,
    //  32 for thread 1.
    if (cfg.c_remainder) {
        shl(1, temp, temp, uint16_t(2));
        add(1 | sat, m.ud(), m, -m0);
        add(1 | sat, n.ud(), n, -n0);
        add(1 | SWSB(3), temp, -temp, uint16_t(64));
    }

    sync(SyncFunction::nop, SWSB<AllPipes>(1));

    if (mrem_save.getByteOffset() == m.getByteOffset())
        mov(4, mrem_save.ud()(1), m.ud()(1)); // m/n/alpha/beta
    else {
        // Shuffle into correct lanes before moving to accumulator.
        mov(1, save_copy.f(4), m.f());
        mov(1, save_copy.f(5), n.f());
        mov(1, save_copy.f(6), alpha.f());
        mov(1, save_copy.f(7), beta.f());
        mov(4 | SWSB<float>(1), mrem_save.ud()(1), save_copy.ud(4)(1));
    }

    // Check whether to use remainder path, and save in f0.0/f1.0 for later.
    if (cfg.c_remainder) {
        cmp(1 | lt | f0[0], null.ud(), m, temp);
        cmp(1 | lt | f1[0], null.ud(), n, uint32_t(48));
    }

    // Main body.
    body();

    // Epilogue.
    epilogue();

    // Kernel padding for instruction prefetch.
    for (int rep = 0; rep < 8; rep++)
        nop();
}

} // namespace ocl
} // namespace impl
} // namespace dnnl

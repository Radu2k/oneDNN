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

#ifndef GPU_JIT_GEN12HP_SYSTOLIC_GEMM_KERNEL_HPP
#define GPU_JIT_GEN12HP_SYSTOLIC_GEMM_KERNEL_HPP

#include <cstdint>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gpu/jit/ngen/ngen_opencl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class gen12hp_systolic_gemm_kernel_t
    : public ngen::OpenCLCodeGenerator<ngen::HW::Gen12HP> {
public:
    enum class bias_t { none, fixed, row, column };

    struct config_t {
        ngen::DataType a_type, b_type, c_type, acc_type;
        bool alpha1, beta0, beta1;

        bool a_bias = false;
        bool b_bias = false;
        bias_t c_bias = bias_t::none;

        bool alt_barriers = false;
        bool use_slm_fence = true;
        bool c_remainder = true;
        bool c_align16_check = true;
        bool pad_a = true;
        bool fulsim = true;

        bool valid() {
            using ngen::DataType;

            bool ok = true;
            if (c_type == DataType::d || c_type == DataType::ud) {
                ok &= (a_type == DataType::b || a_type == DataType::ub);
                ok &= (b_type == DataType::b || b_type == DataType::ub);
                ok &= (acc_type == c_type);
            } else {
                ok &= (a_type == b_type);
                ok &= (a_type == DataType::bf || a_type == DataType::hf);
                ok &= (c_type == DataType::f || c_type == a_type);
                ok &= (acc_type == DataType::f);
                ok &= !a_bias && !b_bias && (c_bias == bias_t::none);
            }
            ok &= (alt_barriers || use_slm_fence);
            return ok;
        }
    };

    static constexpr size_t unroll_m = 32;
    static constexpr size_t unroll_n = 48;
    static constexpr size_t unroll_k_bytes = 96;
    static constexpr size_t thread_group_m = 4;
    static constexpr size_t thread_group_n = 4;
    static constexpr size_t nominal_subgroup_size = 8;

    static size_t unroll_k(data_type_t dt) {
        return unroll_k_bytes / types::data_type_size(dt);
    }

    static int64_t get_ld_packed(int64_t block_k, data_type_t dt, bool sum) {
        auto ld = block_k;
        ld = utils::rnd_up(ld, unroll_k(dt));
        if (sum) ld += 32 / types::data_type_size(dt);
        if (!(ld & 0x7FF)) ld += 32;
        return ld;
    }

    static int64_t max_ld_packed(int64_t block_k, data_type_t dt, bool sum) {
        auto ld = block_k;
        ld = utils::rnd_up(ld, unroll_k(dt));
        if (sum) ld += 32 / types::data_type_size(dt);
        ld += 32;
        return ld;
    }

private:
    config_t cfg;

    // Surface assignments
    int ap_surface, bp_surface, co_surface;

    // Register assignments (main loop)
    ngen::GRFRange a_copy = r40 - r47;
    ngen::GRFRange b_copy = r2 - r13;
    ngen::GRFRange a_regs = r48 - r63;
    ngen::GRFRange b_regs = r14 - r37;
    ngen::GRFRange c_regs = r64 - r255;
    ngen::GRF addr0 = r1;
    ngen::GRF addr1 = r38;
    ngen::GRF addr2 = r39;
    ngen::GRF addr3 = r0;
    ngen::Subregister a_ptr_mem = addr1.uq(3);
    ngen::Subregister b_ptr_mem = addr2.uq(3);
    ngen::Subregister c_ptr_mem = addr2.uq(2);
    ngen::Subregister slm_a_offset_load = addr1.uw(8); // offsets in OWords
    ngen::Subregister slm_b_offset_load = addr1.uw(9);
    ngen::Subregister slm_a_offset_store = addr1.uw(10);
    ngen::Subregister slm_b_offset_store = addr1.uw(11);
    ngen::Subregister slm_a_offset_load_init = addr1.uw(6);
    ngen::Subregister slm_b_offset_load_init = addr1.uw(7);
    ngen::Subregister slm_a_offset_store_init = addr2.uw(6);
    ngen::Subregister slm_b_offset_store_init = addr2.uw(7);
    ngen::Register base_save = acc0.ud();
    ngen::Subregister k_counter = acc0.ud(0);
    ngen::Subregister ldc_save = acc0.ud(1);
    ngen::Subregister off_co_save = acc0.ud(2);
    ngen::Subregister k_save = acc0.ud(3);
    ngen::Subregister mrem_save = acc0.uw(8);
    ngen::Subregister nrem_save = acc0.uw(9);
    ngen::Subregister abo_save = acc0.ud(5);
    ngen::Subregister ao_save = acc0.w(10);
    ngen::Subregister bo_save = acc0.w(11);
    ngen::Subregister alpha_save = acc0.ud(6);
    ngen::Subregister beta_save = acc0.ud(7);
    ngen::AccumulatorRegister r0_save = acc2;
    ngen::Subregister off_asum_save = a0.ud(0);
    ngen::Subregister off_bsum_save = a0.ud(1);

    ngen::InstructionModifier dep_addr0 {}, dep_addr1 {}, dep_addr2 {},
            dep_addr3 {}; // Dependencies for addr registers.

    // Register assignments (C update)
    ngen::GRFRange utemp = r32 - r63;
    ngen::GRFRange uheaders = r0 - r15;
    ngen::GRFRange uoffset = r22 - r27;

    ngen::Subregister uldc_x2 = r18.ud(1);
    ngen::Subregister uldc_x4 = r18.ud(2);
    ngen::Subregister uldc_x8 = r18.ud(3);

    ngen::Subregister uc_base = r17.uq(0);
    ngen::Subregister uoff_co2 = r17.ud(2);
    ngen::Subregister uao_bo_k = r19.ud(0);
    ngen::GRF ubase = r28.ud();
    ngen::Subregister uldc = r28.ud(1);
    ngen::Subregister uoff_co = r28.ud(2);
    ngen::Subregister uk = r28.ud(3);
    ngen::Subregister um_rem = r28.uw(8);
    ngen::Subregister un_rem = r28.uw(9);
    ngen::Subregister uao = r28.w(10);
    ngen::Subregister ubo = r28.w(11);
    ngen::Subregister ualpha_regs[2] = {r28.f(6), r30.f(6)};
    ngen::Subregister ubeta_regs[2] = {r28.f(7), r30.f(7)};

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

    static constexpr int acc_stride = 48;

    int slm_buf_size() const {
        return cfg.pad_a
                ? 10752 // 4.5k A (128x32 + 4*128 padding) + 6k B (192x32)
                : 10240; // 4k A (128x32) + 6k B (192x32)
    }

    void zero_c();

    void scattered_setup_c(int stride, bool load);
    void block_setup_c(bool remainder, bool load);

    int interleave(int j);

    void load_c_bias();
    void add_c_bias();
    bool merge_abc_bias();
    void add_ab_bias();

    void load_c(bool remainder, bool c_align16);
    void store_c(bool remainder, bool c_align16);
    void update_c(bool remainder);
    void update_c();

    void dpasw_typed(const ngen::InstructionModifier &mod, uint8_t sdepth,
            uint8_t rcount, const ngen::GRF &c_reg, const ngen::GRF &a_reg,
            const ngen::GRF &b_reg);

    void multiply_chunk(int ao, int i0, bool waitb,
            const ngen::InstructionModifier &swsb0
            = ngen::InstructionModifier(),
            const ngen::InstructionModifier &swsb_end
            = ngen::InstructionModifier());
    void multiply(int buffer, bool last_multiply = false);

    void copy_load(int load_buffer, bool use_c = false);
    void copy_store(int store_buffer, bool first = false);
    void store_signal();

    void body();

    void epilogue();

public:
    gen12hp_systolic_gemm_kernel_t(config_t cfg_);
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_JIT_GEN12HP_SYSTOLIC_GEMM_KERNEL_HPP

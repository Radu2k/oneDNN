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

#ifndef NGEN_GEN12HP_SYSTOLIC_GEMM_KERNEL_HPP
#define NGEN_GEN12HP_SYSTOLIC_GEMM_KERNEL_HPP

#include <cstdint>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "ocl/ngen/ngen_opencl.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

class ngen_gen12hp_systolic_gemm_kernel_t
    : public ngen::OpenCLCodeGenerator<ngen::HW::Gen12HP> {
public:
    struct config_t {
        ngen::DataType a_type, b_type, c_type, acc_type;
        bool alpha1, beta0, beta1;

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
                ok &= (beta0 || beta1) && alpha1;
                ok &= (a_type == DataType::b || a_type == DataType::ub);
                ok &= (b_type == DataType::b || b_type == DataType::ub);
            } else {
                ok &= (a_type == b_type);
                ok &= (a_type == DataType::bf || a_type == DataType::hf);
                ok &= (c_type == DataType::f || c_type == a_type);
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

private:
    config_t cfg;

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
    ngen::Subregister slm_a_offset_load
            = addr1.uw(8); // All SLM offsets are in units of OWords.
    ngen::Subregister slm_b_offset_load = addr1.uw(9);
    ngen::Subregister slm_a_offset_store = addr1.uw(10);
    ngen::Subregister slm_b_offset_store = addr1.uw(11);
    ngen::Subregister slm_a_offset_load_init = addr1.uw(6);
    ngen::Subregister slm_b_offset_load_init = addr1.uw(7);
    ngen::Subregister slm_a_offset_store_init = addr2.uw(6);
    ngen::Subregister slm_b_offset_store_init = addr2.uw(7);
    ngen::Subregister k_counter = acc0.ud(0);
    ngen::Subregister ldc_save = acc0.ud(1);
    ngen::Subregister mrem_save = acc0.ud(4);
    ngen::Subregister nrem_save = acc0.ud(5);
    ngen::Subregister alpha_save = acc0.ud(6);
    ngen::Subregister beta_save = acc0.ud(7);
    ngen::AccumulatorRegister r0_save = acc2;

    ngen::InstructionModifier dep_addr0 {}, dep_addr1 {}, dep_addr2 {},
            dep_addr3 {}; // Dependencies for addr registers.

    // Register assignments (C update)
    ngen::GRFRange utemp = r32 - r63;
    ngen::GRFRange uheaders = r0 - r15;

    ngen::Subregister uldc = r16.ud(1);
    ngen::Subregister uldc_x2 = r18.ud(1);
    ngen::Subregister uldc_x4 = r18.ud(2);
    ngen::Subregister uldc_x8 = r18.ud(3);

    ngen::Subregister uc_base = r17.uq(0);
    ngen::Subregister um_rem = r28.ud(4);
    ngen::Subregister un_rem = r28.ud(5);
    ngen::Subregister ualpha_regs[2] = {r28.f(6), r30.f(6)};
    ngen::Subregister ubeta_regs[2] = {r28.f(7), r30.f(7)};

    constexpr int slm_buf_size() const {
        return cfg.pad_a
                ? 10752 // 4.5k A (128x32 + 4*128 padding) + 6k B (192x32)
                : 10240; // 4k A (128x32) + 6k B (192x32)
    }

    void zero_c();

    void scattered_setup_c(int stride, bool load);
    void block_setup_c(bool remainder, bool load);

    int interleave(int j);

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
    ngen_gen12hp_systolic_gemm_kernel_t(config_t cfg_);
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif /* NGEN_GEN12HP_SYSTOLIC_GEMM_KERNEL_HPP */

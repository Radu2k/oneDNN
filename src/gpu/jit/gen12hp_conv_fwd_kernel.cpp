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

#include "gpu/jit/gen12hp_conv_fwd_kernel.hpp"

#include "common/utils.hpp"
#include "gpu/jit/jit_eltwise_injector.hpp"
#include "gpu/jit/jit_generator.hpp"
#include "gpu/jit/ngen/ngen_register_allocator.hpp"
#include "gpu/jit/ngen_type_bridge.hpp"
#include "gpu/ocl/ocl_gpu_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace ngen;

// Controls for performance debugging.
const bool enable_dpasw = true;

const bool enable_gmem_read = true;
const bool enable_gmem_write = true;

const bool enable_smem_read = true;
const bool enable_smem_write = true;

const bool enable_barrier = true;

class gen12hp_conv_fwd_kernel_t : public jit_generator<HW::Gen12HP> {
public:
    gen12hp_conv_fwd_kernel_t(const conv_conf_t &conf)
        : conf(conf), attr_info(conf.attr_info), ra(HW::Gen12HP) {

        src_type = convert_dnnl_type_to_ngen(conf.src_data_type);
        wei_type = convert_dnnl_type_to_ngen(conf.weights_data_type);
        bia_type = convert_dnnl_type_to_ngen(conf.bias_data_type);
        dst_type = convert_dnnl_type_to_ngen(conf.dst_data_type);
        acc_type = utils::one_of(src_type, DataType::b, DataType::ub)
                ? DataType::d
                : DataType::f;

        src_size = (int)types::data_type_size(conf.src_data_type);
        bia_size = (int)types::data_type_size(conf.bias_data_type);
        dst_size = (int)types::data_type_size(conf.dst_data_type);

        // Destination layout:
        // - 32n16c for f16/bf16/f32
        // - 32n32c for s8/u8/s32
        dst_oc_block = (acc_type == DataType::f) ? 16 : 32;

        ic_bytes_padded = utils::rnd_up(conf.ic * src_size, 32);
        oc_padded = utils::rnd_up(conf.oc, 32);
        oc_tg_padded = utils::rnd_up(conf.oc, conf.oc_group * 32);
        ow_padded = utils::rnd_up(conf.ow, conf.ow_group);

        if (ic_bytes_padded * conf.kd * conf.kh * conf.kw >= 128
                && conf.oc_group == 4) {
            slm_nbuf = 3;
        } else {
            slm_nbuf = 2;
        }

        auto has_padding = [](int o, int i, int k, int p, int s, int d) {
            return (p > 0) || (o - 1) * s - p + (k - 1) * (1 + d) >= i;
        };

        has_pad_d = has_padding(conf.od, conf.id, conf.kd, conf.f_pad,
                conf.stride_d, conf.dilate_d);
        has_pad_h = has_padding(conf.oh, conf.ih, conf.kh, conf.t_pad,
                conf.stride_h, conf.dilate_h);
        has_pad_w = has_padding(conf.ow, conf.iw, conf.kw, conf.l_pad,
                conf.stride_w, conf.dilate_w);

        has_h = (conf.ih > 1 || conf.oh > 1 || conf.kh > 1);
        has_d = (conf.id > 1 || conf.od > 1 || conf.kd > 1);

        // Kernel interface.
        newArgument("src", ExternalArgumentType::GlobalPtr);
        newArgument("wei", ExternalArgumentType::GlobalPtr);
        newArgument("bia", ExternalArgumentType::GlobalPtr);
        newArgument("dst", ExternalArgumentType::GlobalPtr);
        newArgument("oscales", ExternalArgumentType::GlobalPtr);
        newArgument("common_oscales", DataType::f);

        externalName("gen12hp_conv_fwd");
        requireLocalID(3);
        requireLocalSize();
        requireGRF(256);
        requireSIMD(8);
        requireBarrier();
        requireDPAS();

        // Implement double/triple SLM buffering. Layout for int8 for 4x4
        // thread-group:
        //     src - 4w x 32n32c
        //     wei - 4o x 4o8i8o4i
        // Pad 256 to 284 to avoid SLM bank conflicts.
        // Apply renaming: src -> A, wei -> B, dst -> C.
        a_slm_size = conf.ow_group * a_slm_block_size;
        b_slm_size = conf.oc_group * b_slm_block_size;
        ab_slm_size = a_slm_size + b_slm_size;
        requireSLM(slm_nbuf * ab_slm_size);

        finalizeInterface();

        setDefaultNoMask();
        enable_auto_swsb();
        prologue();

        // Enable IEEE f32 -> s32 rounding and f32/f16 denorms.
        or_(1, cr0, cr0, uint16_t(0x1480));

        ra.claim(r0);

        src_ptr = getArgument("src");
        ra.claim(src_ptr);

        wei_ptr = getArgument("wei");
        ra.claim(wei_ptr);

        dst_ptr = getArgument("dst");
        ra.claim(dst_ptr);

        bia_surf = getArgumentSurface("bia");
        oscales_surf = getArgumentSurface("oscales");

        if (attr_info.with_common_oscales && !attr_info.with_runtime_oscales) {
            common_oscales = getArgument("common_oscales");
            ra.claim(common_oscales);
        }

        local_id_0 = getLocalID(0);
        ra.claim(local_id_0);

        local_id_1 = getLocalID(1);
        ra.claim(local_id_1);

        // Claim A, B, C (manually allocated).
        ra.claim(A);
        ra.claim(B);
        ra.claim(C);

        allocate_registers();

        // ithr0 = get_local_id(0) / 8    [0, oc_group - 1]
        // ithr1 = get_local_id(1)        [0, ow_group - 1]
        shr<uint32_t>(1, ithr0, local_id_0.uw(0), 3);
        mov(1, ithr1, local_id_1.uw(0));

        // Initial SLM read offset for A in owords.
        // Account for DPASW offset.
        and_<uint32_t>(1, off_tmp, ithr0, 1);
        mul(1, off_tmp, off_tmp, uint16_t(16));

        mad(2, a_slm_off_rd_init[0], off_tmp, ithr1,
                uint16_t(a_slm_block_size / 16));
        for (int i = 0; i < 16; i += 8) {
            add(1, a_slm_off_rd_init[i / 8], a_slm_off_rd_init[i / 8],
                    (i / 8) * 32);
        }

        // Initial SLM read offset for B in owords.
        mov(4, b_slm_off_rd_init[0], uint16_t(a_slm_size / 16));
        mad(4, b_slm_off_rd_init[0], b_slm_off_rd_init[0], ithr0,
                uint16_t(b_slm_block_size / 16));

        for (int i = 0; i < 32; i += 8) {
            add(1, b_slm_off_rd_init[i / 8], b_slm_off_rd_init[i / 8],
                    (i / 8) * 16);
        }

        mul(1, oc_tg, group_id_0, uint16_t(conf.oc_group * 32));
        mad(1, oc, oc_tg, ithr0, uint16_t(32));
        // oc index shared between fused EUs.
        and_<uint32_t>(1, oc_fused, oc, ~(1 << 5));
        mul(1, mb, group_id_2, uint16_t(32));

        auto odh = tmp0.d(0);

        // od = get_group_id(1) / (OW_PADDED / ow_group) / OH
        // oh = get_group_id(1) / (OW_PADDED / ow_group) % OH
        // ow = get_group_id(1) % (OW_PADDED / ow_group) * ow_group +
        //      get_local_id(1)
        e_idiv(odh, ow_tg, group_id_1.d(0), ow_padded / conf.ow_group,
                tmp0.d(1));
        mul<int32_t>(1, ow_tg, ow_tg, uint16_t(conf.ow_group));
        add<int32_t>(1, ow, ow_tg, ithr1);

        if (has_d && has_h) {
            e_idiv(od, oh, odh, conf.oh, tmp0.d(1));
        } else if (has_d) {
            mov(1, od, odh);
        } else if (has_h) {
            mov(1, oh, odh);
        }

        if (has_d) {
            mul(1, id_init, od, uint16_t(conf.stride_d));
            if (conf.f_pad > 0) add(1, id_init, id_init, -conf.f_pad);
        }

        if (has_h) {
            mul(1, ih_init, oh, uint16_t(conf.stride_h));
            if (conf.t_pad > 0) add(1, ih_init, ih_init, -conf.t_pad);
        }

        mov(1, iw_tg, ow_tg);
        mul(1, iw_tg, ow_tg, uint16_t(conf.stride_w));
        if (conf.l_pad > 0) add(1, iw_tg, iw_tg, -conf.l_pad);

        // iw to start reading from global memory.
        mad(1, iw_init, iw_tg, ithr0, uint16_t(conf.stride_w));

        // Initial SLM write offsets for A and B in owords.
        mul(1, a_slm_off_wr_init, ithr0, uint16_t(a_slm_block_size / 16));
        mad(1, a_slm_off_wr_init, a_slm_off_wr_init, ithr1, uint16_t(16));
        add(1, b_slm_off_wr_init, a_slm_off_wr_init, a_slm_size / 16);

        // Set C to zero.
        for (int i = 0; i < 128; i += 2)
            mov<float>(16, C[i], 0.0f);

        // Initialize TG offsets for source and weights.
        init_src_off_tg();
        init_wei_off_tg();

        // Initialize thread read offsets for source.
        mul(1, src_off_rd, ithr0, uint16_t(conf.stride_w * 32 * 32));
        mad(1, src_off_rd, src_off_rd, ithr1, uint16_t(256));
        add(1, src_off_rd, src_off_rd, src_off_tg);

        // Initialize thread read address for weights.
        mul(1, wei_addr.uq(0), ithr0,
                ic_bytes_padded * conf.kd * conf.kh * conf.kw * 32);
        mad(1, wei_addr.d(0), wei_addr.d(0), ithr1, uint16_t(256));
        add(1, wei_addr.d(0), wei_addr.d(0), wei_off_tg);
        add(1, wei_addr.uq(0), wei_addr.uq(0), wei_ptr);

        if (slm_nbuf == 3) {
            // Triple SLM buffering.
            // SLM buffer indices (reverse order):
            // Load:     2 -> 1 -> 0 -> 2 -> ...
            // Compute:  1 -> 0 -> 2 -> 1 -> ...
            // Counter:  0 -> 1 -> 2 -> 3 -> ...
            // Reverse order allows to save one cmp call by using conditional
            // modifier.
            mov(4, slm_idx, Immediate::uv(2, 1, 0, 0, 0, 0, 0, 0));
        } else {
            // Double SLM buffering.
            // SLM buffer indices (reverse order):
            // Load:     1 -> 0 -> 1 -> ...
            // Compute:  0 -> 1 -> 0 -> 1 -> ...
            // Counter:  0 -> 1 -> 2 -> 3 -> ...
            mov(4, slm_idx, Immediate::uv(1, 0, 0, 0, 0, 0, 0, 0));
        }

        // Disable auto-SWSB for the inner loop for better control over SBID-s.
        // SBID usage:
        //   $0-1   A load from global memory and store to SLM
        //   $2     B load from global memory and store to SLM
        //   $8-11  A/B SLM loads and DPASW
        //   $15    Barrier/SLM fence
        disable_auto_swsb();

        // To complete all OOO writes.
        sync(SyncFunction::allwr);

        if (slm_nbuf == 3) fence_and_signal(/*skip_fence=*/true);

        sync(SyncFunction::nop, SWSB<AllPipes>(1));

        mov(1, ic_bytes, 0);

        // Reduction loop.
        Label ic_loop;
        mark(ic_loop);

        Label kd_loop;
        Label kd_skip;
        if (has_d) {
            mov(1, kd, 0);
            mov(1, id, id_init);
            mark(kd_loop);

            // Check padding for ID.
            if (has_pad_d) {
                cmp(1 | lt | f1[0] | SWSB(1), id, conf.id);
                cmp(1 | f1[0] | ge, id, 0);
                add(1 | ~f1[0], wei_addr.uq(0), wei_addr.uq(0),
                        conf.kh * conf.kw * 32 * 32);
                jmpi(1 | ~f1[0], kd_skip);
            }
        }

        Label kh_loop;
        Label kh_skip;
        if (has_h) {
            mov(1, kh, 0);
            mov(1, ih, ih_init);

            mark(kh_loop);

            // Check padding for IH.
            if (has_pad_h) {
                cmp(1 | lt | f1[0] | SWSB(1), ih, conf.ih);
                cmp(1 | f1[0] | ge, ih, 0);
                add(1 | ~f1[0], wei_addr.uq(0), wei_addr.uq(0),
                        conf.kw * 32 * 32);
                jmpi(1 | ~f1[0], kh_skip);
            }
        }

        mov(1, kw, 0);
        mov(1, iw, iw_init);

        Label kw_loop;
        mark(kw_loop);

        if (slm_nbuf == 2) {
            fence_and_signal();
            wait();
        }
        gmem2reg();
        multiply();
        reg2smem();

        slm_buffer_advance();

        // Advance kw = kw + 1.
        add(1, kw, kw, 1);
        add(1, iw, iw, 1 + conf.dilate_w);
        add(1, src_off_rd, src_off_rd, (1 + conf.dilate_w) * 32 * 32);
        cmp(8 | lt | f0[0] | SWSB(3), kw, conf.kw);
        while_(8 | f0[0], kw_loop);

        // Restore src offset after kw loop.
        add(1 | SWSB(3), src_off_rd, src_off_rd,
                -conf.kw * (1 + conf.dilate_w) * 32 * 32);

        if (has_h) {
            mark(kh_skip);

            // Advance kh = kh + 1.
            add(1, kh, kh, 1);
            add(1, ih, ih, 1 + conf.dilate_h);
            add(1 | SWSB(3), src_off_rd, src_off_rd,
                    (1 + conf.dilate_h) * conf.iw * 32 * 32);
            cmp(8 | lt | f0[0] | SWSB(3), kh, conf.kh);
            while_(8 | f0[0], kh_loop);

            // Restore src offset after kh loop.
            add(1 | SWSB(3), src_off_rd, src_off_rd,
                    -conf.kh * (1 + conf.dilate_h) * conf.iw * 32 * 32);
        }

        if (has_d) {
            mark(kd_skip);

            // Advance kd = kd + 1.
            add(1, kd, kd, 1);
            add(1, id, id, 1 + conf.dilate_d);
            add(1 | SWSB(3), src_off_rd, src_off_rd,
                    (1 + conf.dilate_d) * conf.ih * conf.iw * 32 * 32);
            cmp(8 | lt | f0[0] | SWSB(3), kd, conf.kd);
            while_(8 | f0[0], kd_loop);

            // Restore src offset after kd loop.
            add(1 | SWSB(3), src_off_rd, src_off_rd,
                    -conf.kd * (1 + conf.dilate_d) * conf.ih * conf.iw * 32
                            * 32);
        }

        // Advance ic_bytes = ic_bytes + 32.
        add(1, ic_bytes, ic_bytes, 32);
        add(1 | SWSB(2), src_off_rd, src_off_rd,
                conf.id * conf.ih * conf.iw * 32 * 32);
        cmp(8 | lt | f0[0] | SWSB(2), ic_bytes, ic_bytes_padded);
        while_(8 | f0[0], ic_loop);

        if (slm_nbuf == 2) {
            fence_and_signal();
            wait();
        }

        // Now complete the remaining multiply calls.
        for (int iter = 0; iter < slm_nbuf - 1; iter++) {
            if (iter + 1 < slm_nbuf - 1) {
                multiply();
                slm_buffer_advance();
                sync(SyncFunction::nop, SWSB<int32_t>(1));
            } else {
                multiply(/*skip_signal=*/true);
            }
        }

        // To ensure all DPASW calls updated their results.
        sync(SyncFunction::allwr);

        // Release registers to make them available for destination update.
        ra.safeRelease(A);
        ra.safeRelease(B);
        ra.safeRelease(A_tmp);
        ra.safeRelease(B_tmp);

        // Re-enable auto-SWSB back.
        enable_auto_swsb();

        read_update_write_dst();

        epilogue();

        // Kernel padding for instruction prefetch.
        for (int rep = 0; rep < 8; rep++)
            nop();
    }

    void epilogue() {
        memfence(tmp0);
        mov<uint32_t>(8, null, tmp0);

        slmfence(tmp0, r0);
        mov<int32_t>(8, null, tmp0);

        mov<uint32_t>(8, r255, r0);
        threadend(r255);
    }

    void allocate_registers() {
        tmp = ra.alloc_range(2);
        tmp0 = tmp[0];
        tmp1 = tmp[1];

        ithr0 = ra.alloc_sub<int32_t>();
        ithr1 = ra.alloc_sub<int32_t>();

        a_slm_rd = ra.alloc_range(2);
        b_slm_rd = ra.alloc_range(4);

        for (int i = 0; i < 2; i++)
            a_slm_off_rd_init[i] = ra.alloc_sub<int32_t>();

        for (int i = 0; i < 4; i++)
            b_slm_off_rd_init[i] = ra.alloc_sub<int32_t>();

        a_slm_off_wr_init = ra.alloc_sub<int32_t>();
        b_slm_off_wr_init = ra.alloc_sub<int32_t>();

        a_slm_wr = ra.alloc();
        b_slm_wr = ra.alloc();

        off_tmp = ra.alloc_sub<int32_t>();

        src_off_tg = ra.alloc_sub<int32_t>();
        wei_off_tg = ra.alloc_sub<int32_t>();

        mb = ra.alloc_sub<int32_t>();

        ic_bytes = ra.alloc_sub<int32_t>();
        oc = ra.alloc_sub<int32_t>();
        oc_fused = ra.alloc_sub<int32_t>();

        if (has_d) {
            od = ra.alloc_sub<int32_t>();
            id = ra.alloc_sub<int32_t>();
            kd = ra.alloc_sub<int32_t>();
        }

        if (has_h) {
            oh = ra.alloc_sub<int32_t>();
            ih = ra.alloc_sub<int32_t>();
            kh = ra.alloc_sub<int32_t>();
        }

        ow = ra.alloc_sub<int32_t>();
        iw = ra.alloc_sub<int32_t>();
        kw = ra.alloc_sub<int32_t>();

        id_init = ra.alloc_sub<int32_t>();
        ih_init = ra.alloc_sub<int32_t>();
        iw_init = ra.alloc_sub<int32_t>();

        oc_tg = ra.alloc_sub<int32_t>();
        ow_tg = ra.alloc_sub<int32_t>();
        iw_tg = ra.alloc_sub<int32_t>();

        slm_idx = ra.alloc().w(0);
        slm_buf_load = slm_idx.w(0);
        slm_buf_compute = slm_idx.w(1);
        slm_counter = slm_idx.d(1);

        src_addr = ra.alloc();
        wei_addr = ra.alloc();

        src_off_rd = ra.alloc_sub<int32_t>();

        A_tmp = ra.alloc_range(conf.oc_group == 4 ? 8 : 16);
        B_tmp = ra.alloc_range(conf.ow_group == 4 ? 8 : 16);
    }

    void enable_auto_swsb() {
        is_auto_swsb = true;
        setDefaultAutoSWSB(true);
    }

    void disable_auto_swsb() {
        is_auto_swsb = false;
        setDefaultAutoSWSB(false);
    }

    // Emulate integer division, math.iqot/math.irem do not work with integers.
    // For x >= 0, y > 0:
    //     qot = x / y
    //     rem = x % y
    // For now implementing very naive version, using a loop.
    void e_idiv(const Subregister &qot, const Subregister &rem,
            const Subregister &x, int y, const Subregister &tmp_d) {
        assert(is_auto_swsb);
        assert(x.getType() == DataType::d);
        assert(qot.getType() == DataType::d);
        assert(rem.getType() == DataType::d);

        auto qot_by_y = tmp_d;
        mov(1, qot_by_y, 0);
        mov(1, qot, 0);

        Label loop;
        mark(loop);

        add(1, qot, qot, 1);
        add(1, qot_by_y, qot_by_y, y);
        cmp(8 | le | f0[0], qot_by_y, x);

        while_(8 | f0[0], loop);

        add(1, qot_by_y, qot_by_y, -y);
        add(1, qot, qot, -1);
        add(1, rem, x, -qot_by_y);
    }

    void init_src_off_tg() {
        // (mb / 32) * (IC / 32) * ID * IH * IW * 32 * 32
        mul(1, tmp0.uq(0), mb, conf.id * conf.ih * conf.iw * ic_bytes_padded);
        mov(1, src_off_tg, tmp0.d(0));

        if (has_d) {
            // id * IH * IW * 32 * 32
            mul(1, tmp0.uq(0), id_init, conf.ih * conf.iw * 32 * 32);
            add(1, src_off_tg, src_off_tg, tmp0.d(0));
        }

        if (has_h) {
            // ih * IW * 32 * 32
            mul(1, tmp0.uq(0), ih_init, conf.iw * 32 * 32);
            add(1, src_off_tg, src_off_tg, tmp0.d(0));
        }

        // iw * 32 * 32
        mad(1, src_off_tg, src_off_tg, iw_tg, uint16_t(32 * 32));
    }

    void init_wei_off_tg() {
        // (oc / 32) * (IC / 32) * KH * KW * 32 * 32
        mul(1, tmp0.uq(0), oc_tg,
                ic_bytes_padded * conf.kd * conf.kh * conf.kw);
        mov(1, wei_off_tg, tmp0.d(0));
    }

    void slm_buffer_advance() {
        // Move to the next buffer: buf = buf - 1.
        add(2 | ge | f0[1], slm_idx, slm_idx(1), int16_t(-1));
        add(1, slm_counter, slm_counter, 1);
        // Wrap around: -1 -> (slm_nbuf - 1).
        add(2 | ~f0[1] | SWSB(2), slm_idx, slm_idx(1), int16_t(slm_nbuf));
    }

    void gmem2reg() {
        if (!enable_gmem_read) return;
        assert(!is_auto_swsb);

        // Load weights.
        Label wei_skip;
        // TODO: move condition out of the loop.
        if (oc_padded != oc_tg_padded) {
            cmp(8 | lt | f0[1], oc, oc_padded);
            if_(8 | f0[1], wei_skip, wei_skip);
        }
        {
            load(16 | SWSB(sb2, 1), B_tmp[0], block_hword(8), A64, wei_addr);
            add(1 | sb2.src, wei_addr.uq(0), wei_addr.uq(0), 32 * 32);
        }
        if (oc_padded != oc_tg_padded) {
            mark(wei_skip);
            endif(8);
        }

        // Load source.
        for (int iter = 0; iter < (conf.oc_group == 4 ? 1 : 2); iter++) {
            int ithr = iter * conf.oc_group;
            Label src_skip, src_end;
            // TODO: No padding and non-multiple OW case does not require >= 0
            // check.
            bool check_w = (has_pad_w || conf.ow != ow_padded);
            if (check_w) {
                // iw + ithr * SW < IW
                cmp(8 | lt | f1[0], iw, conf.iw - ithr * conf.stride_w);
                // iw + ithr * SW >= 0
                cmp(8 | f1[0] | ge, iw, -ithr * conf.stride_w);
                if_(8 | f1[0], src_skip, src_end);
            }
            add(1 | sb0.src, src_addr.uq(0), src_ptr, src_off_rd);
            if (iter == 1) {
                add(1 | SWSB(1), src_addr.uq(0), src_addr.uq(0),
                        2 * conf.stride_w * 32 * 32);
            }

            load(16 | SWSB(SBID(iter), 1), A_tmp[iter * 8], block_hword(8), A64,
                    src_addr);
            if (check_w) {
                else_(8, src_end, src_end);
                mark(src_skip);
                for (int i = 0; i < 8; i += 2) {
                    mov<float>(16 | SBID(iter).src, A_tmp[iter * 8 + i], 0.0f);
                }
                mark(src_end);
                endif(8);
            }
        }
    }

    void reg2smem() {
        if (!enable_smem_write) return;
        assert(!is_auto_swsb);

        mad(1, a_slm_wr.d(2), a_slm_off_wr_init, slm_buf_load,
                uint16_t(ab_slm_size / 16));
        mad(1, b_slm_wr.d(2), b_slm_off_wr_init, slm_buf_load,
                uint16_t(ab_slm_size / 16));

        sync(SyncFunction::nop, SWSB<int>(1));

        store(16 | SWSB(sb0, 2), block_oword(16), SLM, a_slm_wr, A_tmp[0]);
        if (conf.oc_group == 2) {
            sync(SyncFunction::nop, sb0.src);
            add(1, a_slm_wr.d(2), a_slm_wr.d(2),
                    uint16_t(a_slm_block_size * 2 / 16));
            store(16 | SWSB(sb1, 1), block_oword(16), SLM, a_slm_wr, A_tmp[8]);
        }

        store(16 | SWSB(sb2, 1), block_oword(16), SLM, b_slm_wr, B_tmp[0]);
    }

    void load_A(int i, const SBID &sb) {
        if (!enable_smem_read) return;
        assert(!is_auto_swsb);

        auto off = a_slm_rd[i / 8];
        mad(1, off.d(2), a_slm_off_rd_init[i / 8], slm_buf_compute,
                uint16_t(ab_slm_size / 16));
        load(16 | SWSB(sb, 1), A[i], block_oword(16), SLM, off);
    }

    void load_B(int i, const SBID &sb) {
        if (!enable_smem_read) return;
        assert(!is_auto_swsb);

        auto off = b_slm_rd[i / 8];
        mad(1, off.d(2), b_slm_off_rd_init[i / 8], slm_buf_compute,
                uint16_t(ab_slm_size / 16));
        load(16 | SWSB(sb, 1), B[i % 16], block_oword(16), SLM, off);
    }

    void fence_and_signal(bool skip_fence = false) {
        if (!enable_barrier) return;
        assert(!is_auto_swsb);

        // TODO: Replace by waiting on the first SLM read. This should
        // guarantee that the previous SLM writes are flushed - no need to use
        // fence.
        if (!skip_fence) {
            slmfence(sb15, tmp0, r0);
            mov<int32_t>(8 | sb15.dst, null, tmp0);
        }
        and_(8 | SWSB(1), tmp0.ud(), r0.ud(2), uint32_t(0x7F000000));
        barriermsg(SWSB(sb15, 1), tmp0);
    }

    void wait() {
        if (!enable_barrier) return;
        barrierwait();
    }

    void dpasw_typed(const InstructionModifier &mod, uint8_t sdepth,
            uint8_t rcount, const GRF &c_reg, const GRF &a_reg,
            const GRF &b_reg) {
        if (!enable_dpasw) return;

        dpasw(mod, sdepth, rcount, c_reg.retype(acc_type),
                c_reg.retype(acc_type), a_reg.retype(wei_type),
                b_reg.retype(src_type));
    }

    void multiply(bool skip_signal = false) {
        Label end, skip;

        cmp(8 | ge | f0[0], slm_counter, slm_nbuf - 1);

        if (conf.ow != ow_padded) cmp(8 | f0[0] | lt, ow, conf.ow);
        if (oc_padded != oc_tg_padded) cmp(8 | f0[0] | lt, oc_fused, oc_padded);
        if_(8 | f0[0], skip, end);

        if (slm_nbuf == 3) wait();

        load_B(0, sb10);
        load_B(8, sb11);
        load_A(0, sb8);
        load_A(8, sb9);

        // [0:8, 0:32]
        sync(SyncFunction::nop, sb10.dst);
        dpasw_typed(8 | Atomic | sb8.dst, 8, 8, C[0], B[0], A[0]);
        dpasw_typed(8 | Atomic, 8, 8, C[8], B[0], A[4]);
        dpasw_typed(8 | Atomic | sb9.dst, 8, 8, C[16], B[0], A[8]);
        dpasw_typed(8 | sb10, 8, 8, C[24], B[0], A[12]);

        load_B(16, sb10);

        // [8:16, 0:32]
        dpasw_typed(8 | Atomic | sb11.dst, 8, 8, C[32], B[8], A[0]);
        dpasw_typed(8 | Atomic, 8, 8, C[40], B[8], A[4]);
        dpasw_typed(8 | Atomic, 8, 8, C[48], B[8], A[8]);
        dpasw_typed(8 | sb11, 8, 8, C[56], B[8], A[12]);

        load_B(24, sb11);
        if (slm_nbuf == 3 && !skip_signal) fence_and_signal();

        // [16:24, 0:32]
        dpasw_typed(8 | Atomic | sb10.dst, 8, 8, C[64], B[0], A[0]);
        dpasw_typed(8 | Atomic, 8, 8, C[72], B[0], A[4]);
        dpasw_typed(8 | Atomic, 8, 8, C[80], B[0], A[8]);
        dpasw_typed(8 | sb10, 8, 8, C[88], B[0], A[12]);

        // [24:32, 0:32]
        dpasw_typed(8 | Atomic | sb11.dst, 8, 8, C[96], B[8], A[0]);
        dpasw_typed(8 | Atomic, 8, 8, C[104], B[8], A[4]);
        dpasw_typed(8 | Atomic, 8, 8, C[112], B[8], A[8]);
        dpasw_typed(8 | sb11, 8, 8, C[120], B[8], A[12]);

        else_(8, end, end);
        mark(skip);
        if (slm_nbuf == 3) {
            wait();
            if (!skip_signal) fence_and_signal();
        }
        mark(end);
        endif(8);
    }

    // Computes register offset for n-th row (across 32n block). The rows are
    // interleaved after dpasw.
    int mb_off(int mb_idx) {
        int shuf[4] = {0, 2, 1, 3};
        int x = mb_idx / 4;
        int y = (x / 4) * 4 + shuf[x % 4];
        return y * 4 + (mb_idx % 4);
    }

    void apply_post_ops(const GRF &C_reg, int idx) {
        auto &all_po = attr_info.all_post_ops;
        for (int i = 0; i < all_po.len(); i++) {
            auto &e = all_po.entry_[i];
            switch (e.kind) {
                case primitive_kind::sum:
                    if (e.sum.scale != 0) {
                        auto old = C_old[idx / 8];
                        mad(8, C_reg.f(), C_reg.f(), old.f(), sum_scale);
                    }
                    break;
                case primitive_kind::eltwise: {
                    // TODO: Move out of the loop and reuse to reduce overhead.
                    jit_eltwise_injector_f32<HW::Gen12HP> inj(this,
                            e.eltwise.alg, e.eltwise.alpha, e.eltwise.beta,
                            e.eltwise.scale);
                    auto scratch = ra.alloc_range(inj.preferred_scratch_regs());
                    inj.set_scratch(scratch);
                    inj.prepare();
                    inj.compute(GRFRange(C_reg, 1));
                    ra.safeRelease(scratch);
                    break;
                }
                default: assert(!"not supported");
            }
        }
    }

    void read_update_write_dst_range(
            int mb_idx, int mb_step, int oc_idx, int oc_step) {

        auto c_off = [&](int mb_inner, int oc_inner) {
            return mb_off(mb_idx + mb_inner) + (oc_idx + oc_inner) / 8 * 32;
        };

        auto C_tmp = ra.alloc_range(4);

        bool do_f32_cvt = conf.with_bias || attr_info.with_oscales
                || attr_info.all_post_ops.len() > 0;

        // Convert s32 -> f32 in-place if needed.
        DataType post_op_type = do_f32_cvt ? DataType::f : acc_type;
        if (acc_type != post_op_type) {
            for_(int mb_inner = 0; mb_inner < mb_step; mb_inner++)
            for_(int oc_inner = 0; oc_inner < oc_step; oc_inner += 8)
            {
                auto C_reg = C[c_off(mb_inner, oc_inner)];
                mov(8, C_reg.retype(post_op_type), C_reg.retype(acc_type));
            }
        }

        // Load old values for sum post-op.
        if (attr_info.with_sum) {
            // Read 128 bytes of destination and convert to f32.
            GRFRange C_old_tmp = ra.alloc_range(4);
            load(16, C_old_tmp[0], block_oword(8), A64, dst_addr);

            C_old = (dst_size == 4)
                    ? C_old_tmp
                    : ra.alloc_range(4 * (sizeof(float) / dst_size));
            convert(128 / dst_size, C_old[0].f(0),
                    C_old_tmp[0].sub(0, sum_type));
            if (C_old != C_old_tmp) ra.safeRelease(C_old_tmp);
        }

        for_(int mb_inner = 0; mb_inner < mb_step; mb_inner++)
        for_(int oc_inner = 0; oc_inner < oc_step; oc_inner += 8)
        {
            auto C_reg = C[c_off(mb_inner, oc_inner)];

            // Apply bias.
            if (conf.with_bias)
                add(8, C_reg.f(), C_reg.f(), bia[oc_inner / 8].f());

            // Apply output scales.
            if (attr_info.with_oscales) {
                if (attr_info.with_common_oscales) {
                    mul(8, C_reg.f(), C_reg.f(), common_oscales);
                } else {
                    // Per-oc output scales.
                    mul(8, C_reg.f(), C_reg.f(), oscales[oc_inner / 8].f());
                }
            }

            apply_post_ops(C_reg, mb_inner * oc_step + oc_inner);
        }

        // Convert to the destination type and write.
        int c_bytes = 0;
        for_(int mb_inner = 0; mb_inner < mb_step; mb_inner++)
        for_(int oc_inner = 0; oc_inner < oc_step; oc_inner += 8)
        {
            auto C_reg = C[c_off(mb_inner, oc_inner)];
            auto packed = C_tmp[c_bytes / 32]
                                  .b(c_bytes % 32)
                                  .reinterpret(0, dst_type);
            convert(8, packed, C_reg.retype(post_op_type)[0]);
            c_bytes += 8 * dst_size;
        }
        assert(c_bytes == 128);
        store(16, block_oword(8), A64, dst_addr, C_tmp[0]);

        ra.safeRelease(C_tmp);
        if (attr_info.with_sum) ra.safeRelease(C_old);
    }

    void load_bias(int oc_idx, int oc_step) {
        // TODO: Add bounds check.
        switch (bia_size) {
            case 2: {
                assert(oc_step == 16);
                auto idx_vec = ra.alloc().uw();
                mov(8, idx_vec, Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
                add(8, idx_vec.uw(8), idx_vec, uint16_t(8));
                shl(16, idx_vec, idx_vec,
                        uint16_t(ngen::utils::log2(bia_size)));
                add3(16, oc_off[0].d(), bia_off_init, idx_vec,
                        uint16_t(oc_idx * bia_size));
                load(16, bia[0], scattered_byte(bia_size), Surface(bia_surf),
                        oc_off);
                ra.safeRelease(idx_vec);
                break;
            }
            case 4:
                add(1, oc_off[0].d(2), bia_off_init, oc_idx * bia_size);
                load(16, bia[0], aligned_block_oword(oc_step * bia_size / 16),
                        Surface(bia_surf), oc_off[0]);
                break;
            default: assert(!"not expected");
        }
    }

    void load_per_oc_oscales(int oc_idx, int oc_step) {
        // TODO: Add bounds check.
        add(1, oc_off[0].d(2), oscales_off_init,
                int32_t(oc_idx * sizeof(float)));
        load(16, oscales[0], aligned_block_oword(oc_step * sizeof(float) / 16),
                Surface(oscales_surf), oc_off[0]);
    }

    // Fixes sub-register base/offset when offset is too big and crosses GRF
    // boundary.
    Subregister fixup_sub(const Subregister &sub, int off, int stride = 1) {
        auto new_off = (sub.getOffset() + off) * sub.getBytes();
        auto grf = GRF(sub.getBase() + new_off / 32).retype(sub.getType());
        return grf[(new_off % 32) / sub.getBytes()];
    }

    // to_stride is always assumed to be 1.
    void convert_impl(
            int simd, Subregister to, Subregister from, int from_stride) {
        auto from_type = from.getType();
        auto to_type = to.getType();
        auto from_bytes = getBytes(from_type);
        auto to_bytes = getBytes(to_type);

        assert(utils::one_of(simd, 8, 16));

        if (to.getType() == from.getType() && from_stride == 1) {
            mov(simd, to(1), from(from_stride));
            return;
        }

        // bf16 -> f32:
        // - bf16 must be packed: use left shift instead.
        if (from_type == DataType::bf && to_type == DataType::f) {
            to.setType(DataType::ud);
            from.setType(DataType::uw);
            shl(simd, to(1), from(from_stride), uint16_t(16));
            return;
        }

        // f32 -> bf16 or f32 -> f16:
        // - SIMD16 does not support mixed mode move.
        if (simd == 16 && from_type == DataType::f
                && utils::one_of(to_type, DataType::bf, DataType::hf)) {
            mov(8, to(1), from(from_stride));
            mov(8, fixup_sub(to, 8)(1), fixup_sub(from, 8, from_stride));
            return;
        }

        // f32/s32 -> s8/u8:
        // - Use saturation
        // - s8/u8 must be DW-strided: use temporary
        if (from_bytes == 4
                && utils::one_of(to_type, DataType::b, DataType::ub)) {
            auto strided = tmp0.retype(to_type)[0](4);
            mov(simd | sat, strided, from(from_stride));
            mov(simd, to(1), strided);
            return;
        }

        // s8/u8 -> f32/s32:
        // - s8/u8 must be DW-strided: use temporary
        if (utils::one_of(from_type, DataType::b, DataType::ub)
                && to_bytes == 4) {
            auto strided = tmp0.retype(from_type)[0](4);
            mov(simd, strided, from(from_stride));
            mov(simd, to(1), strided);
            return;
        }

        mov(simd, to(1), from(from_stride));
    }

    // to_stride is always assumed to be 1.
    void convert(int width, const Subregister &to, const Subregister &from,
            int from_stride = 1) {
        assert(width % 8 == 0);

        int simd = std::min(width, 16);
        for (int off = 0; off < width;) {
            convert_impl(simd, fixup_sub(to, off),
                    fixup_sub(from, off, from_stride), from_stride);
            off += simd;
            if (off + simd >= width) simd = 8;
        }
    }

    void read_update_write_dst() {
        if (!enable_gmem_write) return;

        dst_addr = ra.alloc();
        dst_addr_init = ra.alloc_sub<uint64_t>();

        Label skip;

        if (conf.ow != ow_padded || oc_padded != oc_tg_padded) {
            cmp(8 | lt | f0[0], ow, conf.ow);
            cmp(8 | f0[0] | lt, oc, oc_padded);
            if_(8 | f0[0], skip, skip);
        }

        // Compute destination address.
        // (mb / 32) * (OC / 32) * OD * OH * OW * 32 * 32
        mul(1, dst_addr_init, mb, conf.od * conf.oh * conf.ow * oc_padded);

        // (oc / 32) * OD * OH * OW * 32 * 32
        mul(1, tmp0.uq(0), oc, conf.od * conf.oh * conf.ow * 32);
        add(1, dst_addr_init.ud(0), dst_addr_init.ud(0), tmp0.d(0));

        if (has_d) {
            // od * OH * OW * 32 * dst_oc_block
            mul(1, tmp0.uq(0), od, conf.oh * conf.ow * 32 * dst_oc_block);
            add(1, dst_addr_init.ud(0), dst_addr_init.ud(0), tmp0.d(0));
        }

        if (has_h) {
            // oh * OW * 32 * dst_oc_block
            mul(1, tmp0.uq(0), oh, conf.ow * 32 * dst_oc_block);
            add(1, dst_addr_init.ud(0), dst_addr_init.ud(0), tmp0.d(0));
        }

        // ow * 32 * dst_oc_block
        mad(1, dst_addr_init.ud(0), dst_addr_init.ud(0), ow,
                uint16_t(32 * dst_oc_block));

        // Convert offset from elements to bytes.
        shl(1, dst_addr_init, dst_addr_init, ngen::utils::log2(dst_size));
        add(1, dst_addr_init, dst_addr_init, dst_ptr);

        int oc_step = (acc_type == DataType::f) ? 16 : 32;
        int mb_step = 128 / (oc_step * dst_size);

        oc_off = ra.alloc_range(2);

        // Initialize bias offset.
        if (conf.with_bias) {
            bia = ra.alloc_range(oc_step * sizeof(float) / 32);
            bia_off_init = ra.alloc_sub<int32_t>();
            mul(1, bia_off_init, oc, uint16_t(bia_size));
        }

        // Initialize output scales offset.
        if (attr_info.with_per_oc_oscales) {
            oscales = ra.alloc_range(oc_step * sizeof(float) / 32);
            oscales_off_init = ra.alloc_sub<int32_t>();
            mul(1, oscales_off_init, oc, uint16_t(sizeof(float)));
        } else if (attr_info.with_common_oscales
                && attr_info.with_runtime_oscales) {
            common_oscales = ra.alloc().f(0);
            mov(1, tmp0.d(0), 0);
            load(1, common_oscales, scattered_dword(1), Surface(oscales_surf),
                    tmp0);
        }

        // Setup for sum post-op.
        if (attr_info.with_sum) {
            sum_type = dst_type;
            if (attr_info.sum_data_type != data_type::undef)
                sum_type = convert_dnnl_type_to_ngen(attr_info.sum_data_type);
            sum_scale = ra.alloc_sub<float>();
            mov(1, sum_scale, attr_info.sum_scale);
        }

        for (int oc_idx = 0; oc_idx < 32; oc_idx += oc_step) {
            // Load and convert bias to float.
            if (conf.with_bias) {
                load_bias(oc_idx, oc_step);
                convert(oc_step, bia[0].f(0), bia[0].retype(bia_type)[0],
                        4 / bia_size);
            }

            // Load per-oc output scales.
            if (attr_info.with_per_oc_oscales)
                load_per_oc_oscales(oc_idx, oc_step);

            add(1, dst_addr.uq(0), dst_addr_init,
                    oc_idx * conf.od * conf.oh * conf.ow * 32 * dst_size);

            for (int mb_idx = 0; mb_idx < 32; mb_idx += mb_step) {
                // Update and write 128 bytes.
                read_update_write_dst_range(mb_idx, mb_step, oc_idx, oc_step);
                if (mb_idx + mb_step < 32)
                    add(1, dst_addr.uq(0), dst_addr.uq(0), 128);
            }
        }

        if (conf.ow != ow_padded || oc_padded != oc_tg_padded) {
            mark(skip);
            endif(8);
        }
    }

    const conv_conf_t &conf;
    const attr_info_t &attr_info;

    int ic_bytes_padded;
    int oc_padded;
    int oc_tg_padded;
    int ow_padded;

    bool has_pad_d;
    bool has_pad_h;
    bool has_pad_w;

    bool has_h;
    bool has_d;

    DataType src_type;
    DataType wei_type;
    DataType bia_type;
    DataType dst_type;
    DataType acc_type;

    int src_size;
    int bia_size;
    int dst_size;

    int dst_oc_block;

    int slm_nbuf;
    int a_slm_block_size = (4 * 284);
    int b_slm_block_size = (4 * 284);
    int a_slm_size;
    int b_slm_size;
    int ab_slm_size;

    Subregister group_id_0 = r0.ud(1);
    Subregister group_id_1 = r0.ud(6);
    Subregister group_id_2 = r0.ud(7);

    GRF local_id_0;
    GRF local_id_1;

    Subregister src_ptr;
    Subregister wei_ptr;
    Subregister dst_ptr;

    int bia_surf;
    int oscales_surf;

    GRFRange tmp;
    GRF tmp0;
    GRF tmp1;

    Subregister ithr0;
    Subregister ithr1;

    GRFRange a_slm_rd;
    GRFRange b_slm_rd;

    // SLM read offsets in owords for the first buffer.
    Subregister a_slm_off_rd_init[2];
    Subregister b_slm_off_rd_init[4];

    Subregister a_slm_off_wr_init;
    Subregister b_slm_off_wr_init;

    GRF a_slm_wr;
    GRF b_slm_wr;

    Subregister off_tmp;

    Subregister src_off_tg;
    Subregister wei_off_tg;

    Subregister mb;
    Subregister ic_bytes;
    Subregister oc;
    Subregister oc_fused;
    Subregister oc_tg;

    Subregister od, oh, ow;
    Subregister id, ih, iw;
    Subregister kd, kh, kw;

    Subregister iw_init;
    Subregister ih_init;
    Subregister id_init;

    Subregister iw_tg;
    Subregister ow_tg;

    Subregister slm_idx;
    Subregister slm_buf_load;
    Subregister slm_buf_compute;
    Subregister slm_counter;

    GRF src_addr;
    Subregister src_off_rd;

    GRF wei_addr;

    GRFRange oc_off;
    Subregister bia_off_init;

    GRF dst_addr;
    Subregister dst_addr_init;

    GRFRange A_tmp;
    GRFRange B_tmp;

    GRFRange bia;

    // Post-ops support.
    GRFRange C_old;
    DataType sum_type;
    Subregister sum_scale;

    // Output scales.
    Subregister common_oscales;
    Subregister oscales_off_init;
    GRFRange oscales;

    GRFRange A = r90 - r105;
    GRFRange B = r106 - r121;
    GRFRange C = r128 - r255;

    bool is_auto_swsb = false;

    // 256 registers.
    RegisterAllocator<256> ra;
};

status_t gen12hp_conv_fwd_create_kernel(const conv_conf_t &conf,
        compute::kernel_t *kernel, gpu_primitive_t *primitive,
        engine_t *engine) {
    gen12hp_conv_fwd_kernel_t ngen_kernel(conf);
    return primitive->create_kernel(engine, kernel, ngen_kernel);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

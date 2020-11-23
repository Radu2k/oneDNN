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

#include "gpu/jit/gen12hp_conv_data_kernel.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>
#include <initializer_list>

#include "common/eltwise_pd.hpp"
#include "common/math_utils.hpp"
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
namespace {

using namespace ngen;
using namespace prop_kind;

// Controls for performance debugging.
const bool enable_dpasw = true;

const bool enable_gmem_read = true;
const bool enable_gmem_write = true;

const bool enable_smem_read = true;
const bool enable_smem_write = true;

const bool enable_barrier = true;

class gen12hp_conv_data_kernel_t;

// Convert registers between data types.
// to_stride is always assumed to be 1.
class convertor_t {
public:
    convertor_t(gen12hp_conv_data_kernel_t *host, int width, DataType to_type,
            DataType from_type, int from_stride = 1)
        : host_(host)
        , width_(width)
        , to_type_(to_type)
        , from_type_(from_type)
        , from_stride_(from_stride)
        , from_bytes_(getBytes(from_type))
        , to_bytes_(getBytes(to_type)) {
        assert(width % 8 == 0);
    }

    int preferred_scratch_regs() const {
        if (from_bytes_ == 4 && to_bytes_ == 1) return 8;
        if (from_bytes_ == 1 && to_bytes_ == 4) return 8;
        return 0;
    }

    int max_batch_size() const {
        if (preferred_scratch_regs() == 0) return 128;
        return scratch_.getLen() * 8;
    }

    int phase_count() const {
        if (from_type_ == DataType::f
                && utils::one_of(to_type_, DataType::bf, DataType::hf))
            return 2;
        if (from_bytes_ == 4 && to_bytes_ == 1) return 2;
        if (from_bytes_ == 1 && to_bytes_ == 4) return 2;
        return 1;
    }

    void convert(const Subregister &to, const Subregister &from,
            RegisterAllocator &ra) {
        scratch_ = ra.alloc_range(preferred_scratch_regs());

        int bmax = max_batch_size();
        for (int width0 = 0; width0 < width_; width0 += bmax) {
            for (int phase = 0; phase < phase_count(); phase++) {
                int batch = std::min(width_ - width0, bmax);
                int simd = std::min(batch, 16);
                for (int off = width0; off < width0 + batch;) {
                    convert_impl(simd, off - width0, fixup_sub(to, off),
                            fixup_sub(from, off, from_stride_), phase);
                    off += simd;
                    if (off + simd > width0 + batch) simd = 8;
                }
            }
        }
        ra.safeRelease(scratch_);
    }

private:
    void convert_impl(
            int simd, int off, Subregister to, Subregister from, int phase);

    // Fixes sub-register base/offset when offset is too big and crosses GRF
    // boundary.
    Subregister fixup_sub(const Subregister &sub, int off, int stride = 1) {
        auto new_off = (sub.getOffset() + off) * sub.getBytes();
        auto grf = GRF(sub.getBase() + new_off / 32).retype(sub.getType());
        return grf[(new_off % 32) / sub.getBytes()];
    }

    gen12hp_conv_data_kernel_t *host_;

    int width_;
    DataType to_type_;
    DataType from_type_;
    int to_bytes_;
    int from_bytes_;
    int from_stride_;

    GRFRange scratch_;
};

// Represents an (mb_len x w_len) region to read.
class nw_read_region_t {
public:
    nw_read_region_t(gen12hp_conv_data_kernel_t *host, int mb_len, int w_val,
            int w_shift, int w_len)
        : host(host)
        , mb_len(mb_len)
        , w_val(w_val)
        , w_shift(w_shift)
        , w_len(w_len) {}

    nw_read_region_t(gen12hp_conv_data_kernel_t *host, int mb_len, int w_len)
        : host(host)
        , mb_len(mb_len)
        , w_val(INT_MAX)
        , w_shift(0)
        , w_len(w_len) {}

    bool with_w_padding() const { return w_val != INT_MAX; }

    void read_and_reorder();

    gen12hp_conv_data_kernel_t *host;

    int mb_len;
    int w_val;
    int w_shift;
    int w_len;

    Label label;
};

// Helper class to implement reduction across (IC * KD * KH * KW) and SLM
// buffering.
class loop_iterator_t {
public:
    loop_iterator_t(const conv_conf_t &conf, int unroll, int slm_nbuf,
            int ab_slm_size, bool check_src_load)
        : conf(conf)
        , unroll(unroll)
        , slm_nbuf(slm_nbuf)
        , ab_slm_size(ab_slm_size)
        , check_src_load(check_src_load) {

        assert(unroll % (conf.kd * conf.kh * conf.kw) == 0);
        assert(slm_nbuf == 2 || slm_nbuf == 3);

        int ic_iters = utils::div_up(conf.ic, conf.ic_block);

        iters = ic_iters * conf.kd * conf.kh * conf.kw + (slm_nbuf - 1);
        assert(iters >= slm_nbuf);

        ramp_up_iters = slm_nbuf;
        ramp_down_iters = std::min(slm_nbuf - 1, iters - ramp_up_iters);
        body_iters = iters - ramp_up_iters - ramp_down_iters;

        body_iters = utils::rnd_dn(body_iters, unroll);
        ramp_down_iters = iters - ramp_up_iters - body_iters;

        assert(ramp_up_iters + body_iters + ramp_down_iters == iters);

        iter = 0;
        riter = iters - 1;
    }

    loop_iterator_t &operator++() {
        ++iter;
        --riter;

        if (!do_gmem2reg()) return *this;

        if (++kd < conf.kd) return *this;
        kd = 0;

        if (++kw < conf.kw) return *this;
        kw = 0;

        if (++kh < conf.kh) return *this;
        kh = 0;

        ic += conf.ic_block;
        return *this;
    }

    void advance(int n) {
        assert(n % unroll == 0);

        iter += n;
        riter -= n;
    }

    bool do_multiply() const { return iter >= slm_nbuf - 1; }
    bool is_first_multiply() const { return iter == slm_nbuf - 1; }
    bool is_last_multiply() const { return riter == 0; }

    bool do_gmem2reg() const { return riter >= slm_nbuf - 1; }

    int gmem_read_src_off_update() const;

    int smem_read_off_update() const {
        assert(do_multiply());

        int cur_slm_idx = (iter + 1) % slm_nbuf;
        int next_slm_idx = (iter + 2) % slm_nbuf;
        return next_slm_idx * ab_slm_size - cur_slm_idx * ab_slm_size;
    }

    int smem_write_off_update() const {
        assert(do_gmem2reg());

        int cur_slm_idx = iter % slm_nbuf;
        int next_slm_idx = (iter + 1) % slm_nbuf;
        return next_slm_idx * ab_slm_size - cur_slm_idx * ab_slm_size;
    }

    // NCdhw32n32c layout.
    int src_off_impl() const {
        int off = 0;
        off += (ic / conf.ic_block) * conf.id * conf.ih * conf.iw
                * conf.mb_block * 32;
        off += kd * (1 + conf.dilate_d) * conf.ih * conf.iw * conf.mb_block
                * 32;
        off += kh * (1 + conf.dilate_h) * conf.iw * conf.mb_block * 32;
        off += kw * (1 + conf.dilate_w) * conf.mb_block * 32;
        return off;
    }

    int iw_update() const;
    int ih_update() const;
    int id_update() const;

    const conv_conf_t &conf;
    int unroll;
    int slm_nbuf;
    int ab_slm_size;
    bool check_src_load;

    int iters;
    int ramp_up_iters;
    int body_iters;
    int ramp_down_iters;

    // iter + riter = iters - 1
    int iter;
    int riter;

    int ic = 0;
    int kd = 0;
    int kh = 0;
    int kw = 0;
};

loop_iterator_t operator+(const loop_iterator_t &a, int b) {
    assert(b >= 0);
    loop_iterator_t ret = a;
    for (int i = 0; i < b; i++)
        ++ret;
    return ret;
}

int loop_iterator_t::gmem_read_src_off_update() const {
    assert(do_gmem2reg());

    int cur_off = src_off_impl();
    int next_off = (*this + 1).src_off_impl();

    return next_off - cur_off;
}

int loop_iterator_t::iw_update() const {
    int cur_iw = kw * (1 + conf.dilate_w);
    int next_iw = (*this + 1).kw * (1 + conf.dilate_w);
    return next_iw - cur_iw;
}

int loop_iterator_t::ih_update() const {
    int cur_ih = kh * (1 + conf.dilate_h);
    int next_ih = (*this + 1).kh * (1 + conf.dilate_h);
    return next_ih - cur_ih;
}

int loop_iterator_t::id_update() const {
    int cur_id = kd * (1 + conf.dilate_d);
    int next_id = (*this + 1).kd * (1 + conf.dilate_d);
    return next_id - cur_id;
}

class gen12hp_conv_data_kernel_t : public jit_generator<HW::Gen12HP> {
public:
    friend class convertor_t;
    friend class nw_read_region_t;

    gen12hp_conv_data_kernel_t(const conv_conf_t &conf)
        : conf(conf), attr_info(conf.attr_info), ra(HW::Gen12HP) {

        src_type = convert_dnnl_type_to_ngen(conf.src_data_type);
        wei_type = convert_dnnl_type_to_ngen(conf.weights_data_type);
        bia_type = convert_dnnl_type_to_ngen(conf.bias_data_type);
        dst_type = convert_dnnl_type_to_ngen(conf.dst_data_type);
        acc_type = utils::one_of(src_type, DataType::b, DataType::ub)
                ? DataType::d
                : DataType::f;

        src_size = (int)types::data_type_size(conf.src_data_type);
        wei_size = (int)types::data_type_size(conf.weights_data_type);
        bia_size = (int)types::data_type_size(conf.bias_data_type);
        dst_size = (int)types::data_type_size(conf.dst_data_type);

        is_4x2_tg = (conf.sp_group == 4) && (conf.oc_group == 2);

        kdhw = conf.kd * conf.kh * conf.kw;
        idhw = conf.id * conf.ih * conf.iw;
        odhw = conf.od * conf.oh * conf.ow;

        mb_padded = utils::rnd_up(conf.mb, conf.mb_block);
        oc_padded = utils::rnd_up(conf.oc, conf.oc_block);
        ic_padded = utils::rnd_up(conf.ic, conf.ic_block);
        ow_padded = utils::rnd_up(conf.ow, conf.sp_group);

        oc_tg_padded = utils::rnd_up(conf.oc, conf.oc_group * conf.oc_block);
        ic_bytes_padded = ic_padded * src_size;

        int64_t src_bytes = (int64_t)mb_padded * ic_padded * idhw * src_size;
        int64_t wei_bytes = (int64_t)oc_padded * ic_padded * kdhw * wei_size;

        is_src_off_64_bit = (src_bytes > std::numeric_limits<int32_t>::max());
        is_wei_off_64_bit = (wei_bytes > std::numeric_limits<int32_t>::max());

        mb_block = conf.mb_block;
        assert(utils::one_of(mb_block, 32, 40));
        assert(mb_block % conf.oc_group == 0);

        is_1st = utils::one_of(conf.ic, 3, 4);
        if (is_1st) {
            // Only KW = 7 is supported.
            assert(conf.kw == 7);
            mb_read01 = utils::rnd_up(conf.mb_block / conf.oc_group, 4);
            mb_read23 = (conf.mb_block - mb_read01 * (conf.oc_group - 2)) / 2;
        }

        // Destination layout:
        // - <mb_block>n16c for f16/bf16/f32
        // - <mb_block>n32c for s8/u8/s32
        dst_oc_block = (acc_type == DataType::f) ? 16 : 32;

        if (ic_bytes_padded * kdhw >= 128 && conf.oc_group == 4) {
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

        if (conf.ver == ver_v1) {
            check_src_load = (has_pad_w || conf.ow != ow_padded);
            do_ic_loop = (conf.ic > conf.ic_block);
            do_kw_loop = (conf.kw > 1 || check_src_load) && !is_1st;
        } else {
            // conf.ver == ver_v2.
            check_src_load = (has_pad_w || has_pad_h || has_pad_d
                    || (odhw % conf.sp_group != 0));
            do_ic_loop = false;
            do_kw_loop = false;
        }

        // Kernel interface.
        newArgument("src", ExternalArgumentType::GlobalPtr);
        newArgument("wei", ExternalArgumentType::GlobalPtr);
        newArgument("bia", ExternalArgumentType::GlobalPtr);
        newArgument("dst", ExternalArgumentType::GlobalPtr);
        newArgument("oscales", ExternalArgumentType::GlobalPtr);
        newArgument("common_oscales", DataType::f);

        externalName("gen12hp_conv_data");
        requireLocalID(3);
        requireLocalSize();
        requireGRF(256);
        requireSIMD(8);
        requireBarrier();
        requireDPAS();

        // Implement double/triple SLM buffering. Layout for int8 for 4x4
        // thread-group:
        // - 1st convolution:
        //     src - 4w x <mb_block>n8w4c
        //     wei - 4o x 4o8w8o4i
        // - other convolutions:
        //     src - 4w x <mb_block>n32c
        //     wei - 4o x 4o8i8o4i
        // Pad inner blocks to avoid SLM bank conflicts.
        // Apply renaming: src -> A, wei -> B, dst -> C.
        switch (mb_block) {
            case 32:
                a_slm_block_size = 1104;
                a_slm_blocks = 2;
                break;
            case 40:
                a_slm_block_size = 1376;
                a_slm_blocks = 3;
                break;
            default: assert(!"not expected");
        }
        b_slm_block_size = 1104;
        b_slm_blocks = 4;

        a_slm_size = conf.sp_group * a_slm_block_size;
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

        // Manually allocate and claim A, B, C.
        reuse_b_regs = (conf.oc_group != 4 || is_1st);
        A = GRFRange(r34, mb_block / 2);
        B = GRFRange(r54, reuse_b_regs ? 16 : 32);
        C = GRFRange(r96, mb_block * 4);

        ra.claim(A);
        ra.claim(B);
        ra.claim(C);

        allocate_registers();

        // Fill nw_read_regions for 1st convolution and zero-initialize
        // buffer registers.
        if (is_1st) {
            int mb_cases = (mb_read01 != mb_read23 ? 2 : 1);
            int mb_lens[2] = {mb_read01, mb_read23};

            // No-padding region.
            for (int i = 0; i < mb_cases; i++) {
                nw_read_regions.emplace_back(this, mb_lens[i], conf.kw);
            }

            // Regions with padding.
            for (int i_ow = 0; i_ow < ow_padded; i_ow++) {
                int i_iw = -conf.l_pad + i_ow * conf.stride_w;
                if (i_iw >= 0 && i_iw + conf.kw <= conf.iw) continue;

                int w_beg = std::max(i_iw, 0);
                int w_end = std::min(conf.iw, i_iw + conf.kw);
                int w_len = std::max(0, w_end - w_beg);
                for (int i = 0; i < mb_cases; i++) {
                    nw_read_regions.emplace_back(
                            this, mb_lens[i], i_iw, w_beg - i_iw, w_len);
                }
            }

            // Set registers to zero to account for zero-padded area.
            for (int i = 0; i < A_tmp.getLen(); i += 2) {
                mov<float>(16, A_tmp[i], 0.0f);
            }
        }

        // Start kernel body.

        // Header for signal.
        and_(8, signal_header.ud(), r0.ud(2), uint32_t(0x7F000000));

        // ithr0 = get_local_id(0) / 8    [0, oc_group - 1]
        // ithr1 = get_local_id(1)        [0, sp_group - 1]
        shr<uint32_t>(1, ithr0, local_id_0.uw(0), 3);
        mov(1, ithr1, local_id_1.uw(0));

        // Initialize SLM read offset for A in owords.
        // Account for DPASW offset.
        and_<uint32_t>(1, fused_idx, ithr0, 1);

        for (int i = 0; i < a_slm_blocks; i++) {
            int fused_block = std::min(mb_block - i * 16, 16);
            mul(1, off_tmp, fused_idx, uint16_t(fused_block));
            mad(1, a_slm_off_rd_init[i], off_tmp, ithr1,
                    uint16_t(a_slm_block_size / 16));
            add(1, a_slm_off_rd_init[i], a_slm_off_rd_init[i], i * 32);
        }

        // Initialze SLM read offset for B in owords.
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
        mul(1, mb, group_id_2, uint16_t(mb_block));

        pad_sign = (conf.prop_kind == backward_data) ? 1 : -1;
        if (conf.ver == ver_v1) {
            // od = get_group_id(1) / (OW_PADDED / sp_group) / OH
            // oh = get_group_id(1) / (OW_PADDED / sp_group) % OH
            // ow = get_group_id(1) % (OW_PADDED / sp_group) * sp_group +
            //      get_local_id(1)
            unpack_1d_to_3d(group_id_1.d(0), ow_tg, ow_padded / conf.sp_group,
                    oh, conf.oh, od, conf.od);
            mul<int32_t>(1, ow_tg, ow_tg, uint16_t(conf.sp_group));
            add<int32_t>(1, ow, ow_tg, ithr1);
        } else {
            // conf.ver == ver_v2.
            // Boundary conditions: sp[idx] <= sp_bound[idx]:
            //   [0]   iw_load <= IW - 1
            //   [1]  -iw_load <= 0
            //   [2]   ih_load <= IH - 1
            //   [3]  -ih_load <= 0
            //   [4]   id_load <= ID - 1
            //   [5]  -id_load <= 0
            if (check_src_load) {
                mov(8, sp[0], 0);
                if (is_4x2_tg) mov(8, sp[1], 0);
                mov(8, sp_bound, 0);
                mov(1, sp_bound[0], conf.iw - 1);
                mov(1, sp_bound[2], conf.ih - 1);
                mov(1, sp_bound[4], conf.id - 1);
            }

            auto odhw = tmp0.d(0);
            Subregister odhw_load[2] = {tmp0.d(1), tmp0.d(2)};

            mul(1, odhw, group_id_1.d(0), conf.sp_group);
            add(1, odhw_load[0], odhw, ithr0);
            if (is_4x2_tg) add(1, odhw_load[1], odhw_load[0], 2);

            add(1, odhw, odhw, ithr1);

            unpack_1d_to_3d(odhw, ow, conf.ow, oh, conf.oh, od, conf.od);

            for (int i = 0; i < (is_4x2_tg ? 2 : 1); i++) {
                unpack_1d_to_3d(odhw_load[i], iw_load[i], conf.ow, ih_load[i],
                        conf.oh, id_load[i], conf.od);

                if (has_d) {
                    mul(1, id_load[i], id_load[i], uint16_t(conf.stride_d));
                    if (conf.f_pad > 0)
                        add(1, id_load[i], id_load[i], pad_sign * conf.f_pad);
                }

                if (has_h) {
                    mul(1, ih_load[i], ih_load[i], uint16_t(conf.stride_h));
                    if (conf.t_pad > 0)
                        add(1, ih_load[i], ih_load[i], pad_sign * conf.t_pad);
                }

                mul(1, iw_load[i], iw_load[i], uint16_t(conf.stride_w));
                if (conf.l_pad > 0)
                    add(1, iw_load[i], iw_load[i], pad_sign * conf.l_pad);
            }

            if (check_src_load) {
                for (int i = 0; i < (is_4x2_tg ? 2 : 1); i++) {
                    mov(1, sp[i][1], -iw_load[i]);
                    if (has_h) mov(1, sp[i][3], -ih_load[i]);
                    if (has_d) mov(1, sp[i][5], -id_load[i]);
                }
            }
        }

        if (conf.ver == ver_v1) {
            if (has_d) {
                if (conf.prop_kind == backward_data)
                    mov(1, id_load0, od);
                else
                    mul(1, id_load0, od, uint16_t(conf.stride_d));
                if (conf.f_pad > 0)
                    add(1, id_load0, id_load0, pad_sign * conf.f_pad);
            }

            if (has_h) {
                if (conf.prop_kind == backward_data)
                    mov(1, ih_load0, oh);
                else
                    mul(1, ih_load0, oh, uint16_t(conf.stride_h));
                if (conf.t_pad > 0)
                    add(1, ih_load0, ih_load0, pad_sign * conf.t_pad);
            }

            if (conf.prop_kind == backward_data)
                mov(1, iw_tg, ow_tg);
            else
                mul(1, iw_tg, ow_tg, uint16_t(conf.stride_w));
            if (conf.l_pad > 0) add(1, iw_tg, iw_tg, pad_sign * conf.l_pad);

            if (conf.prop_kind == backward_data) {
                // No need to take is_1st in consideration since there is
                // no such case on bwd.
                add(1, iw_load0, iw_tg, ithr0);
            } else {
                // iw to start reading from global memory.
                if (is_1st) {
                    mad(1, iw_load0, iw_tg, ithr1, uint16_t(conf.stride_w));
                } else {
                    mad(1, iw_load0, iw_tg, ithr0, uint16_t(conf.stride_w));
                }
            }
        }

        // Initialize SLM write offsets for A and B in owords.
        if (is_1st) {
            mul(1, a_slm_off_wr_init, ithr1, uint16_t(a_slm_block_size / 16));
        } else {
            mul(1, a_slm_off_wr_init, ithr0, uint16_t(a_slm_block_size / 16));
        }
        mul(1, b_slm_off_wr_init, ithr0, uint16_t(b_slm_block_size / 16));
        if (is_1st) {
            mad(1, a_slm_off_wr_init, a_slm_off_wr_init, ithr0,
                    uint16_t(mb_read01 * 32 / 16));
            if (conf.oc_group == 4) {
                cmp(1 | eq | f0[0], ithr0, 3);
                add(1 | f0[0], a_slm_off_wr_init, a_slm_off_wr_init,
                        -(mb_read01 - mb_read23) * 32 / 16);
            }
        } else {
            mad(1, a_slm_off_wr_init, a_slm_off_wr_init, ithr1,
                    uint16_t((conf.mb_block / 4) * 32 / 16));
        }
        mad(1, b_slm_off_wr_init, b_slm_off_wr_init, ithr1,
                uint16_t(32 * 32 / 4 / 16));
        add(1, b_slm_off_wr_init, b_slm_off_wr_init, a_slm_size / 16);

        init_src_off();
        init_wei_off();

        switch (conf.ver) {
            case ver_v1: loop_v1(); break;
            case ver_v2: loop_v2(false); break;
            default: assert(!"not expected");
        }

        // Release registers to make them available for destination update.
        ra.safeRelease(A);
        ra.safeRelease(B);
        ra.safeRelease(A_tmp);
        ra.safeRelease(B_tmp);

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
        tmp0 = ra.alloc();
        signal_header = ra.alloc();

        ithr0 = ra.alloc_sub<int32_t>();
        ithr1 = ra.alloc_sub<int32_t>();
        fused_idx = ra.alloc_sub<int32_t>();

        a_slm_rd = ra.alloc_range(a_slm_blocks);
        b_slm_rd = ra.alloc_range(b_slm_blocks);

        for (int i = 0; i < a_slm_blocks; i++)
            a_slm_off_rd_init[i] = ra.alloc_sub<int32_t>();

        auto b_slm_off_rd_init_reg = ra.alloc();
        assert(b_slm_blocks < 8);
        for (int i = 0; i < b_slm_blocks; i++)
            b_slm_off_rd_init[i] = b_slm_off_rd_init_reg.d(i);

        a_slm_off_wr_init = ra.alloc_sub<int32_t>();
        b_slm_off_wr_init = ra.alloc_sub<int32_t>();

        off_tmp = ra.alloc_sub<int32_t>();

        src_off_init0
                = ra.alloc_sub(is_src_off_64_bit ? DataType::q : DataType::d);
        src_off_init[0] = src_off_init0;

        if (conf.ver == ver_v2 && is_4x2_tg) {
            src_off_init1 = ra.alloc_sub(
                    is_src_off_64_bit ? DataType::q : DataType::d);
            src_off_init[1] = src_off_init1;
        }

        wei_off_init
                = ra.alloc_sub(is_wei_off_64_bit ? DataType::q : DataType::d);

        mb = ra.alloc_sub<int32_t>();

        if (do_ic_loop) ic_bytes = ra.alloc_sub<int32_t>();
        oc = ra.alloc_sub<int32_t>();
        oc_fused = ra.alloc_sub<int32_t>();

        if (has_d) {
            od = ra.alloc_sub<int32_t>();
            id = ra.alloc_sub<int32_t>();
        }
        if (has_h) {
            oh = ra.alloc_sub<int32_t>();
            ih = ra.alloc_sub<int32_t>();
        }
        ow = ra.alloc_sub<int32_t>();
        iw = ra.alloc_sub<int32_t>();

        oc_tg = ra.alloc_sub<int32_t>();

        if (conf.ver == ver_v1) {
            ow_tg = ra.alloc_sub<int32_t>();
            iw_tg = ra.alloc_sub<int32_t>();

            slm_idx = ra.alloc().w(0);
            slm_buf_load = slm_idx.w(0);
            slm_buf_compute = slm_idx.w(1);
            slm_counter = slm_idx.d(1);

            src_off_ic = ra.alloc_sub(
                    is_src_off_64_bit ? DataType::q : DataType::d);
            if (has_d) {
                kd = ra.alloc_sub<int32_t>();
                src_off_kd = ra.alloc_sub(
                        is_src_off_64_bit ? DataType::q : DataType::d);
            }
            if (has_h) {
                kh = ra.alloc_sub<int32_t>();
                src_off_kh = ra.alloc_sub(
                        is_src_off_64_bit ? DataType::q : DataType::d);
            }
            if (!is_1st) kw = ra.alloc_sub<int32_t>();
            src_off_kw = ra.alloc_sub(
                    is_src_off_64_bit ? DataType::q : DataType::d);

            if (has_d) id_load0 = id_load[0] = ra.alloc_sub<int32_t>();
            if (has_h) ih_load0 = ih_load[0] = ra.alloc_sub<int32_t>();
            iw_load0 = iw_load[0] = ra.alloc_sub<int32_t>();
        } else {
            // conf.ver == ver_v2.
            _sp = ra.alloc_range(2);
            sp[0] = _sp[0].w();
            sp[1] = _sp[1].w();
            sp_bound = ra.alloc().w();

            for (int i = 0; i < 2; i++) {
                if (has_d) id_load[i] = sp[i][4];
                if (has_h) ih_load[i] = sp[i][2];
                iw_load[i] = sp[i][0];
            }
        }

        int A_tmp_regs = (is_1st ? mb_read01 : conf.mb_block / 4);
        int B_tmp_regs = 8;

        if (conf.oc_group == 2 && !is_1st) A_tmp_regs *= 2;

        A_tmp = ra.alloc_range(A_tmp_regs);
        B_tmp = ra.alloc_range(B_tmp_regs);

        if (is_1st) {
            assert(A.getLen() >= mb_read01);
            A_tmp_reorder = A;
        }
    }

    // i0 = x % n0
    // i1 = (x / n0) % n1
    // i2 = (x / n0) / n1
    void unpack_1d_to_3d(const Subregister &x, const Subregister &i0, int n0,
            const Subregister &i1, int n1, const Subregister &i2, int n2) {

        // 1D
        if (n1 == 1 && n2 == 1) {
            mov(1, i0, x);
            return;
        }

        // 2D
        if (n2 == 1) {
            e_idiv(i1, i0, x, n0);
            return;
        }

        auto i12 = ra.alloc_sub<int32_t>();

        e_idiv(i12, i0, x, n0);
        e_idiv(i2, i1, i12, n1);

        ra.safeRelease(i12);
    }

    void loop_v1() {
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

        int src_header_regs = 1;
        if (is_1st) {
            src_header_regs = 3;
        } else if (mb_block > 32) {
            src_header_regs = 2;
        }

        src_header = ra.alloc_range(src_header_regs);
        wei_header = ra.alloc_range(1);

        a_slm_wr = ra.alloc_range(src_header_regs);
        b_slm_wr = ra.alloc_range(1);

        // Set C to zero.
        for (int i = 0; i < C.getLen(); i += 2)
            mov<float>(16, C[i], 0.0f);

        mov(1, src_off_ic,
                is_src_off_64_bit ? src_off_init0 : src_off_init0.d(0));
        add(1, wei_header[0].uq(0), wei_off_init, wei_ptr);

        // To complete all OOO writes.
        sync(SyncFunction::allwr);

        // Disable auto-SWSB for the inner loop for better control over SBID-s.
        disable_auto_swsb();

        if (slm_nbuf == 3) fence_and_signal(/*skip_fence=*/true);

        sync(SyncFunction::nop, SWSB<AllPipes>(1));

        if (do_ic_loop) mov(1, ic_bytes, 0);

        // Dependency for src_off* updates.
        auto src_off_dep
                = is_src_off_64_bit ? SWSB<int64_t>(1) : SWSB<int32_t>(3);

        // Reduction loop.
        Label ic_loop;
        if (do_ic_loop) mark(ic_loop);

        Label kd_loop;
        Label kd_skip;
        if (has_d) {
            mov(1 | src_off_dep, src_off_kd, src_off_ic);
            mov(1, kd, 0);
            mov(1, id, id_load0);

            mark(kd_loop);

            // Check padding for ID.
            if (has_pad_d) {
                cmp(1 | lt | f1[0] | SWSB(1), id, conf.id);
                cmp(1 | f1[0] | ge, id, 0);
                if (is_1st) {
                    // OIx8o2i or OIx8o4i.
                    add(1 | ~f1[0], wei_header[0].uq(0), wei_header[0].uq(0),
                            conf.kh * conf.kw * 32);
                } else {
                    // OIx4o8i8o2i or OIx4o8i8o4i.
                    add(1 | ~f1[0], wei_header[0].uq(0), wei_header[0].uq(0),
                            conf.kh * conf.kw * 32 * 32);
                }
                jmpi(1 | ~f1[0], kd_skip);
            }
        }

        Label kh_loop;
        Label kh_skip;
        if (has_h) {
            mov(1 | src_off_dep, src_off_kh, has_d ? src_off_kd : src_off_ic);
            mov(1, kh, 0);
            mov(1, ih, ih_load0);

            mark(kh_loop);

            // Check padding for IH.
            if (has_pad_h) {
                cmp(1 | lt | f1[0] | SWSB(1), ih, conf.ih);
                cmp(1 | f1[0] | ge, ih, 0);
                if (is_1st) {
                    // OIx8o2i or OIx8o4i.
                    add(1 | ~f1[0], wei_header[0].uq(0), wei_header[0].uq(0),
                            conf.kw * 32);
                } else {
                    // OIx4o8i8o2i or OIx4o8i8o4i.
                    add(1 | ~f1[0], wei_header[0].uq(0), wei_header[0].uq(0),
                            conf.kw * 32 * 32);
                }
                jmpi(1 | ~f1[0], kh_skip);
            }
        }

        mov(1 | src_off_dep, src_off_kw,
                has_h ? src_off_kh : has_d ? src_off_kd : src_off_ic);

        Label kw_loop;
        if (do_kw_loop) {
            mov(1, kw, 0);
            mov(1, iw, iw_load0);

            mark(kw_loop);
        }

        if (slm_nbuf == 2) {
            fence_and_signal();
            wait();
        }
        gmem2reg();
        multiply();
        reg2smem();

        slm_buffer_advance();

        if (do_kw_loop) {
            // Advance kw = kw + 1.
            add(1, src_off_kw, src_off_kw,
                    -pad_sign * (1 + conf.dilate_w) * mb_block * 32);
            add(1, kw, kw, 1);
            add(1, iw, iw, -pad_sign * (1 + conf.dilate_w));
            cmp(8 | lt | f0[0] | SWSB(2), kw, conf.kw);
            while_(8 | f0[0], kw_loop);
        }

        if (has_h) {
            mark(kh_skip);

            // Advance kh = kh + 1.
            add(1, kh, kh, 1);
            add(1, ih, ih, -pad_sign * (1 + conf.dilate_h));
            if (is_1st) {
                // NCx4n2c or NCx4n4c.
                add(1, src_off_kh, src_off_kh,
                        (1 + conf.dilate_h) * conf.iw * 4 * 4);
            } else {
                // NCx<mb_block>n16c or NCx<mb_block>n32c.
                add(1, src_off_kh, src_off_kh,
                        -pad_sign * (1 + conf.dilate_h) * conf.iw * mb_block
                                * 32);
            }
            cmp(8 | lt | f0[0] | SWSB(2), kh, conf.kh);
            while_(8 | f0[0], kh_loop);
        }

        if (has_d) {
            mark(kd_skip);

            // Advance kd = kd + 1.
            add(1, kd, kd, 1);
            add(1, id, id, -pad_sign * (1 + conf.dilate_d));
            if (is_1st) {
                // NCx4n2c or NCx4n4c.
                add(1, src_off_kd, src_off_kd,
                        (1 + conf.dilate_d) * conf.ih * conf.iw * 4 * 4);
            } else {
                // NCx<mb_block>n16c or NCx<mb_block>n32c.
                add(1, src_off_kd, src_off_kd,
                        -pad_sign * (1 + conf.dilate_d) * conf.ih * conf.iw
                                * mb_block * 32);
            }
            cmp(8 | lt | f0[0] | SWSB(2), kd, conf.kd);
            while_(8 | f0[0], kd_loop);
        }

        if (do_ic_loop) {
            // Advance ic_bytes.
            add(1, ic_bytes, ic_bytes, conf.ic_block * src_size);
            if (is_1st) {
                // NCx4n2c or NCx4n4c.
                add(1, src_off_ic, src_off_ic,
                        conf.id * conf.ih * conf.iw * 4 * 4);
            } else {
                // NCx<mb_block>n16c or NCx<mb_block>n32c.
                add(1, src_off_ic, src_off_ic,
                        conf.id * conf.ih * conf.iw * mb_block * 32);
            }
            cmp(8 | lt | f0[0] | SWSB(1), ic_bytes, ic_bytes_padded);
            while_(8 | f0[0], ic_loop);
        }

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

        // Re-enable auto-SWSB back.
        enable_auto_swsb();

        // To ensure all DPASW calls updated their results.
        sync(SyncFunction::allwr);

        ra.safeRelease(src_header);
        ra.safeRelease(wei_header);

        ra.safeRelease(a_slm_wr);
        ra.safeRelease(b_slm_wr);
    }

    void loop_v2(bool skip_check) {
        int regs_per_read = 8;

        int src_regs_per_thr = mb_block / conf.sp_group;
        int wei_regs_per_thr = 32 / conf.sp_group;

        int src_blocks = utils::div_up(src_regs_per_thr, regs_per_read);
        if (is_4x2_tg) src_blocks *= 2;

        int wei_blocks = utils::div_up(wei_regs_per_thr, regs_per_read);

        src_header = ra.alloc_range(src_blocks);
        wei_header = ra.alloc_range(wei_blocks);

        a_slm_wr = ra.alloc_range(src_blocks);
        b_slm_wr = ra.alloc_range(wei_blocks);

        for (int i = 0, idx = 0; i < (is_4x2_tg ? 2 : 1); i++) {
            for (int j = 0; j < src_regs_per_thr; j += regs_per_read) {
                add(1, src_header[idx].uq(0), src_ptr, src_off_init[i]);
                idx++;
            }
        }
        for (int i = 0; i < wei_header.getLen(); i++)
            add(1, wei_header[i].uq(0), wei_ptr, wei_off_init);

        // Set up source offsets for global reads and SLM writes.
        {
            int idx = 0;
            for (int i = 0; i < (is_4x2_tg ? 2 : 1); i++) {
                for (int j = 0; j < src_regs_per_thr; j += regs_per_read) {
                    int off = j * 32;
                    add(1, src_header[idx].uq(0), src_header[idx].uq(0), off);

                    int slm_off = j * 32;
                    if (i == 1) slm_off += 2 * a_slm_block_size;
                    add(1, a_slm_wr[idx].ud(2), a_slm_off_wr_init,
                            slm_off / 16);

                    idx++;
                }
            }
        }

        // Set up weights offsets for global reads and SLM writes.
        {
            int idx = 0;
            for (int j = 0; j < wei_regs_per_thr; j += regs_per_read) {
                int off = j * 32;
                add(1, wei_header[idx].uq(0), wei_header[idx].uq(0), off);
                int slm_off = off;
                add(1, b_slm_wr[idx].ud(2), b_slm_off_wr_init, slm_off / 16);
                idx++;
            }
        }

        // Set up offsets for SLM reads.
        for (int i = 0; i < a_slm_blocks; i++)
            mov(1, a_slm_rd[i].ud(2), a_slm_off_rd_init[i]);

        for (int i = 0; i < b_slm_blocks; i++)
            mov(1, b_slm_rd[i].ud(2), b_slm_off_rd_init[i]);

        if (check_src_load) {
            for (int i = 0; i < (is_4x2_tg ? 2 : 1); i++) {
                sp_check_flags[i] = f1[i];
                mov(1, sp_check_flags[i], uint16_t(0xFFFF));
            }
        }

        // To complete all OOO writes.
        sync(SyncFunction::allwr);

        sync(SyncFunction::nop, SWSB<AllPipes>(1));

        // Disable auto-SWSB for the inner loop for better control over SBID-s.
        disable_auto_swsb();

        // Fully unroll (kd * kh * kw) and also ensure we can hard-code SLM
        // offsets.
        int unroll = slm_nbuf * kdhw / math::gcd(slm_nbuf, kdhw);
        loop_iterator_t it(conf, unroll, slm_nbuf, ab_slm_size, check_src_load);

        // Ramp-up.
        for (int i = 0; i < it.ramp_up_iters; i++) {
            loop_iterate_v2(it);
            ++it;
        }

        auto iter = ra.alloc_sub<int32_t>();

        // Body.
        if (it.body_iters > 0) {
            mov(1, iter, it.body_iters);

            Label ic_loop;
            mark(ic_loop);

            for (int i = 0; i < it.unroll; i++) {
                loop_iterate_v2(it);
                ++it;
            }

            add(1 | gt | f0[0], iter, iter, -it.unroll);
            jmpi(1 | f0[0], ic_loop);

            it.advance(it.body_iters - it.unroll);
        }

        ra.safeRelease(iter);

        // Ramp-down.
        for (int i = 0; i < it.ramp_down_iters; i++) {
            loop_iterate_v2(it);
            ++it;
        }

        // Re-enable auto-SWSB back.
        enable_auto_swsb();

        // To ensure all DPASW calls updated their results.
        sync(SyncFunction::allwr);

        ra.safeRelease(src_header);
        ra.safeRelease(wei_header);

        ra.safeRelease(a_slm_wr);
        ra.safeRelease(b_slm_wr);
    }

    void loop_iterate_v2(const loop_iterator_t &it) {
        bool do_gmem2reg = it.do_gmem2reg();
        bool do_multiply = it.do_multiply();

        if (slm_nbuf == 3 && do_multiply) wait();
        if (do_gmem2reg) gmem2reg_v2(it);

        if (slm_nbuf == 3 && it.iter == 1) fence_and_signal();

        if (do_multiply) {

            // Wait on DPASW read to be able to write to A and B.
            auto sync_mask = to_sbid_mask({dpasw_sbid(0), dpasw_sbid(1),
                    dpasw_sbid(2), dpasw_sbid(3)});
            sync(SyncFunction::allrd, sync_mask);

            smem2reg_v2(it);
            multiply_v2(it);
        }

        if (do_gmem2reg) {
            reg2smem_v2(it);
            if (slm_nbuf == 2) {
                fence_and_signal();
                wait();
            }
        }
    }

    // SBID usage:
    //   $0-3       A load from global memory and store to SLM
    //   $4         B load from global memory and store to SLM
    //   $8-10      A SLM loads
    //   $11-14     DPASW and B SLM loads
    //   $15        Barrier/SLM fence
    SBID gmem_a_sbid(int iter, int reg) const {
        assert(reg < 10);
        if (reg < 8) return SBID(iter);
        return SBID(2 + iter);
    }

    SBID gmem_a_sbid(int iter) const { return SBID(iter % 4); }

    SBID gmem_b_sbid() const { return sb4; }

    SBID smem_a_sbid(int idx) const { return SBID(8 + idx); }

    SBID dpasw_sbid(int idx) const {
        assert(idx < 4);
        return SBID(11 + idx);
    }

    uint32_t to_sbid_mask(std::initializer_list<SBID> sbids) const {
        uint32_t mask = 0;
        for (auto &sb : sbids)
            mask |= (1 << sb.set.getID());
        return mask;
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
            const Subregister &x, int y) {

        assert(is_auto_swsb);
        assert(x.getType() == DataType::d);
        assert(utils::one_of(qot.getType(), DataType::d, DataType::w));
        assert(utils::one_of(rem.getType(), DataType::d, DataType::w));
        assert(y >= 1);

        if (y == 1) {
            mov(1, qot, x);
            mov(1, rem, 0);
            return;
        }

        auto tmp = ra.alloc_sub<int32_t>();

        auto qot_by_y = tmp;
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

        ra.safeRelease(tmp);
    }

    void e_mul(const Subregister &dst, const Subregister &src1, int src2) {
        auto tmp = ra.alloc_sub<int64_t>();

        assert(is_auto_swsb);

        bool dst_is_64_bit
                = utils::one_of(dst.getType(), DataType::q, DataType::uq);

        if (dst_is_64_bit) {
            mul(1, dst, src1, src2);
            return;
        }

        if (src2 >= std::numeric_limits<int16_t>::min()
                && src2 <= std::numeric_limits<uint16_t>::max()) {
            mul(1, dst, src1, src2);
            return;
        }

        mul(1, tmp.q(0), src1, src2);
        mov(1, dst, tmp.reinterpret(0, dst.getType()));

        ra.safeRelease(tmp);
    }

    void e_mad(const Subregister &dst, const Subregister &src1, int src2) {
        assert(is_auto_swsb);

        bool dst_is_32_bit
                = utils::one_of(dst.getType(), DataType::d, DataType::ud);
        if (dst_is_32_bit) {
            if (src2 >= std::numeric_limits<int16_t>::min()
                    && src2 <= std::numeric_limits<uint16_t>::max()) {
                mad(1, dst, dst, src1, src2);
                return;
            }
        }
        mul(1, tmp0.q(0), src1, src2);
        add(1, dst, dst, tmp0.retype(dst.getType()));
    }

    void init_src_off() {
        init_src_off_tg();
        init_src_off_thr();
    }

    void init_wei_off() {
        init_wei_off_tg();
        init_wei_off_thr();
    }

    void init_src_off_tg() {
        if (is_1st) {
            init_src_off_tg_4n4c();
        } else {
            init_src_off_tg_Xn32c();
        }
    }

    // Initializes src offset in bytes for TG for 4n4c (4n2c) layout.
    void init_src_off_tg_4n4c() {
        assert(conf.ver == ver_v1);

        // (mb / 4) * (IC / 4) * ID * IH * IW * 4 * 4
        e_mul(src_off_init0, mb, conf.id * conf.ih * conf.iw * ic_bytes_padded);

        if (has_d) {
            // id * IH * IW * 4 * 4
            e_mad(src_off_init0, id_load0, conf.ih * conf.iw * 4 * 4);
        }

        if (has_h) {
            // ih * IW * 4 * 4
            e_mad(src_off_init0, ih_load0, conf.iw * 4 * 4);
        }

        // iw * 4 * 4
        e_mad(src_off_init0, iw_load0, 4 * 4);
    }

    // Initializes src offset in bytes for TG for Xn32c (Xn16c) layout.
    void init_src_off_tg_Xn32c() {
        // (mb / MB_BLOCK) * (IC / 32) * ID * IH * IW * MB_BLOCK * 32
        e_mul(src_off_init0, mb, conf.id * conf.ih * conf.iw * ic_bytes_padded);

        if (conf.ver == ver_v2 && is_4x2_tg)
            mov(1, src_off_init1, src_off_init0);

        for (int i = 0; i < (conf.ver == ver_v2 && is_4x2_tg ? 2 : 1); i++) {
            if (has_d) {
                // id * IH * IW * MB_BLOCK * 32
                e_mad(src_off_init[i], id_load[i],
                        conf.ih * conf.iw * mb_block * 32);
            }

            if (has_h) {
                // ih * IW * MB_BLOCK * 32
                e_mad(src_off_init[i], ih_load[i], conf.iw * mb_block * 32);
            }

            // iw * MB_BLOCK * 32
            e_mad(src_off_init[i], iw_load[i], mb_block * 32);
        }
    }

    void init_src_off_thr() {
        // Initialize thread read offsets for source.
        if (is_1st) {
            assert(conf.ver == ver_v1);
            mul(1, tmp0.uq(0), ithr0,
                    mb_read01 * conf.id * conf.ih * conf.iw * 4 * src_size);
            if (conf.oc_group == 4) {
                cmp(1 | eq | f0[0], ithr0, 3);
                add(1 | f0[0], tmp0.uq(0), tmp0.uq(0),
                        -(mb_read01 - mb_read23) * conf.id * conf.ih * conf.iw
                                * 4 * src_size);
            }

            add(1, src_off_init0, src_off_init0, tmp0.d(0));
        } else {
            for (int i = 0; i < (conf.ver == ver_v2 && is_4x2_tg ? 2 : 1);
                    i++) {
                e_mad(src_off_init[i], ithr1, mb_block * 32 / 4);
            }
        }
    }

    void init_wei_off_tg() {
        // OIx4o8i8o4i: (oc / 32) * (IC / 32) * KH * KW * 32 * 32
        // OIx8o4i:     (oc / 8) * (IC / 4) * KH * KW * 8 * 4
        e_mul(wei_off_init, oc_tg,
                ic_bytes_padded * conf.kd * conf.kh * conf.kw);
    }

    void init_wei_off_thr() {
        // Initialize thread read offset for weights.
        if (is_1st) {
            // OIx8o2i or OIx8o4i.
            e_mad(wei_off_init, ithr0,
                    conf.kd * conf.kh * conf.kw * 32 * 4 * wei_size);
            e_mad(wei_off_init, ithr1,
                    conf.kd * conf.kh * conf.kw * 8 * 4 * wei_size);
        } else {
            // OIx4o8i8o2i or OIx4o8i8o4i.
            e_mad(wei_off_init, ithr0,
                    ic_bytes_padded * conf.kd * conf.kh * conf.kw * 32);
            e_mad(wei_off_init, ithr1, 256);
        }
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

        int A_block_regs = mb_block / 4;
        assert(A_block_regs >= 8);

        auto src_off_dep = is_src_off_64_bit
                ? SWSB<int64_t>(1)
                : SWSB<int32_t>(do_kw_loop ? 3 : 1);

        for (int i = 0; i < src_header.getLen(); i++) {
            InstructionModifier mod
                    = (i == 0) ? src_off_dep : InstructionModifier();
            add(1 | mod, src_header[i].uq(0), src_ptr, src_off_kw);
        }

        gmem2reg_weights();

        if (is_1st) {
            gmem2reg_source_1st();
        } else {
            gmem2reg_source(A_block_regs);
        }
    }

    // Loads source from global memory.
    void gmem2reg_source(int A_block_regs) {
        int src_load_iters = (conf.oc_group == 4 ? 1 : 2);
        if (A_block_regs > 8)
            add(1, src_header[1].uq(0), src_header[1].uq(0), 256);
        for (int iter = 0; iter < src_load_iters; iter++) {
            int ithr = iter * conf.oc_group;
            Label src_skip, src_end;
            // TODO: No padding and non-multiple OW case does not require >= 0
            // check.
            if (check_src_load) {
                // FWD:
                // - iw + ithr * SW < IW
                // - iw + ithr * SW >= 0
                // BWD:
                // - (iw + ithr) / SW < IW
                // - (iw + ithr) / SW >= 0
                if (conf.prop_kind == backward_data) {
                    cmp(8 | lt | f1[0], iw, conf.iw * conf.stride_w - ithr);
                    cmp(8 | f1[0] | ge, iw, -ithr);
                } else {
                    cmp(8 | lt | f1[0], iw, conf.iw - ithr * conf.stride_w);
                    cmp(8 | f1[0] | ge, iw, -ithr * conf.stride_w);
                }
                if_(8 | f1[0], src_skip, src_end);
            }
            load(16 | SWSB(gmem_a_sbid(iter, 0), 1), A_tmp[iter * A_block_regs],
                    block_hword(8), A64, src_header[0]);
            if (A_block_regs > 8) {
                load(16 | SWSB(gmem_a_sbid(iter, 8), 1),
                        A_tmp[iter * A_block_regs + 8],
                        block_hword(A_block_regs - 8), A64, src_header[1]);
            }
            if (check_src_load) {
                else_(8, src_end, src_end);
                mark(src_skip);
                for (int i = 0; i < A_block_regs; i += 2) {
                    mov<float>(16 | gmem_a_sbid(iter, i).src,
                            A_tmp[iter * A_block_regs + i], 0.0f);
                }
                mark(src_end);
                endif(8);
            }

            if (iter + 1 < src_load_iters) {
                add(1 | gmem_a_sbid(iter, 0).src, src_header[0].uq(0),
                        src_header[0].uq(0), 2 * conf.stride_w * mb_block * 32);
                if (A_block_regs > 8)
                    add(1 | gmem_a_sbid(iter, 8).src, src_header[1].uq(0),
                            src_header[1].uq(0),
                            2 * conf.stride_w * mb_block * 32);
            }
        }
    }

    // Loads source from global memory (for 1st convolution).
    void gmem2reg_source_1st() {
        assert(is_1st);

        // Wait for DPASW reads to use A registers (for A_tmp_reorder).
        auto sync_mask = to_sbid_mask(
                {dpasw_sbid(0), dpasw_sbid(1), dpasw_sbid(2), dpasw_sbid(3)});
        sync(SyncFunction::allrd, sync_mask);

        Label end;
        Label mb_tail_label;

        // Handle different MB read lengths.
        bool has_mb_tail = (mb_read23 != mb_read01);
        if (has_mb_tail) {
            cmp(8 | ge | f0[0], ithr0, 2);
            jmpi(1 | f0[0], mb_tail_label);
        }

        // MB cases: first EU pair (mb_read01) and second EU pair (mb_read23).
        for (int mb_case = 0; mb_case < (has_mb_tail ? 2 : 1); mb_case++) {
            int mb_len = (mb_case == 0 ? mb_read01 : mb_read23);
            if (mb_case == 1) mark(mb_tail_label);

            // Handle cases for different W read lengths.
            nw_read_region_t *no_w_pad = nullptr;
            for (auto &r : nw_read_regions) {
                if (r.mb_len != mb_len) continue;
                if (!r.with_w_padding()) {
                    no_w_pad = &r;
                    continue;
                }
                cmp(1 | eq | f0[0], iw_load0, r.w_val);
                jmpi(1 | f0[0], r.label);
            }

            // No padding case.
            no_w_pad->read_and_reorder();

            jmpi(1, end);

            // Cases with padding.
            for (auto &r : nw_read_regions) {
                if (r.mb_len != mb_len) continue;
                if (!r.with_w_padding()) continue;
                mark(r.label);
                r.read_and_reorder();
                jmpi(1, end);
            }
        }

        mark(end);
    }

    // Loads weights from global memory.
    void gmem2reg_weights() {
        Label wei_skip;
        // TODO: move condition out of the loop.
        if (oc_padded != oc_tg_padded) {
            cmp(8 | lt | f0[1], oc, oc_padded);
            if_(8 | f0[1], wei_skip, wei_skip);
        }
        {
            // TODO: Fix out-of-bounds access for the 1st convolution:
            // - KW = 7 but always reading 8 hwords
            // - When (conf.oc % 32) != 0: some threads read out-of-bounds
            load(16 | SWSB(gmem_b_sbid(), 1), B_tmp[0], block_hword(8), A64,
                    wei_header[0]);
            if (is_1st) {
                add(1 | gmem_b_sbid().src, wei_header[0].uq(0),
                        wei_header[0].uq(0), conf.kw * 32);
            } else {
                add(1 | gmem_b_sbid().src, wei_header[0].uq(0),
                        wei_header[0].uq(0), 32 * 32);
            }
        }
        if (oc_padded != oc_tg_padded) {
            mark(wei_skip);
            endif(8);
        }
    }

    void gmem2reg_v2(const loop_iterator_t &it) {
        if (!enable_gmem_read) return;

        int src_update = it.gmem_read_src_off_update();
        int wei_update = 32 * 32;

        int idx;
        int reg_off;

        int regs_per_read = 8;

        // Load source.
        int src_regs_per_thr = mb_block / conf.sp_group;

        if (it.check_src_load) {
            for (int i = 0; i < (is_4x2_tg ? 2 : 1); i++) {
                cmp(8 | le | sp_check_flags[i], sp[i], sp_bound);
            }
        }

        idx = 0;
        reg_off = 0;
        for (int i = 0; i < (is_4x2_tg ? 2 : 1); i++) {
            auto sp_flag = sp_check_flags[i];
            for (int j = 0; j < src_regs_per_thr; j += regs_per_read) {
                int hwords = std::min(8, src_regs_per_thr - j);

                assert(reg_off + hwords <= A_tmp.getLen());

                auto sbid = gmem_a_sbid(idx);
                InstructionModifier mod = SWSB(sbid, 2);
                if (it.check_src_load) {
                    mod |= sp_flag;
                    mod |= all16h;
                }
                load(16 | mod, A_tmp[reg_off], block_hword(hwords), A64,
                        src_header[idx]);
                if (it.check_src_load) {
                    for (int k = 0; k < hwords; k += 2) {
                        mov(16 | ~sp_flag | all16h, A_tmp[reg_off + k].f(),
                                0.0f);
                    }
                }
                add(1 | sbid.src, src_header[idx].uq(0), src_header[idx].uq(0),
                        src_update);

                idx++;
                reg_off += hwords;
            }
        }

        if (it.check_src_load) {
            int iw_upd = it.iw_update();
            int ih_upd = it.ih_update();
            int id_upd = it.id_update();
            for (int i = 0; i < (is_4x2_tg ? 2 : 1); i++) {
                add(8 | SWSB(1), sp[i], sp[i],
                        Immediate::v(iw_upd, -iw_upd, ih_upd, -ih_upd, id_upd,
                                -id_upd, 0, 0));
            }
        }

        // Load weights.
        int wei_regs_per_thr = 32 / conf.sp_group;

        idx = 0;
        reg_off = 0;
        for (int j = 0; j < wei_regs_per_thr; j += regs_per_read) {
            auto sbid = gmem_b_sbid();
            load(16 | SWSB(sbid, 2), B_tmp[reg_off], block_hword(8), A64,
                    wei_header[idx]);
            add(1 | sbid.src, wei_header[idx].uq(0), wei_header[idx].uq(0),
                    wei_update);

            idx++;
            reg_off += 8;
        }
    }

    void reg2smem_v2(const loop_iterator_t &it) {
        if (!enable_smem_write) return;

        int upd = it.smem_write_off_update();

        int idx;
        int reg_off;

        int regs_per_read = 8;

        // Store source.
        int src_regs_per_thr = mb_block / conf.sp_group;

        idx = 0;
        reg_off = 0;
        for (int i = 0; i < (is_4x2_tg ? 2 : 1); i++) {
            for (int j = 0; j < src_regs_per_thr; j += regs_per_read) {
                int owords = std::min(8, src_regs_per_thr - j) * 2;

                assert(reg_off + owords / 2 <= A_tmp.getLen());

                auto sbid = gmem_a_sbid(idx);
                auto mod = it.check_src_load ? SWSB(sbid, 1) : SWSB(sbid);
                store(16 | mod, block_oword(owords), SLM, a_slm_wr[idx],
                        A_tmp[reg_off]);
                add(1 | sbid.src, a_slm_wr[idx].ud(2), a_slm_wr[idx].ud(2),
                        upd / 16);

                idx++;
                reg_off += owords / 2;
            }
        }

        // Store weights.
        int wei_regs_per_thr = 32 / conf.sp_group;

        idx = 0;
        reg_off = 0;
        for (int j = 0; j < wei_regs_per_thr; j += regs_per_read) {
            auto sbid = gmem_b_sbid();
            store(16 | SWSB(sbid), block_oword(16), SLM, b_slm_wr[idx],
                    B_tmp[reg_off]);
            add(1 | sbid.src, b_slm_wr[idx].ud(2), b_slm_wr[idx].ud(2),
                    upd / 16);

            idx++;
            reg_off += 8;
        }
    }

    void smem2reg_v2(const loop_iterator_t &it) {
        bool is_last = it.is_last_multiply();
        int upd = it.smem_read_off_update();
        int dist = a_slm_blocks + b_slm_blocks;

        // Load A.
        for (int i = 0; i < mb_block / 2; i += 8) {
            int idx = i / 8;
            int owords = std::min(8, mb_block / 2 - i) * 2;
            auto sbid = smem_a_sbid(idx);
            auto mod = it.do_gmem2reg() ? SWSB(sbid)
                                        : SWSB(sbid, std::min(7, dist--));
            load(16 | mod, A[i], block_oword(owords), SLM, a_slm_rd[idx]);
            if (!is_last)
                add(1 | sbid.src, a_slm_rd[idx].ud(2), a_slm_rd[idx].ud(2),
                        upd / 16);
        }

        // Load B.
        for (int i = 0; i < B.getLen(); i += 8) {
            int idx = i / 8;
            auto sbid = dpasw_sbid(idx);
            auto mod = it.do_gmem2reg() ? SWSB(sbid)
                                        : SWSB(sbid, std::min(7, dist--));
            load(16 | mod, B[i], block_oword(16), SLM, b_slm_rd[idx]);
            if (!is_last)
                add(1 | sbid.src, b_slm_rd[idx].ud(2), b_slm_rd[idx].ud(2),
                        upd / 16);
        }
    }

    void multiply_v2(const loop_iterator_t &it) {
        bool is_first = it.is_first_multiply();
        bool is_last = it.is_last_multiply();

        // Multiply C = A * B.
        for (int oc = 0, idx = 0; oc < 32; oc += 8, idx++) {
            if (oc == 0) {
                // Wait A.
                uint32_t sync_mask = to_sbid_mask(
                        {smem_a_sbid(0), smem_a_sbid(1), smem_a_sbid(2)});
                sync(SyncFunction::allwr, sync_mask);
            }
            // Compute [0:mb_block, oc:oc + 8] block of C.
            multiply_chunk(idx, B[oc % B.getLen()], C[idx * mb_block],
                    /*wait_a=*/false, is_first);

            // Reuse B[oc] for later multiplies (case of oc_group == 2).
            if (reuse_b_regs && oc < B.getLen()) {
                auto sbid = dpasw_sbid(idx);
                int upd = it.smem_read_off_update();
                load(16 | SWSB(sbid), B[oc], block_oword(16), SLM,
                        b_slm_rd[idx + 2]);
                if (!is_last) {
                    add(1 | sbid.src, b_slm_rd[idx + 2].ud(2),
                            b_slm_rd[idx + 2].ud(2), upd / 16);
                }
            }
        }

        if (slm_nbuf == 3 && !is_last) fence_and_signal(/*skip_fence=*/true);
    }

    void reg2smem_A() {
        int A_block_regs = mb_block / 4;
        int src_load_iters = (conf.oc_group == 4 ? 1 : 2);

        mad(1, a_slm_wr[0].d(2), a_slm_off_wr_init, slm_buf_load,
                uint16_t(ab_slm_size / 16));
        if (A_block_regs > 8) {
            mad(1, a_slm_wr[1].d(2), a_slm_off_wr_init, slm_buf_load,
                    uint16_t(ab_slm_size / 16));
            add(1 | SWSB(1), a_slm_wr[1].d(2), a_slm_wr[1].d(2), uint16_t(16));
        }

        for (int iter = 0; iter < src_load_iters; iter++) {
            auto sb_store0 = gmem_a_sbid(iter, 0);
            auto sb_store1 = gmem_a_sbid(iter, 8);
            store(16 | SWSB(sb_store0, 1), block_oword(16), SLM, a_slm_wr[0],
                    A_tmp[iter * A_block_regs]);
            if (A_block_regs > 8) {
                store(16 | SWSB(sb_store1, iter == 0 ? 2 : 1),
                        block_oword((A_block_regs - 8) * 2), SLM, a_slm_wr[1],
                        A_tmp[iter * A_block_regs + 8]);
            }
            if (iter + 1 < src_load_iters) {
                add(1 | sb_store0.src, a_slm_wr[0].d(2), a_slm_wr[0].d(2),
                        uint16_t(a_slm_block_size * 2 / 16));
                if (A_block_regs > 8) {
                    add(1 | sb_store1.src, a_slm_wr[1].d(2), a_slm_wr[1].d(2),
                            uint16_t(a_slm_block_size * 2 / 16));
                }
            }
        }
    }

    void reg2smem_A_1st() {
        int A_block_regs = mb_read01;

        assert(a_slm_wr.getLen() >= 2);
        mad(1, a_slm_wr[0].d(2), a_slm_off_wr_init, slm_buf_load,
                uint16_t(ab_slm_size / 16));
        if (A_block_regs > 8) {
            mad(1, a_slm_wr[1].d(2), a_slm_off_wr_init, slm_buf_load,
                    uint16_t(ab_slm_size / 16));
            add(1 | SWSB(1), a_slm_wr[1].d(2), a_slm_wr[1].d(2), uint16_t(16));
        }

        Label end;
        Label mb_tail_label;

        bool has_mb_tail = (mb_read23 != mb_read01);
        if (has_mb_tail) {
            cmp(8 | ge | f0[0], ithr0, 2);
            jmpi(1 | f0[0], mb_tail_label);
        }

        for (int mb_case = 0; mb_case < (has_mb_tail ? 2 : 1); mb_case++) {
            int mb_len = (mb_case == 0 ? mb_read01 : mb_read23);
            if (mb_case == 1) mark(mb_tail_label);

            for (int i = 0; i < mb_len; i += 8) {
                int idx = i / 8;
                int owords = std::min(mb_len - i, 8) * 2;
                auto sbid = gmem_a_sbid(idx);
                store(16 | SWSB(sbid, (i == 0 && A_block_regs > 8) ? 3 : 1),
                        block_oword(owords), SLM, a_slm_wr[idx % 2], A_tmp[i]);
                if (i + 16 < mb_len) {
                    add(1 | sbid.src, a_slm_wr[idx % 2].d(2),
                            a_slm_wr[idx % 2].d(2), uint16_t(32));
                }
            }

            if (has_mb_tail) jmpi(1, end);
        }

        mark(end);
    }

    void reg2smem() {
        if (!enable_smem_write) return;
        assert(!is_auto_swsb);

        mad(1, b_slm_wr[0].d(2), b_slm_off_wr_init, slm_buf_load,
                uint16_t(ab_slm_size / 16));

        if (is_1st) {
            // Set 8o4i block to zero for kw = 7.
            mov<float>(8 | gmem_b_sbid().dst, B_tmp[7], 0.0f);
        }

        if (is_1st) {
            reg2smem_A_1st();
        } else {
            reg2smem_A();
        }

        store(16 | SWSB(gmem_b_sbid(), 1), block_oword(16), SLM, b_slm_wr[0],
                B_tmp[0]);
    }

    void fence_and_signal(bool skip_fence = false) {
        if (!enable_barrier) return;
        assert(!is_auto_swsb);

        if (!skip_fence) {
            slmfence(sb15, tmp0, r0);
            mov<int32_t>(8 | sb15.dst, null, tmp0);
        }
        barriermsg(SWSB(sb15), signal_header);
    }

    void wait() {
        if (!enable_barrier) return;
        barrierwait();
    }

    void dpasw_typed(const InstructionModifier &mod, uint8_t sdepth,
            uint8_t rcount, const GRF &dst, const Register &src0,
            const GRF &src1, const GRF &src2) {
        if (!enable_dpasw) return;

        dpasw(mod, sdepth, rcount, dst.retype(acc_type), src0.retype(acc_type),
                src1.retype(wei_type), src2.retype(src_type));
    }

    void multiply_chunk(int idx, const GRF &B_reg, const GRF &C_reg,
            bool wait_a, bool is_first = false) {
        if (wait_a) {
            uint32_t sync_mask = to_sbid_mask(
                    {smem_a_sbid(0), smem_a_sbid(1), smem_a_sbid(2)});
            sync(SyncFunction::allwr, sync_mask);
        }

        auto sb = dpasw_sbid(idx);
        for (int i = 0; i < mb_block; i += 8) {
            InstructionModifier mod = 8;
            if (i == 0) {
                if (!reuse_b_regs || idx < 2) {
                    mod |= sb.dst;
                } else {
                    mod |= dpasw_sbid(idx - 2).dst;
                }
            }
            if (i + 8 != mb_block) {
                mod |= Atomic;
            } else {
                mod |= sb;
            }
            auto dst = GRF(C_reg.getBase() + i);
            auto src1 = B_reg;
            auto src2 = A[i / 2];
            if (is_first) {
                // dst is not initialized, use null as src0.
                dpasw_typed(mod, 8, 8, dst, null, src1, src2);
            } else {
                dpasw_typed(mod, 8, 8, dst, dst, src1, src2);
            }
        }
    }

    void multiply(bool skip_signal = false) {
        Label end, skip;

        cmp(8 | ge | f0[0], slm_counter, slm_nbuf - 1);

        if (conf.ow != ow_padded) cmp(8 | f0[0] | lt, ow, conf.ow);
        if (oc_padded != oc_tg_padded) cmp(8 | f0[0] | lt, oc_fused, oc_padded);
        if_(8 | f0[0], skip, end);

        if (slm_nbuf == 3) wait();

        if (enable_smem_read) {
            for (int i = 0; i < a_slm_blocks; i++) {
                mad(1, a_slm_rd[i].d(2), a_slm_off_rd_init[i], slm_buf_compute,
                        uint16_t(ab_slm_size / 16));
            }

            for (int i = 0; i < b_slm_blocks; i++) {
                mad(1, b_slm_rd[i].d(2), b_slm_off_rd_init[i], slm_buf_compute,
                        uint16_t(ab_slm_size / 16));
            }

            auto sync_mask = to_sbid_mask({dpasw_sbid(0), dpasw_sbid(1),
                    dpasw_sbid(2), dpasw_sbid(3)});
            sync(SyncFunction::allrd, sync_mask);

            // Load A.
            for (int i = 0; i < mb_block / 2; i += 8) {
                int owords = std::min(8, mb_block / 2 - i) * 2;
                int dist = std::min(7, b_slm_blocks + a_slm_blocks - i / 8);
                load(16 | SWSB(smem_a_sbid(i / 8), dist), A[i],
                        block_oword(owords), SLM, a_slm_rd[i / 8]);
            }

            // Load B.
            for (int i = 0; i < B.getLen(); i += 8) {
                int dist = b_slm_blocks - i / 8;
                load(16 | SWSB(dpasw_sbid(i / 8), dist), B[i], block_oword(16),
                        SLM, b_slm_rd[i / 8]);
            }
        }

        // Multiply C = A * B.
        for (int oc = 0, idx = 0; oc < 32; oc += 8, idx++) {
            // Compute [0:mb_block, oc:oc + 8] block of C.
            multiply_chunk(idx, B[oc % B.getLen()], C[idx * mb_block],
                    /*wait_a=*/oc == 0);

            // Reuse B[oc] for later multiplies (case of oc_group == 2).
            if (reuse_b_regs && oc < B.getLen()) {
                load(16 | dpasw_sbid(idx), B[oc], block_oword(16), SLM,
                        b_slm_rd[(oc + 16) / 8]);
            }
        }

        // SLM writes are flushed at this point (after SLM reads) so skip fence.
        if (slm_nbuf == 3 && !skip_signal)
            fence_and_signal(/*skip_fence=*/true);

        else_(8, end, end);
        mark(skip);
        if (slm_nbuf == 3) {
            wait();
            if (!skip_signal) fence_and_signal();
        }
        mark(end);
        endif(8);
    }

    bool need_to_restore_zero_padding() const {
        bool has_mb_padding = (conf.mb % conf.mb_block != 0);
        bool has_oc_padding = (conf.oc % 32 != 0);

        if (!has_mb_padding && !has_oc_padding) return false;

        if (conf.with_bias) return true;

        for (int po_idx = 0; po_idx < attr_info.all_post_ops.len(); po_idx++) {
            auto &e = attr_info.all_post_ops.entry_[po_idx];
            if (e.kind != primitive_kind::eltwise) continue;
            if (!eltwise_fwd_pd_t::eltwise_preserves_zero(e.eltwise))
                return true;
        }
        return false;
    }

    // Computes register offset for n-th row (across mb_block block). The rows
    // are interleaved after dpasw.
    int mb_off(int mb_idx) {
        // [32:40] range has direct mapping for 40n32c layout.
        if (mb_idx >= 32 && mb_block == 40) return mb_idx;

        int shuf[4] = {0, 2, 1, 3};
        int x = mb_idx / 4;
        int y = (x / 4) * 4 + shuf[x % 4];
        return y * 4 + (mb_idx % 4);
    }

    int c_off(int mb_idx, int mb_inner, int oc_idx, int oc_inner) {
        return (oc_idx + oc_inner) / 8 * mb_block + mb_off(mb_idx + mb_inner);
    }

    void reorder_C_to_dense(int mb_idx, int mb_step, int oc_idx, int oc_step) {
        int ireg = 0;
        for_(int mb_inner = 0; mb_inner < mb_step; mb_inner++)
        for_(int oc_inner = 0; oc_inner < oc_step; oc_inner += 8)
        {
            auto C_dense_reg = C_dense[ireg++];
            auto C_reg = C[c_off(mb_idx, mb_inner, oc_idx, oc_inner)];
            mov(8, C_dense_reg.retype(post_op_type), C_reg.retype(acc_type));
        }
    }

    void load_C_old() {
        if (!attr_info.with_sum) return;

        // Read 128 bytes of destination.
        C_old = ra.alloc_range(4);
        load(16, C_old[0], block_oword(8), A64, dst_header);
        C_old_cvt = (dst_size == 4)
                ? C_old
                : ra.alloc_range(4 * (sizeof(float) / dst_size));
    }

    void convert_C_old() {
        if (!attr_info.with_sum) return;

        convertor_t cvt(this, 128 / dst_size, DataType::f, sum_type);
        cvt.convert(C_old_cvt[0].f(0), C_old[0].sub(0, sum_type), ra);

        if (C_old_cvt != C_old) ra.safeRelease(C_old);
    }

    void apply_bias(int mb_idx, int mb_step, int oc_idx, int oc_step) {
        if (!conf.with_bias) return;

        int ireg = 0;
        for_(int mb_inner = 0; mb_inner < mb_step; mb_inner++)
        for_(int oc_inner = 0; oc_inner < oc_step; oc_inner += 16)
        {
            auto C_reg = C_dense[ireg];
            add(16, C_reg.f(), C_reg.f(), bia[oc_inner / 8].f());
            ireg += 2;
        }
    }

    void apply_oscales(int mb_idx, int mb_step, int oc_idx, int oc_step) {
        if (!attr_info.with_oscales) return;

        int ireg = 0;
        for_(int mb_inner = 0; mb_inner < mb_step; mb_inner++)
        for_(int oc_inner = 0; oc_inner < oc_step; oc_inner += 16)
        {
            auto C_reg = C_dense[ireg];
            if (attr_info.with_common_oscales) {
                mul(16, C_reg.f(), C_reg.f(), common_oscales);
            } else {
                // Per-oc output scales.
                mul(16, C_reg.f(), C_reg.f(), oscales[oc_inner / 8].f());
            }
            ireg += 2;
        }
    }

    void apply_post_ops() {
        for (int po_idx = 0; po_idx < attr_info.all_post_ops.len(); po_idx++) {
            auto &e = attr_info.all_post_ops.entry_[po_idx];
            switch (e.kind) {
                case primitive_kind::sum:
                    if (e.sum.scale != 0) {
                        for (int ireg = 0; ireg < C_dense.getLen(); ireg += 2) {
                            auto upd = C_dense[ireg].f();
                            auto old = C_old_cvt[ireg];
                            mad(16, upd, upd, old.f(), sum_scale);
                        }
                    }
                    break;
                case primitive_kind::eltwise: {
                    jit_eltwise_injector_f32<HW::Gen12HP> inj(this,
                            e.eltwise.alg, e.eltwise.alpha, e.eltwise.beta,
                            e.eltwise.scale);
                    auto scratch = ra.alloc_range(inj.preferred_scratch_regs());
                    inj.set_scratch(scratch);
                    inj.prepare();
                    inj.compute(C_dense);
                    ra.safeRelease(scratch);
                    break;
                }
                default: assert(!"not supported");
            }
        }
    }

    void convert_C_to_dst(const GRFRange &C_tmp, int mb_idx, int mb_step,
            int oc_idx, int oc_step) {
        convertor_t cvt(this, 128 / dst_size, dst_type, post_op_type);
        cvt.convert(C_tmp[0].retype(dst_type)[0],
                C_dense[0].retype(post_op_type)[0], ra);
    }

    void read_update_write_dst_range(
            int mb_idx, int mb_step, int oc_idx, int oc_step) {
        bool do_f32_cvt = conf.with_bias || attr_info.with_oscales
                || attr_info.all_post_ops.len() > 0;
        post_op_type = do_f32_cvt ? DataType::f : acc_type;

        // Load old values if needed for sum post-op.
        load_C_old();

        // Reorder mb_step x oc_step block of C to a dense GRF region and
        // convert for post-ops if needed.
        C_dense = ra.alloc_range(4 * (4 / dst_size));
        reorder_C_to_dense(mb_idx, mb_step, oc_idx, oc_step);

        // Convert old values to f32 if needed for sum post-op.
        convert_C_old();

        apply_bias(mb_idx, mb_step, oc_idx, oc_step);
        apply_oscales(mb_idx, mb_step, oc_idx, oc_step);
        apply_post_ops();

        // Zero out the padded area if needed.
        if (need_to_restore_zero_padding()) {
            auto oc_vec = ra.alloc_range(2);
            auto oc_tmp = ra.alloc_sub<int32_t>();
            mov(8, oc_vec[0].uw(0), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
            add(8, oc_vec[0].uw(8), oc_vec[0].uw(), uint16_t(8));
            add(1, oc_tmp, -oc, conf.oc);
            int ireg = 0;
            for (int mb_inner = 0; mb_inner < mb_step; mb_inner++) {
                add(16, oc_vec[1].uw(), oc_vec[0].uw(), uint16_t(oc_idx));
                for (int oc_inner = 0; oc_inner < oc_step; oc_inner += 16) {
                    // For non-padded area: (mb + mb_idx + mb_inner) < MB.
                    cmp(16 | lt | f0[0], mb, conf.mb - mb_idx - mb_inner);
                    // For non-padded area: (oc + oc_idx + oc_inner) < OC.
                    cmp(16 | f0[0] | gt, oc_tmp, oc_vec[1].uw());
                    mov(16 | ~f0[0], C_dense[ireg].f(), 0.0f);
                    if (oc_inner + 16 < oc_step)
                        add(16, oc_vec[1].uw(), oc_vec[1].uw(), 16);
                    ireg += 2;
                }
            }
            ra.safeRelease(oc_vec);
            ra.safeRelease(oc_tmp);
        }

        // Convert to the destination type and write.
        auto C_tmp = ra.alloc_range(4);
        convert_C_to_dst(C_tmp, mb_idx, mb_step, oc_idx, oc_step);
        store(16, block_oword(8), A64, dst_header, C_tmp[0]);
        ra.safeRelease(C_tmp);

        ra.safeRelease(C_dense);
        if (attr_info.with_sum) ra.safeRelease(C_old_cvt);
    }

    void load_bias(int oc_idx, int oc_step) {
        if (!conf.with_bias) return;

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

    void convert_bias(int oc_idx, int oc_step) {
        UNUSED(oc_idx);
        if (!conf.with_bias) return;

        convertor_t cvt(this, oc_step, DataType::f, bia_type, 4 / bia_size);
        cvt.convert(bia[0].f(0), bia[0].retype(bia_type)[0], ra);
    }

    void load_per_oc_oscales(int oc_idx, int oc_step) {
        if (!attr_info.with_per_oc_oscales) return;

        // TODO: Add bounds check.
        add(1, oc_off[0].d(2), oscales_off_init,
                int32_t(oc_idx * sizeof(float)));
        load(16, oscales[0], aligned_block_oword(oc_step * sizeof(float) / 16),
                Surface(oscales_surf), oc_off[0]);
    }

    void read_update_write_dst() {
        if (!enable_gmem_write) return;

        dst_header = ra.alloc();
        dst_off_init = ra.alloc_sub<uint64_t>();

        Label skip;

        bool do_if = (oc_padded != oc_tg_padded);
        if (conf.ver == ver_v1) {
            if (conf.ow != ow_padded) do_if = true;
        } else {
            // conf.ver == ver_v2.
            if (odhw % conf.sp_group != 0) do_if = true;
        }

        if (do_if) {
            cmp(8 | lt | f0[0], oc, oc_padded);
            cmp(8 | f0[0] | lt, ow, conf.ow);
            if (conf.ver == ver_v2) {
                if (has_h) cmp(8 | f0[0] | lt, oh, conf.oh);
                if (has_d) cmp(8 | f0[0] | lt, od, conf.od);
            }
            if_(8 | f0[0], skip, skip);
        }

        // Compute destination offset.
        // (mb / MB_BLOCK) * (OC / dst_oc_block) * OD * OH * OW * MB_BLOCK * dst_oc_block
        mul(1, dst_off_init, mb,
                conf.od * conf.oh * conf.ow
                        * utils::rnd_up(conf.oc, dst_oc_block));

        // (oc / 32) * OD * OH * OW * MB_BLOCK * 32
        mul(1, tmp0.uq(0), oc, conf.od * conf.oh * conf.ow * mb_block);
        add(1, dst_off_init.ud(0), dst_off_init.ud(0), tmp0.d(0));

        if (has_d) {
            // od * OH * OW * mb_block * dst_oc_block
            mul(1, tmp0.uq(0), od, conf.oh * conf.ow * mb_block * dst_oc_block);
            add(1, dst_off_init.ud(0), dst_off_init.ud(0), tmp0.d(0));
        }

        if (has_h) {
            // oh * OW * mb_block * dst_oc_block
            mul(1, tmp0.uq(0), oh, conf.ow * mb_block * dst_oc_block);
            add(1, dst_off_init.ud(0), dst_off_init.ud(0), tmp0.d(0));
        }

        // ow * mb_block * dst_oc_block
        mad(1, dst_off_init.ud(0), dst_off_init.ud(0), ow,
                uint16_t(mb_block * dst_oc_block));

        // Convert offset from elements to bytes.
        shl(1, dst_off_init, dst_off_init, ngen::utils::log2(dst_size));
        add(1, dst_off_init, dst_off_init, dst_ptr);

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

        int dst_oc_padded = utils::rnd_up(conf.oc, dst_oc_block);
        bool check_oc = (dst_oc_padded % 32 != 0);
        for (int oc_idx = 0; oc_idx < 32; oc_idx += oc_step) {
            Label skip_oc;
            if (check_oc) {
                // Check oc + oc_idx < dst_oc_padded.
                cmp(8 | lt | f0[0], oc, dst_oc_padded - oc_idx);
                if_(8 | f0[0], skip_oc, skip_oc);
            }

            load_bias(oc_idx, oc_step);
            load_per_oc_oscales(oc_idx, oc_step);

            add(1, dst_header.uq(0), dst_off_init,
                    oc_idx * conf.od * conf.oh * conf.ow * mb_block * dst_size);

            convert_bias(oc_idx, oc_step);

            for (int mb_idx = 0; mb_idx < mb_block; mb_idx += mb_step) {
                // Update and write 128 bytes.
                read_update_write_dst_range(mb_idx, mb_step, oc_idx, oc_step);
                if (mb_idx + mb_step < mb_block)
                    add(1, dst_header.uq(0), dst_header.uq(0), 128);
            }
            if (check_oc) {
                mark(skip_oc);
                endif(8);
            }
        }

        if (do_if) {
            mark(skip);
            endif(8);
        }
    }

    const conv_conf_t &conf;
    const attr_info_t &attr_info;

    bool is_4x2_tg;

    int mb_padded;
    int ic_padded;
    int ic_bytes_padded;
    int oc_padded;
    int oc_tg_padded;
    int ow_padded;

    bool is_src_off_64_bit;
    bool is_wei_off_64_bit;

    int kdhw;
    int idhw;
    int odhw;

    int mb_read01 = 0;
    int mb_read23 = 0;

    int pad_sign;

    bool is_1st;

    // Used for 1st convolution only.
    std::vector<nw_read_region_t> nw_read_regions;

    bool has_pad_d;
    bool has_pad_h;
    bool has_pad_w;

    bool has_h;
    bool has_d;

    bool check_src_load;
    bool do_kw_loop;
    bool do_ic_loop;
    bool reuse_b_regs;

    DataType src_type;
    DataType wei_type;
    DataType bia_type;
    DataType dst_type;
    DataType acc_type;
    DataType post_op_type;

    int src_size;
    int wei_size;
    int bia_size;
    int dst_size;

    int mb_block;
    int dst_oc_block;

    int slm_nbuf;
    int a_slm_block_size;
    int b_slm_block_size;
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

    GRF tmp0;
    GRF signal_header;

    Subregister ithr0;
    Subregister ithr1;

    Subregister fused_idx;

    GRFRange a_slm_rd;
    GRFRange b_slm_rd;

    int a_slm_blocks;
    int b_slm_blocks;

    // SLM read offsets in owords for the first buffer.
    // Reading up to 3 SLM blocks for A (for 40n32c).
    Subregister a_slm_off_rd_init[3];
    Subregister b_slm_off_rd_init[4];

    Subregister a_slm_off_wr_init;
    Subregister b_slm_off_wr_init;

    GRFRange a_slm_wr;
    GRFRange b_slm_wr;

    Subregister off_tmp;

    Subregister src_off_init0;
    Subregister src_off_init1;
    Subregister src_off_init[2];
    Subregister wei_off_init;

    Subregister mb;
    Subregister ic_bytes;
    Subregister oc;
    Subregister oc_fused;
    Subregister oc_tg;

    Subregister od, oh, ow;
    Subregister id, ih, iw;
    Subregister kd, kh, kw;

    // For loop_v2.
    GRFRange _sp;
    GRF sp[2];
    GRF sp_bound;
    FlagRegister sp_check_flags[2];

    Subregister iw_load0;
    Subregister ih_load0;
    Subregister id_load0;
    Subregister iw_load[2];
    Subregister ih_load[2];
    Subregister id_load[2];

    Subregister iw_tg;
    Subregister ow_tg;

    Subregister slm_idx;
    Subregister slm_buf_load;
    Subregister slm_buf_compute;
    Subregister slm_counter;

    GRFRange src_header;
    GRFRange wei_header;
    GRF dst_header;

    Subregister src_off_ic;
    Subregister src_off_kd;
    Subregister src_off_kh;
    Subregister src_off_kw;

    GRFRange oc_off;
    Subregister bia_off_init;

    Subregister dst_off_init;

    GRFRange A_tmp;
    GRFRange A_tmp_reorder;
    GRFRange B_tmp;

    GRFRange bia;

    GRFRange C_dense;

    // Post-ops support.
    GRFRange C_old;
    GRFRange C_old_cvt;
    DataType sum_type;
    Subregister sum_scale;

    // Output scales.
    Subregister common_oscales;
    Subregister oscales_off_init;
    GRFRange oscales;

    GRFRange A;
    GRFRange B;
    GRFRange C;

    bool is_auto_swsb = false;

    // 256 registers.
    RegisterAllocator ra;
};

void convertor_t::convert_impl(
        int simd, int off, Subregister to, Subregister from, int phase) {
    assert(utils::one_of(simd, 8, 16));

    if (to_type_ == from_type_ && from_stride_ == 1) {
        if (phase == 0) host_->mov(simd, to(1), from(from_stride_));
        return;
    }

    // bf16 -> f32:
    // - bf16 must be packed: use left shift instead.
    if (from_type_ == DataType::bf && to_type_ == DataType::f) {
        to.setType(DataType::ud);
        from.setType(DataType::uw);
        if (phase == 0)
            host_->shl(simd, to(1), from(from_stride_), uint16_t(16));
        return;
    }

    // f32 -> bf16 or f32 -> f16:
    // - SIMD16 does not support mixed mode move.
    if (simd == 16 && from_type_ == DataType::f
            && utils::one_of(to_type_, DataType::bf, DataType::hf)) {
        switch (phase) {
            case 0: host_->mov(8, to(1), from(from_stride_)); return;
            case 1:
                host_->mov(8, fixup_sub(to, 8)(1),
                        fixup_sub(from, 8, from_stride_)(from_stride_));
                return;
            default: assert(!"not expected"); return;
        }
    }

    // f32/s32 -> s8/u8:
    // - Use saturation
    // - s8/u8 must be DW-strided: use temporary
    if (from_bytes_ == 4 && to_bytes_ == 1) {
        auto strided = scratch_[off / 8].retype(to_type_)[0](4);
        switch (phase) {
            case 0:
                host_->mov(simd | host_->sat, strided, from(from_stride_));
                return;
            case 1: host_->mov(simd, to(1), strided); return;
            default: assert(!"not expected"); return;
        }
    }

    // s8/u8 -> f32/s32:
    // - s8/u8 must be DW-strided: use temporary
    if (from_bytes_ == 1 && to_bytes_ == 4) {
        auto strided = scratch_[off / 8].retype(from_type_)[0](4);
        switch (phase) {
            case 0: host_->mov(simd, strided, from(from_stride_)); return;
            case 1: host_->mov(simd, to(1), strided); return;
            default: assert(!"not expected"); return;
        }
    }

    if (phase == 0) host_->mov(simd, to(1), from(from_stride_));
}

void nw_read_region_t::read_and_reorder() {
    if (w_len == 0) return;

    auto &conf = host->conf;

    // Source address is set to the initial offset (maybe in the padded area).
    auto &header = host->src_header;

    auto &w_region = host->A_tmp;
    auto &tmp = host->A_tmp_reorder;

    auto get_sbid = [&](int idx) {
        idx = (idx + 1) % 16;
        if (idx >= (int)host->gmem_b_sbid().set.getID())
            return SBID((idx + 1) % 16);
        return SBID(idx);
    };

    int off = w_shift * 16;
    int reads = 0;

    // 4 owords, 2 owords, 1 oword (up to 7 W elements to read).
    for (int i = 2; i >= 0; i--) {
        int len = (1 << i);
        if ((w_len & len) == 0) continue;
        host->add(1, header[i].uq(0), header[i].uq(0), off);
        off += (1 << i) * 16;
        reads++;
    }

    // Read data in 7w4n2c/7w4n4c format.
    // TODO: Fix out-of-bounds access when (conf.mb % mb_block) != 0.
    for (int idx = 0; idx < mb_len / 4; idx++) {
        SBID sbids[] = {get_sbid(3 * idx + 0), get_sbid(3 * idx + 1),
                get_sbid(3 * idx + 2)};

        int read_off = 0;
        int dist = reads;
        for (int i = 2; i >= 0; i--) {
            int len = (1 << i);
            if ((w_len & len) == 0) continue;
            host->load(16 | SWSB(sbids[i], dist--), tmp[4 * idx + read_off / 2],
                    block_oword(len), host->A64, header[i]);
            read_off += len;
        }

        if (idx + 1 < mb_len / 4) {
            for (int i = 2; i >= 0; i--) {
                int len = (1 << i);
                if ((w_len & len) == 0) continue;
                host->add(1 | sbids[i].src, header[i].uq(0), header[i].uq(0),
                        conf.id * conf.ih * conf.iw * 4 * 4 * host->src_size);
            }
        }
    }

    // Reorder data from 7w4n2c/7w4n4c to 4n8w2c/4n8w4c.
    for (int idx = 0; idx < mb_len / 4; idx++) {
        SBID sbids[] = {get_sbid(3 * idx + 0), get_sbid(3 * idx + 1),
                get_sbid(3 * idx + 2)};

        uint32_t sync_mask = host->to_sbid_mask({sbids[0], sbids[1], sbids[2]});
        host->sync(SyncFunction::allwr, sync_mask);

        for (int w = 0; w < w_len; w += 2) {
            int w_simd = std::min(w_len - w, 2);
            for (int n = 0; n < 4; n++) {
                host->mov(w_simd, w_region[4 * idx + n].d(w_shift + w),
                        tmp[4 * idx + w / 2].d(n)(4));
            }
        }
    }
}

} // namespace

status_t gen12hp_conv_data_create_kernel(const conv_conf_t &conf,
        compute::kernel_t *kernel, gpu_primitive_t *primitive,
        engine_t *engine) {
    gen12hp_conv_data_kernel_t ngen_kernel(conf);
    return primitive->create_kernel(engine, kernel, ngen_kernel);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

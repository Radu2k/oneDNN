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

#ifndef GPU_JIT_CONV_CONFIG_HPP
#define GPU_JIT_CONV_CONFIG_HPP

#include <iostream>
#include <sstream>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/math_utils.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/jit/conv/tensor.hpp"
#include "gpu/jit/conv/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Description of the convolution problem.
class conv_problem_t {
public:
    conv_problem_t() = default;

    status_t init(convolution_pd_t *conv_pd) {
        const convolution_desc_t &desc = *conv_pd->desc();
        (void)desc;

        is_fwd = conv_pd->is_fwd();
        with_bias = conv_pd->with_bias();
        with_groups = conv_pd->with_groups();

        orig_src_md = *conv_pd->arg_md(DNNL_ARG_SRC);
        orig_wei_md = *conv_pd->arg_md(DNNL_ARG_WEIGHTS);
        orig_dst_md = *conv_pd->arg_md(DNNL_ARG_DST);

        src_data_type = orig_src_md.data_type;
        wei_data_type = orig_wei_md.data_type;
        dst_data_type = orig_dst_md.data_type;

        ndims = conv_pd->ndims();

        mb = conv_pd->MB();
        g = conv_pd->G();
        ic = conv_pd->IC();
        oc = conv_pd->OC();

        // Input spatial.
        id = conv_pd->ID();
        ih = conv_pd->IH();
        iw = conv_pd->IW();

        // Output spatial.
        od = conv_pd->OD();
        oh = conv_pd->OH();
        ow = conv_pd->OW();

        // Kernel sizes.
        kd = conv_pd->KD();
        kh = conv_pd->KH();
        kw = conv_pd->KW();

        // Strides.
        sd = conv_pd->KSD();
        sh = conv_pd->KSH();
        sw = conv_pd->KSW();

        // Padding.
        pd = conv_pd->padFront();
        ph = conv_pd->padT();
        pw = conv_pd->padL();

        // Dilation.
        dd = conv_pd->KDD();
        dh = conv_pd->KDH();
        dw = conv_pd->KDW();

        try_reduce_to_1d();

        return status::success;
    }

    // Reduces dimensions for 1x1 kernel.
    void try_reduce_to_1d() {
        bool is_1x1 = (kd * kh * kw == 1);
        bool is_stride1 = (sd == 1 && sh == 1 && sw == 1);
        bool is_eq_oi = (od == id && oh == ih && ow == iw);
        if (is_1x1 && is_stride1 && is_eq_oi) {
            ir_assert(pd == 0 && ph == 0 && pw == 0);
            ow = od * oh * ow;
            iw = id * ih * iw;
            od = id = kd = 1;
            oh = ih = kh = 1;
            reduced_to_1d = true;
        }
    }

    memory_desc_wrapper orig_src_mdw() const {
        return memory_desc_wrapper(orig_src_md);
    }
    memory_desc_wrapper orig_wei_mdw() const {
        return memory_desc_wrapper(orig_wei_md);
    }
    memory_desc_wrapper orig_dst_mdw() const {
        return memory_desc_wrapper(orig_dst_md);
    }
    format_tag_t src_tag() const { return src_layout.to_format_tag(); }
    format_tag_t wei_tag() const { return wei_layout.to_format_tag(); }
    format_tag_t dst_tag() const { return dst_layout.to_format_tag(); }

    std::string desc_str() const {
        std::ostringstream oss;
        oss << "mb" << mb;
        oss << "ic" << ic;
        oss << "id" << id;
        oss << "ih" << ih;
        oss << "iw" << iw;
        oss << "oc" << oc;
        oss << "od" << od;
        oss << "oh" << oh;
        oss << "ow" << ow;
        oss << "kd" << kd;
        oss << "kh" << kh;
        oss << "kw" << kw;
        oss << "pd" << pd;
        oss << "ph" << ph;
        oss << "pw" << pw;
        return oss.str();
    }

    memory_desc_t orig_src_md;
    memory_desc_t orig_wei_md;
    memory_desc_t orig_dst_md;

    layout_t src_layout;
    layout_t wei_layout;
    layout_t dst_layout;

    data_type_t src_data_type;
    data_type_t wei_data_type;
    data_type_t dst_data_type;

    bool is_fwd;
    bool with_bias;
    bool with_groups;

    int ndims;
    int mb; // Batch size.
    int g; // Groups.
    int ic, oc; // Input and output channels.
    int id, ih, iw; // Input spatial sizes.
    int od, oh, ow; // Output spatial sizes.
    int kd, kh, kw; // Kernel sizes.
    int sd, sh, sw; // Strides.
    int pd, ph, pw; // Padding in the beginning.
    int dd, dh, dw; // Dilation.
    bool reduced_to_1d; // Whether the problem spatial was reduced to 1D.
};

// Parameters for kernel generation.
class conv_config_t : public conv_problem_t {
public:
    conv_config_t() = default;

    status_t init(convolution_pd_t *conv_pd, engine_t *engine) {
        using namespace ir_utils;

        CHECK(conv_problem_t::init(conv_pd));
        CHECK(init_acc_data_type());

        // Cases below are not supported yet.
        if (!is_fwd) return status::unimplemented;
        if (with_bias) return status::unimplemented;
        if (with_groups) return status::unimplemented;
        if (!conv_pd->attr()->has_default_values())
            return status::unimplemented;
        if (utils::one_of(data_type::f32, src_data_type, wei_data_type))
            return status::unimplemented;

        // First convolution is not supported.
        if (ic < 16) return status::unimplemented;
        // Current implementation performs full unrolling across the filter,
        // limit the filter size to avoid code bloat.
        if (kd * kh * kw >= 25) return status::unimplemented;

        // Set dispatch and kernel parameters.
        int mb_thr_blk = (mb < 16 ? 1 : 32);
        int oc_thr_blk = 32;
        int ow_thr_blk = (mb < 16 ? 16 : 1);
        if (ow < ow_thr_blk) ow_thr_blk = 8;

#ifdef GEN_CONV_DEBUG
        mb_thr_blk = getenv_int("mb_thr_blk", mb_thr_blk);
        oc_thr_blk = getenv_int("oc_thr_blk", oc_thr_blk);
        ow_thr_blk = getenv_int("ow_thr_blk", ow_thr_blk);
#endif

        simd_size = 8;
        regs = 256;
        tg_grid_dim[0] = std::min(4, utils::div_up(oc, oc_thr_blk));
        tg_grid_dim[1] = std::min(4, utils::div_up(ow, ow_thr_blk));
        tg_grid_dim[2] = 1;

        // Round down to a power of 2.
        tg_grid_dim[0] = (1 << math::ilog2q(tg_grid_dim[0]));
        tg_grid_dim[1] = (1 << math::ilog2q(tg_grid_dim[1]));
        tg_grid_dim[2] = (1 << math::ilog2q(tg_grid_dim[2]));

#ifdef GEN_CONV_DEBUG
        tg_grid_dim[0] = getenv_int("tg0", tg_grid_dim[0]);
        tg_grid_dim[1] = getenv_int("tg1", tg_grid_dim[1]);
#endif

        mb_tg_blk = mb_thr_blk;
        oc_tg_blk = tg_grid_dim[0] * oc_thr_blk;
        ow_tg_blk = tg_grid_dim[1] * ow_thr_blk;
        ic_blk = (is_s32_accumulator() ? 32 : 16);

#ifdef GEN_CONV_DEBUG
        mb_tg_blk = getenv_int("mb_tg_blk", mb_tg_blk);
        oc_tg_blk = getenv_int("oc_tg_blk", oc_tg_blk);
        ow_tg_blk = getenv_int("ow_tg_blk", ow_tg_blk);
#endif

        m_tg_blk = mb_tg_blk * ow_tg_blk;
        n_tg_blk = oc_tg_blk;
        k_tg_blk = ic_blk;

        int mb_tg_padded = utils::rnd_up(mb, mb_tg_blk);
        int oc_tg_padded = utils::rnd_up(oc, oc_tg_blk);
        int ow_tg_padded = utils::rnd_up(ow, ow_tg_blk);

        int mb_tg_dim = mb_tg_padded / mb_tg_blk;
        int oc_tg_dim = oc_tg_padded / oc_tg_blk;

        ow_tg_dim = ow_tg_padded / ow_tg_blk;

        kernel_grid_dim[0] = oc_tg_dim;
        kernel_grid_dim[1] = od * oh * ow_tg_dim;
        kernel_grid_dim[2] = mb_tg_dim;

        use_a_slm = true;
        use_b_slm = true;
        use_dpasw = true;
        pad_slm = true;
        assign_sbids = true;
        slm_bufs = (tg_grid_dim[0] * tg_grid_dim[1] <= 8 ? 2 : 3);
        gmem_bufs = 2;
        reduce_grf_usage = true;
        a_sub_tiles = 1;
        b_sub_tiles = 1;

#ifdef GEN_CONV_DEBUG
        use_a_slm = getenv_bool("use_a_slm", use_a_slm);
        use_b_slm = getenv_bool("use_b_slm", use_b_slm);
        use_dpasw = getenv_bool("use_dpasw", use_dpasw);
        pad_slm = getenv_bool("pad_slm", pad_slm);
        assign_sbids = getenv_bool("assign_sbids", assign_sbids);
        slm_bufs = getenv_int("slm_bufs", slm_bufs);
        gmem_bufs = getenv_int("gmem_bufs", gmem_bufs);
        reduce_grf_usage = getenv_bool("reduce_grf_usage", reduce_grf_usage);
        a_sub_tiles = getenv_int("a_sub_tiles", a_sub_tiles);
        b_sub_tiles = getenv_int("b_sub_tiles", b_sub_tiles);
#endif

        std::string src_tag;
        std::string wei_tag;
        std::string dst_tag;
        if (is_s32_accumulator()) {
            src_tag = (mb_thr_blk == 1 ? "aBx32b" : "ABx32a32b");
            wei_tag = "ABx4a8b8a4b";
            dst_tag = (mb_thr_blk == 1 ? "aBx32b" : "ABx32a32b");
        } else {
            src_tag = (mb_thr_blk == 1 ? "aBx16b" : "ABx32a16b");
            wei_tag = "ABx4a8b8a2b";
            dst_tag = (mb_thr_blk == 1 ? "aBx16b" : "ABx32a16b");
        }

#ifdef GEN_CONV_DEBUG
        src_tag = getenv_str("stag", src_tag);
        wei_tag = getenv_str("wtag", wei_tag);
        dst_tag = getenv_str("dtag", dst_tag);
#endif

        fixup_inference_consistency();
        try_reduce_grf_usage();

        auto &src_md = *conv_pd->invariant_src_md();
        auto &wei_md = *conv_pd->invariant_wei_md();
        auto &dst_md = *conv_pd->invariant_dst_md();

        // Select layouts.
        src_layout = init_layout(src_md, src_tag);
        wei_layout = init_layout(wei_md, wei_tag);
        dst_layout = init_layout(dst_md, dst_tag);

        // Validate layouts.
        bool is_src_nhwc = (orig_src_mdw().is_plain()
                && src_layout == layout_t(src_md, "axb"));
        bool is_dst_nhwc = (orig_dst_mdw().is_plain()
                && dst_layout == layout_t(dst_md, "axb"));
        if (is_src_nhwc != is_dst_nhwc) return status::unimplemented;

        if (!is_src_nhwc && src_layout != layout_t(src_md, src_tag))
            return status::unimplemented;
        if (!is_dst_nhwc && dst_layout != layout_t(dst_md, dst_tag))
            return status::unimplemented;

        if (wei_layout != layout_t(wei_md, wei_tag))
            return status::unimplemented;

        // HWord loads require 32 byte alignment. For NHWC layout it means
        // input/output channels must be multiples of 32 bytes.
        size_t ic_bytes = ic * types::data_type_size(src_data_type);
        size_t oc_bytes = oc * types::data_type_size(dst_data_type);
        if (is_src_nhwc && (ic_bytes % 32 != 0 || oc_bytes % 32 != 0))
            return status::unimplemented;

        // Blocked large batch performance is slightly behind.
        if (!is_src_nhwc && mb >= 16) return status::unimplemented;

        return status::success;
    }

    bool is_s32_accumulator() const { return acc_data_type == data_type::s32; }

    compute::nd_range_t nd_range() const {
        size_t gws[3];
        size_t lws[3];
        for (int i = 0; i < 3; i++) {
            lws[i] = tg_grid_dim[i] * (i == 0 ? simd_size : 1);
            gws[i] = kernel_grid_dim[i] * lws[i];
        }
        return compute::nd_range_t(gws, lws);
    }

    std::string str() const {
        using namespace ir_utils;

        std::ostringstream oss;
        // clang-format off
        oss << "  Problem:                    " << desc_str() << std::endl;
        oss << "  Source layout:              " << src_layout << std::endl;
        oss << "  Weights layout:             " << wei_layout << std::endl;
        oss << "  Destination layout:         " << dst_layout << std::endl;
        oss << "  MB TG block:                " << mb_tg_blk << std::endl;
        oss << "  OW TG block:                " << ow_tg_blk << std::endl;
        oss << "  OC TG block:                " << oc_tg_blk << std::endl;
        oss << "  Thread group:               " << make_seq_print_helper(tg_grid_dim, " x ") << std::endl;
        oss << "  Use SLM for A:              " << to_string(use_a_slm) << std::endl;
        oss << "  Use SLM for B:              " << to_string(use_b_slm) << std::endl;
        oss << "  Use DPASW:                  " << to_string(use_dpasw) << std::endl;
        oss << "  Pad SLM:                    " << to_string(pad_slm) << std::endl;
        oss << "  Assign SBIDs:               " << to_string(assign_sbids) << std::endl;
        oss << "  SLM buffers:                " << slm_bufs << std::endl;
        oss << "  GMEM to SLM, GRF buffers:   " << gmem_bufs << std::endl;
        oss << "  Reduce GRF usage:           " << to_string(reduce_grf_usage) << std::endl;
        oss << "  A sub-tiles:                " << a_sub_tiles << std::endl;
        oss << "  B sub-tiles:                " << b_sub_tiles << std::endl;
        // clang-format on
        return oss.str();
    }

    data_type_t acc_data_type;

    int simd_size; // SIMD width.
    int regs; // Number of registers.

    // Thread group dimensions (thread group grid).
    std::array<int, 3> tg_grid_dim;

    // Number of thread groups across dimensions (kernel grid).
    std::array<int, 3> kernel_grid_dim;

    int ow_tg_dim;

    // Block sizes per thread group (convolution notation).
    int mb_tg_blk;
    int oc_tg_blk;
    int ow_tg_blk;
    int ic_blk;

    // Block sizes per thread group (GEMM notation).
    int m_tg_blk;
    int n_tg_blk;
    int k_tg_blk;

    bool use_a_slm; // Whether to use SLM for A.
    bool use_b_slm; // Whether to use SLM for B.
    bool use_dpasw; // Whether to use DPASW.
    bool pad_slm; // Whether to pad SLM to avoid write conflicts.
    bool assign_sbids; // Whether to manually assign SBID tokens.
    int slm_bufs; // Number of SLM buffers to use.
    int gmem_bufs; // Number of GRF buffers to use for GMEM -> SLM copy.
    bool reduce_grf_usage; // Whether to try to reduce GRF usage based on heuristics.

    // Sub-tiles to split into for the inner A x B multiplication:
    // for i in range(0, a_sub_tiles):
    //     A_i = load(...)
    //     for j in range(0, b_sub_tiles):
    //         B_j = load(...)
    //         C_i_j += A_i * B_j
    //
    // GRF buffers for A_i and B_j are reused. Factors greater than one help to
    // reduce GRF usage.
    int a_sub_tiles;
    int b_sub_tiles;

private:
    status_t init_acc_data_type() {
        data_type_t sdt = src_data_type;
        data_type_t wdt = wei_data_type;
        data_type_t ddt = dst_data_type;
        if (utils::one_of(sdt, data_type::s8, data_type::u8)
                && utils::one_of(wdt, data_type::s8, data_type::u8)) {
            acc_data_type = data_type::s32;
            return status::success;
        }
        if (utils::everyone_is(data_type::f16, sdt, wdt)
                || utils::everyone_is(data_type::bf16, sdt, wdt)) {
            acc_data_type = data_type::f32;
            return status::success;
        }
        if (utils::everyone_is(data_type::f32, sdt, wdt, ddt)) {
            acc_data_type = data_type::f32;
            return status::success;
        }
        return status::unimplemented;
    }

    // Overwrites parameters that are implied by other parameters.
    void fixup_inference_consistency() {
        if (tg_grid_dim[0] == 1) use_a_slm = false;
        if (tg_grid_dim[1] == 1) use_b_slm = false;
        if (!use_a_slm && !use_b_slm) {
            slm_bufs = 0;
            gmem_bufs = 1;
        }
        if (slm_bufs == 0) {
            use_a_slm = false;
            use_b_slm = false;
        }
        if (tg_grid_dim[0] % 2 != 0) use_dpasw = false;
    }

    void try_reduce_grf_usage() {
        if (!reduce_grf_usage) return;

        int max_regs = int(regs * 0.95);
        int regs = estimate_register_count();
        if (regs <= max_regs) return;

        // Try to reduce disable GRF buffering.
        if (gmem_bufs != 1) {
            gmem_bufs = 1;
            int regs = estimate_register_count();
            if (regs <= max_regs) return;
        }

        // Try to use sub-tiles for B.
        if (b_sub_tiles == 1) {
            b_sub_tiles = 2;
            int regs = estimate_register_count();
            if (regs <= max_regs) return;
        }

        // Try to use double SLM buffering.
        if (slm_bufs == 3) {
            slm_bufs = 2;
            int regs = estimate_register_count();
            if (regs <= max_regs) return;
        }

        // Try to use single SLM buffering.
        if (slm_bufs == 2) {
            slm_bufs = 1;
            int regs = estimate_register_count();
            if (regs <= max_regs) return;
        }
    }

    int estimate_register_count() const {
        int reg_bytes = 32;
        int gmem_msg_bytes = reg_bytes; // Assume 1 register per GMEM load.
        int slm_msg_bytes
                = 8 * reg_bytes; // Assume 8 registers per SLM load/store.

        int nthr = tg_grid_dim[0] * tg_grid_dim[1];
        int m_thr_blk = utils::div_up(m_tg_blk, tg_grid_dim[1]);
        int n_thr_blk = utils::div_up(n_tg_blk, tg_grid_dim[0]);
        int k_thr_blk = k_tg_blk;

        int ssize = int(types::data_type_size(orig_src_md.data_type));
        int wsize = int(types::data_type_size(orig_wei_md.data_type));
        int asize = int(types::data_type_size(acc_data_type));

        // Registers for C += A * B operation.
        int a_bytes = utils::div_up(m_thr_blk * k_thr_blk * ssize, a_sub_tiles);
        int b_bytes = utils::div_up(k_thr_blk * n_thr_blk * wsize, b_sub_tiles);
        int c_bytes = m_thr_blk * n_thr_blk * asize;

        int a_regs = utils::div_up(a_bytes, reg_bytes);
        int b_regs = utils::div_up(b_bytes, reg_bytes);
        int c_regs = utils::div_up(c_bytes, reg_bytes);

        int a_headers = utils::div_up(
                a_bytes, use_a_slm ? slm_msg_bytes : gmem_msg_bytes);
        int b_headers = utils::div_up(
                b_bytes, use_b_slm ? slm_msg_bytes : gmem_msg_bytes);

        if (use_dpasw) {
            // Pessimistically reduce the smallest buffer by 2x.
            if (a_regs < b_regs) {
                a_regs /= 2;
                a_headers /= 2;
            } else {
                b_regs /= 2;
                b_headers /= 2;
            }
        }

        // Temporary registers for GMEM -> SLM load.
        int a_g2s_bytes
                = (use_a_slm ? utils::div_up(m_tg_blk * k_tg_blk * ssize, nthr)
                             : 0);
        int b_g2s_bytes
                = (use_b_slm ? utils::div_up(k_tg_blk * n_tg_blk * wsize, nthr)
                             : 0);

        int a_g2s_regs = utils::div_up(a_g2s_bytes, reg_bytes);
        int b_g2s_regs = utils::div_up(b_g2s_bytes, reg_bytes);

        // Two sets of headers for GMEM -> GRF and GRF -> SLM.
        int a_g2s_headers = utils::div_up(a_g2s_bytes, gmem_msg_bytes)
                + utils::div_up(a_g2s_bytes, slm_msg_bytes);
        int b_g2s_headers = utils::div_up(b_g2s_bytes, gmem_msg_bytes)
                + utils::div_up(b_g2s_bytes, slm_msg_bytes);

        int g2s_regs = gmem_bufs * (a_g2s_regs + b_g2s_regs);
        int g2s_headers = a_g2s_headers + b_g2s_headers;

        int data_regs = a_regs + b_regs + c_regs + g2s_regs;
        int header_regs = a_headers + b_headers + g2s_headers;

        return data_regs + header_regs;
    }

    static layout_t init_layout(
            const memory_desc_t &md, const std::string &tag) {
        if (md.format_kind != format_kind::any) return layout_t(md);
        return layout_t(md, tag);
    }
};

inline std::ostream &operator<<(std::ostream &out, const conv_config_t &cfg) {
    out << cfg.str();
    return out;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

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
#include "gpu/compute/compute_engine.hpp"
#include "gpu/jit/conv/fma_support.hpp"
#include "gpu/jit/conv/tensor.hpp"
#include "gpu/jit/conv/utils.hpp"
#include "gpu/jit/jit_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Description of the convolution problem.
class conv_problem_t {
public:
    conv_problem_t() = default;

    status_t init(convolution_pd_t *conv_pd) {
        is_fwd = conv_pd->is_fwd();
        is_bwd_d = conv_pd->is_bwd_d();
        is_bwd_w = conv_pd->is_bwd_w();
        with_bias = conv_pd->with_bias();
        with_groups = conv_pd->with_groups();

        orig_src_md = *conv_pd->invariant_src_md();
        orig_wei_md = *conv_pd->invariant_wei_md();
        orig_dst_md = *conv_pd->invariant_dst_md();
        orig_bia_md = *conv_pd->invariant_bia_md();

        src_data_type = orig_src_md.data_type;
        wei_data_type = orig_wei_md.data_type;
        dst_data_type = orig_dst_md.data_type;
        bia_data_type = orig_bia_md.data_type;

        if (with_bias)
            bia_layout = layout_t(orig_bia_md, "a", /*do_normalize=*/false);

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
    memory_desc_t orig_bia_md;

    layout_t src_layout;
    layout_t wei_layout;
    layout_t dst_layout;
    layout_t bia_layout;

    data_type_t src_data_type;
    data_type_t wei_data_type;
    data_type_t dst_data_type;
    data_type_t bia_data_type;

    bool is_fwd;
    bool is_bwd_d;
    bool is_bwd_w;
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
        CHECK(conv_problem_t::init(conv_pd));
        CHECK(init_hw(engine));
        CHECK(init_abc_data_types());
        CHECK(init_acc_data_type());
        CHECK(init_fma_kind());

        if (!data_types_ok()) return status::unimplemented;
        if (!post_ops_ok(conv_pd)) return status::unimplemented;

        // Groups are not supported yet.
        if (with_groups && g > 1) return status::unimplemented;

        if (is_fwd)
            CHECK(init_fwd(conv_pd));
        else if (is_bwd_d)
            CHECK(init_bwd_d(conv_pd));
        else if (is_bwd_w)
            CHECK(init_bwd_w(conv_pd));
        else
            ir_error_not_expected();

        if (!hw_ok(engine)) return status::unimplemented;

        return status::success;
    }

    status_t init_fwd(convolution_pd_t *conv_pd) {
        using namespace ir_utils;

        // First convolution is not supported.
        if (ic < 16) return status::unimplemented;

        // Set dispatch and kernel parameters.
        int mb_thr_blk, ow_thr_blk;
        int mb_thr_dim, oc_thr_dim, ow_thr_dim;
        if (fma_kind == fma_kind_t::mad && src_data_type == data_type::f32) {
            const int max_tg_size = 16;
            mb_thr_blk = (mb < 16 ? 1 : 8);
            mb_thr_dim = std::min((mb_thr_blk != 1) ? (32 / mb_thr_blk) : 1,
                    utils::div_up(mb, mb_thr_blk));
#ifdef GEN_CONV_DEBUG
            mb_thr_blk = getenv_int("mb_thr_blk", mb_thr_blk);
#endif
            oc_thr_blk = 16;
            oc_thr_dim = std::min(4, utils::div_up(oc, oc_thr_blk));
            oc_thr_dim = (1 << math::ilog2q(oc_thr_dim));

            if (mb_thr_dim > 1) {
                ow_thr_blk = 1;
                ow_thr_dim = 1;
            } else {
                const int pref_ow_thr_dim
                        = max_tg_size / (oc_thr_dim * mb_thr_dim);
                const int pref_ow_block
                        = (mb_thr_blk == 1) ? 8 : kw > 1 ? 4 : 1;
                ow_thr_blk = ow < pref_ow_block * pref_ow_thr_dim
                        ? (1 << math::ilog2q(
                                   utils::div_up(ow, pref_ow_thr_dim)))
                        : pref_ow_block;
                ow_thr_dim = pref_ow_thr_dim;
            }
        } else {
            mb_thr_blk = (mb < 16 ? 1 : 32);
            mb_thr_dim = 1;
            oc_thr_blk = 32;
            oc_thr_dim = std::min(4, utils::div_up(oc, oc_thr_blk));
            oc_thr_dim = (1 << math::ilog2q(oc_thr_dim));
            ow_thr_blk = (mb < 16 ? 16 : 1);
            if (ow < ow_thr_blk) ow_thr_blk = 8;
            ow_thr_dim = std::min(4, utils::div_up(ow, ow_thr_blk));
        }

#ifdef GEN_CONV_DEBUG
        mb_thr_blk = getenv_int("mb_thr_blk", mb_thr_blk);
        mb_thr_dim = getenv_int("mb_thr_dim", mb_thr_dim);
        oc_thr_blk = getenv_int("oc_thr_blk", oc_thr_blk);
        oc_thr_dim = getenv_int("oc_thr_dim", oc_thr_dim);
        ow_thr_blk = getenv_int("ow_thr_blk", ow_thr_blk);
        ow_thr_dim = getenv_int("ow_thr_dim", ow_thr_dim);
#endif

        tg_grid_dim[0] = oc_thr_dim;
        tg_grid_dim[1] = mb_thr_dim * ow_thr_dim;
        tg_grid_dim[2] = 1;

        // Round down to a power of 2.
        tg_grid_dim[0] = (1 << math::ilog2q(tg_grid_dim[0]));
        tg_grid_dim[1] = (1 << math::ilog2q(tg_grid_dim[1]));
        tg_grid_dim[2] = (1 << math::ilog2q(tg_grid_dim[2]));

#ifdef GEN_CONV_DEBUG
        tg_grid_dim[0] = getenv_int("tg0", tg_grid_dim[0]);
        tg_grid_dim[1] = getenv_int("tg1", tg_grid_dim[1]);
#endif

        mb_tg_blk = mb_thr_dim * mb_thr_blk;
        oc_tg_blk = oc_thr_dim * oc_thr_blk;
        ow_tg_blk = ow_thr_dim * ow_thr_blk;
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

        CHECK(init_common_config());

        // Do not perform full unrolling when there are too many inner
        // iterations.
        if (kd * kh * kw > 9) do_loop_unroll = false;

        regs = hw <= ngen::HW::Xe_LP ? 128 : 256;
        fixup_inference_consistency();
        if (!try_reduce_grf_usage()) return status::unimplemented;

#if DNNL_WITH_XE_HPC
        const bool is_wei16aXb = hw >= ngen::HW::Xe_HPC;
        assert(hw != ngen::HW::Unknown);
#else
        const bool is_wei16aXb = false;
#endif

        std::string src_tag;
        std::string wei_tag;
        std::string dst_tag;
        if (fma_kind == fma_kind_t::mad) {
            src_tag = (mb_thr_blk == 1 ? "aBx16b" : "ABx32a16b");
            wei_tag = "BAx16b16a";
            dst_tag = (mb_thr_blk == 1 ? "aBx16b" : "ABx32a16b");

            int max_simd_size = 16;
            if (simd_size > max_simd_size) simd_size = max_simd_size;
        } else if (is_s32_accumulator()) {
            src_tag = (mb_thr_blk == 1 ? "aBx32b" : "ABx32a32b");
            wei_tag = is_wei16aXb ? "ABx2a8b16a4b" : "ABx4a8b8a4b";
            dst_tag = (mb_thr_blk == 1 ? "aBx32b" : "ABx32a32b");
        } else {
            src_tag = (mb_thr_blk == 1 ? "aBx16b" : "ABx32a16b");
            wei_tag = is_wei16aXb ? "ABx2a8b16a2b" : "ABx4a8b8a2b";
            dst_tag = (mb_thr_blk == 1 ? "aBx16b" : "ABx32a16b");
        }

        if (with_groups) wei_tag = prepend_groups_to_tag(wei_tag);

#ifdef GEN_CONV_DEBUG
        src_tag = getenv_str("stag", src_tag);
        wei_tag = getenv_str("wtag", wei_tag);
        dst_tag = getenv_str("dtag", dst_tag);
#endif

        auto &src_md = *conv_pd->invariant_src_md();
        auto &wei_md = *conv_pd->invariant_wei_md();
        auto &dst_md = *conv_pd->invariant_dst_md();
        auto &bia_md = *conv_pd->invariant_bia_md();

        // Select layouts.
        src_layout = init_layout(src_md, src_tag);
        wei_layout = init_layout(wei_md, wei_tag);
        dst_layout = init_layout(dst_md, dst_tag);
        if (with_bias) bia_layout = init_layout(bia_md, "a");

        // Validate layouts.
        bool is_src_nhwc = (orig_src_mdw().is_plain()
                && src_layout == make_layout(src_md, "axb"));
        bool is_dst_nhwc = (orig_dst_mdw().is_plain()
                && dst_layout == make_layout(dst_md, "axb"));
        if (is_src_nhwc != is_dst_nhwc) return status::unimplemented;

        if (!is_src_nhwc && src_layout != make_layout(src_md, src_tag))
            return status::unimplemented;
        if (!is_dst_nhwc && dst_layout != make_layout(dst_md, dst_tag))
            return status::unimplemented;

        if (wei_layout != make_layout(wei_md, wei_tag))
            return status::unimplemented;

        // HWord loads require 32 byte alignment. For NHWC layout it means
        // input/output channels must be multiples of 32 bytes.
        size_t ic_bytes = ic * types::data_type_size(src_data_type);
        size_t oc_bytes = oc * types::data_type_size(dst_data_type);
        if (is_src_nhwc && (ic_bytes % 32 != 0 || oc_bytes % 32 != 0))
            return status::unimplemented;

#if DNNL_WITH_XE_HPC
        if (hw >= ngen::HW::Xe_HPC) {
            // small minibatch and NHWC layout requires new messages support
            // which is currently not yet implemented
            if (is_src_nhwc || is_dst_nhwc) return status::unimplemented;
            if (mb_thr_blk == 1) return status::unimplemented;
        }

        if (hw < ngen::HW::Xe_HPC)
#endif
            // Blocked large batch performance is slightly behind.
            if (!is_src_nhwc && mb >= 16) return status::unimplemented;

        return status::success;
    }

    status_t init_bwd_d(convolution_pd_t *conv_pd) {
        using namespace ir_utils;

        // First convolution is not supported.
        if (ic < 16) return status::unimplemented;

        // Set dispatch and kernel parameters.
        int mb_thr_blk = (mb < 16 ? 1 : 32);
        ic_thr_blk = 32;
        int iw_thr_blk = (mb < 16 ? 16 : 1);
        if (iw < iw_thr_blk) iw_thr_blk = 8;

#ifdef GEN_CONV_DEBUG
        mb_thr_blk = getenv_int("mb_thr_blk", mb_thr_blk);
        ic_thr_blk = getenv_int("ic_thr_blk", ic_thr_blk);
        iw_thr_blk = getenv_int("iw_thr_blk", iw_thr_blk);
#endif

        regs = 256;

        tg_grid_dim[0] = std::min(4, utils::div_up(ic, ic_thr_blk));
        tg_grid_dim[1] = std::min(4, utils::div_up(iw, iw_thr_blk));
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
        ic_tg_blk = tg_grid_dim[0] * ic_thr_blk;
        iw_tg_blk = tg_grid_dim[1] * iw_thr_blk;
        oc_blk = (is_s32_accumulator() ? 32 : 16);

#ifdef GEN_CONV_DEBUG
        mb_tg_blk = getenv_int("mb_tg_blk", mb_tg_blk);
        ic_tg_blk = getenv_int("ic_tg_blk", ic_tg_blk);
        iw_tg_blk = getenv_int("iw_tg_blk", iw_tg_blk);
#endif

        m_tg_blk = mb_tg_blk * iw_tg_blk;
        n_tg_blk = ic_tg_blk;
        k_tg_blk = oc_blk;

        int mb_tg_padded = utils::rnd_up(mb, mb_tg_blk);
        int ic_tg_padded = utils::rnd_up(ic, ic_tg_blk);
        int iw_tg_padded = utils::rnd_up(iw, iw_tg_blk);

        int mb_tg_dim = mb_tg_padded / mb_tg_blk;
        int ic_tg_dim = ic_tg_padded / ic_tg_blk;

        iw_tg_dim = iw_tg_padded / iw_tg_blk;

        kernel_grid_dim[0] = ic_tg_dim;
        kernel_grid_dim[1] = id * ih * iw_tg_dim;
        kernel_grid_dim[2] = mb_tg_dim;

        CHECK(init_common_config());

        // Do not perform full unrolling when there are too many inner
        // iterations.
        if (kd * kh * kw > 9) do_loop_unroll = false;

        fixup_inference_consistency();
        if (!try_reduce_grf_usage()) return status::unimplemented;

#if DNNL_WITH_XE_HPC
        assert(hw != ngen::HW::Unknown);
        const bool is_wei16bXa = hw >= ngen::HW::Xe_HPC;
#else
        const bool is_wei16bXa = false;
#endif

        std::string src_tag;
        std::string wei_tag;
        std::string dst_tag;
        if (!is_s32_accumulator()) {
            src_tag = (mb_thr_blk == 1 ? "aBx16b" : "ABx32a16b");
            wei_tag = is_wei16bXa ? "BAx2b8a16b2a" : "BAx4b8a8b2a";
            dst_tag = (mb_thr_blk == 1 ? "aBx16b" : "ABx32a16b");
        } else {
            src_tag = (mb_thr_blk == 1 ? "aBx32b" : "ABx32a32b");
            wei_tag = is_wei16bXa ? "BAx2b8a16b4a" : "BAx4b8a8b4a";
            dst_tag = (mb_thr_blk == 1 ? "aBx32b" : "ABx32a32b");
        }

        if (with_groups) wei_tag = prepend_groups_to_tag(wei_tag);

#ifdef GEN_CONV_DEBUG
        src_tag = getenv_str("stag", src_tag);
        wei_tag = getenv_str("wtag", wei_tag);
        dst_tag = getenv_str("dtag", dst_tag);
#endif

        auto &src_md = *conv_pd->invariant_src_md();
        auto &wei_md = *conv_pd->invariant_wei_md();
        auto &dst_md = *conv_pd->invariant_dst_md();

        // Select layouts.
        src_layout = init_layout(src_md, src_tag);
        wei_layout = init_layout(wei_md, wei_tag);
        dst_layout = init_layout(dst_md, dst_tag);

        // Validate layouts.
        bool is_src_nhwc = (orig_src_mdw().is_plain()
                && src_layout == make_layout(src_md, "axb"));
        bool is_dst_nhwc = (orig_dst_mdw().is_plain()
                && dst_layout == make_layout(dst_md, "axb"));
        if (is_src_nhwc != is_dst_nhwc) return status::unimplemented;

        if (!is_src_nhwc && src_layout != make_layout(src_md, src_tag))
            return status::unimplemented;
        if (!is_dst_nhwc && dst_layout != make_layout(dst_md, dst_tag))
            return status::unimplemented;

        if (wei_layout != make_layout(wei_md, wei_tag))
            return status::unimplemented;

        // HWord loads require 32 byte alignment. For NHWC layout it means
        // input/output channels must be multiples of 32 bytes.
        size_t ic_bytes = ic * types::data_type_size(src_data_type);
        size_t oc_bytes = oc * types::data_type_size(dst_data_type);
        if (is_dst_nhwc && (ic_bytes % 32 != 0 || oc_bytes % 32 != 0))
            return status::unimplemented;

#if DNNL_WITH_XE_HPC
        if (hw >= ngen::HW::Xe_HPC) {
            // small minibatch and NHWC layout requires new messages support
            // which is currently not yet implemented
            if (is_src_nhwc || is_dst_nhwc) return status::unimplemented;
            if (mb_thr_blk == 1) return status::unimplemented;
        }

        if (hw < ngen::HW::Xe_HPC)
#endif
            // Blocked large batch performance is slightly behind.
            if (!is_src_nhwc && mb >= 16) return status::unimplemented;

        return status::success;
    }

    status_t init_bwd_w(convolution_pd_t *conv_pd) {
        using namespace ir_utils;

        // First convolution is not supported.
        if (ic < 16) return status::unimplemented;
        if (mb < 16) return status::unimplemented;

        oc_thr_blk = (oc <= 16 ? 16 : 32);
        ic_thr_blk = (ic <= 16 ? 16 : 32);

#ifdef GEN_CONV_DEBUG
        oc_thr_blk = getenv_int("oc_thr_blk", oc_thr_blk);
        ic_thr_blk = getenv_int("ic_thr_blk", ic_thr_blk);
#endif

        simd_size = 8;
        regs = 256;
        tg_grid_dim[0] = std::min(4, utils::div_up(oc, oc_thr_blk));
        tg_grid_dim[1] = std::min(4, utils::div_up(ic, ic_thr_blk));
        tg_grid_dim[2] = 1;

        // Round down to a power of 2.
        tg_grid_dim[0] = (1 << math::ilog2q(tg_grid_dim[0]));
        tg_grid_dim[1] = (1 << math::ilog2q(tg_grid_dim[1]));
        tg_grid_dim[2] = (1 << math::ilog2q(tg_grid_dim[2]));

#ifdef GEN_CONV_DEBUG
        tg_grid_dim[0] = getenv_int("tg0", tg_grid_dim[0]);
        tg_grid_dim[1] = getenv_int("tg1", tg_grid_dim[1]);
#endif

        oc_tg_blk = tg_grid_dim[0] * oc_thr_blk;
        ic_tg_blk = tg_grid_dim[1] * ic_thr_blk;
        mb_blk = 16;
        mb_tg_blk = 32;

        init_bwd_w_spatial_blocks();

        m_tg_blk = ic_tg_blk;
        n_tg_blk = oc_tg_blk;
        k_tg_blk = mb_blk;

        int oc_tg_padded = utils::rnd_up(oc, oc_tg_blk);
        int ic_tg_padded = utils::rnd_up(ic, ic_tg_blk);
        int mb_tg_padded = utils::rnd_up(mb, mb_tg_blk);
        int od_tg_padded = utils::rnd_up(od, od_tg_blk);
        int oh_tg_padded = utils::rnd_up(oh, oh_tg_blk);
        int ow_tg_padded = utils::rnd_up(ow, ow_tg_blk);

        oc_tg_dim = oc_tg_padded / oc_tg_blk;
        ic_tg_dim = ic_tg_padded / ic_tg_blk;

        mb_tg_dim = mb_tg_padded / mb_tg_blk;
        od_tg_dim = od_tg_padded / od_tg_blk;
        oh_tg_dim = oh_tg_padded / oh_tg_blk;
        ow_tg_dim = ow_tg_padded / ow_tg_blk;

        kernel_grid_dim[0] = oc_tg_dim;
        kernel_grid_dim[1]
                = ic_tg_dim * kd * kh * kw * od_tg_dim * oh_tg_dim * ow_tg_dim;
        kernel_grid_dim[2] = mb_tg_dim;

        CHECK(init_common_config());

        // Set BWD_W-specific settings.
        do_b_reduction = with_bias;
        do_loop_unroll = false;
        allow_grf_reorder = true;
        zero_out_output = true;
        do_atomic_update = true;
        do_post_wei_reorder = (wei_data_type == data_type::bf16);
        do_post_bia_reorder = (with_bias && bia_data_type == data_type::bf16);

        fixup_inference_consistency();
        if (!try_reduce_grf_usage()) return status::unimplemented;

        std::string src_tag;
        std::string wei_tag;
        std::string dst_tag;
        src_tag = "ABx32a16b";
        wei_tag = "ABx16b16a";
        dst_tag = "ABx32a16b";

        if (with_groups) wei_tag = prepend_groups_to_tag(wei_tag);

#ifdef GEN_CONV_DEBUG
        src_tag = getenv_str("stag", src_tag);
        wei_tag = getenv_str("wtag", wei_tag);
        dst_tag = getenv_str("dtag", dst_tag);
#endif

        auto &src_md = *conv_pd->invariant_src_md();
        auto &wei_md = *conv_pd->invariant_wei_md();
        auto &dst_md = *conv_pd->invariant_dst_md();
        auto &bia_md = *conv_pd->invariant_bia_md();

        // Select layouts.
        src_layout = init_layout(src_md, src_tag);
        wei_layout = init_layout(wei_md, wei_tag);
        dst_layout = init_layout(dst_md, dst_tag);
        if (with_bias) bia_layout = init_layout(bia_md, "a");

        if (src_layout != make_layout(src_md, src_tag))
            return status::unimplemented;
        if (wei_layout != make_layout(wei_md, wei_tag))
            return status::unimplemented;
        if (dst_layout != make_layout(dst_md, dst_tag))
            return status::unimplemented;

        if (do_post_wei_reorder) {
            wei_layout = wei_layout.retype(type_t::f32());
            orig_wei_md.data_type = data_type::f32;
        }
        if (do_post_bia_reorder) {
            bia_layout = bia_layout.retype(type_t::f32());
            orig_bia_md.data_type = data_type::f32;
        }

        return status::success;
    }

    void init_bwd_w_spatial_blocks() {
        od_tg_blk = 1;
        oh_tg_blk = 1;
        ow_tg_blk = 1;
        int sp_min_blk = 24;
        int sp_max_blk = 64;

        auto get_score = [&](int oh_blk, int ow_blk) {
            int sp_blk = oh_blk * ow_blk;
            int oh_padded = utils::rnd_up(oh, oh_blk);
            int ow_padded = utils::rnd_up(ow, ow_blk);

            double extra_work
                    = (oh_padded * ow_padded - oh * ow) / double(oh * ow);
            // ohw_eff == 0: no useful computation
            // ohw_eff == 1: all computation is useful
            double ohw_eff = 1 - std::min(extra_work, 1.0);
            int score = int(ohw_eff * 1000);
            // Prefer [sp_min_blk; sp_max_blk] range for the total spatial size.
            if (sp_blk >= sp_min_blk && sp_blk <= sp_max_blk) score += 100;
            return score;
        };

        int max_score = 0;
        for (int oh_blk = 1; oh_blk <= sp_max_blk; oh_blk++) {
            for (int ow_blk = 1; ow_blk <= sp_max_blk; ow_blk++) {
                int score = get_score(oh_blk, ow_blk);
                if (score > max_score) {
                    oh_tg_blk = oh_blk;
                    ow_tg_blk = ow_blk;
                    max_score = score;
                }
            }
        }

#ifdef GEN_CONV_DEBUG
        od_tg_blk = getenv_int("od_tg_blk", od_tg_blk);
        oh_tg_blk = getenv_int("oh_tg_blk", oh_tg_blk);
        ow_tg_blk = getenv_int("ow_tg_blk", ow_tg_blk);
#endif
    }

    status_t init_common_config() {
        using namespace ir_utils;

        do_b_reduction = false;
        use_a_slm = true;
        use_b_slm = true;
        pad_slm = true;
        assign_sbids
                = utils::one_of(fma_kind, fma_kind_t::dpas, fma_kind_t::dpasw);
        slm_bufs = hw <= ngen::HW::Xe_LP
                ? 0
                : (tg_grid_dim[0] * tg_grid_dim[1] <= 8 ? 2 : 3);
        gmem_bufs = 2;
        do_loop_unroll = hw > ngen::HW::Xe_LP;
        reduce_grf_usage = true;
        allow_grf_reorder = false;
        zero_out_output = false;
        do_atomic_update = false;
        reuse_headers = hw <= ngen::HW::Xe_LP;
        do_post_wei_reorder = false;
        do_post_bia_reorder = false;
        a_sub_tiles = 1;
        b_sub_tiles = 1;

#ifdef GEN_CONV_DEBUG
        use_a_slm = getenv_bool("use_a_slm", use_a_slm);
        use_b_slm = getenv_bool("use_b_slm", use_b_slm);
        pad_slm = getenv_bool("pad_slm", pad_slm);
        assign_sbids = getenv_bool("assign_sbids", assign_sbids);
        slm_bufs = getenv_int("slm_bufs", slm_bufs);
        gmem_bufs = getenv_int("gmem_bufs", gmem_bufs);
        do_loop_unroll = getenv_bool("do_loop_unroll", do_loop_unroll);
        reduce_grf_usage = getenv_bool("reduce_grf_usage", reduce_grf_usage);
        allow_grf_reorder = getenv_bool("allow_grf_reorder", allow_grf_reorder);
        reuse_headers = getenv_bool("reuse_headers", reuse_headers);
        a_sub_tiles = getenv_int("a_sub_tiles", a_sub_tiles);
        b_sub_tiles = getenv_int("b_sub_tiles", b_sub_tiles);
#endif

        simd_size = fma_kind::get_simd_size(
                hw, fma_kind, a_data_type, b_data_type, acc_data_type);

        return status::success;
    }

    bool post_ops_ok(const convolution_pd_t *pd) const {
        auto *attr = pd->attr();

        if (is_fwd || is_bwd_d) {
            auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops
                    | primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::sum_dt;
            if (!attr->has_default_values(attr_skip_mask)) return false;
        } else {
            if (!attr->has_default_values()) return false;
        }

        if (!attr->output_scales_.has_default_values()) {
            // Only common and per_oc output scales were tested.
            if (!utils::one_of(attr->output_scales_.mask_, 0, (1 << 1)))
                return false;
        }
        for (int i = 0; i < attr->post_ops_.len(); i++) {
            auto &po = attr->post_ops_.entry_[i];
            if (po.is_eltwise()) {
                if (!jit_eltwise_injector_f32_is_supported(po.eltwise.alg))
                    return false;
            } else if (po.is_binary()) {
                // Division is not supported.
                if (po.binary.alg == alg_kind::binary_div) return false;
                int mask = utils::get_dims_mask(pd->invariant_dst_md()->dims,
                        po.binary.src1_desc.dims, ndims);
                // per_oc broadcast is always supported.
                if ((mask & (1 << 1)) == 0) continue;
                auto rhs_layout = make_layout(po.binary.src1_desc);
                auto rhs0 = rhs_layout.blocks()[0];
                int block_bytes = rhs0.block * rhs_layout.type().size();
                // Innermost block must:
                // - be across output channels
                // - be dense
                // - aligned to 32 bytes (for HWord loads)
                if (rhs0.dim_idx != 1 || dim_t(rhs0.stride) != 1
                        || block_bytes % 32 != 0)
                    return false;
            }
        }
        return true;
    }

    bool hw_ok(const engine_t *engine) const {
        auto *compute_engine
                = utils::downcast<const compute::compute_engine_t *>(engine);
        if (regs == 256 && !compute_engine->mayiuse_large_grf_mode())
            return false;
        return true;
    }

    bool data_types_ok() const {
        if (is_fwd) { return true; }
        if (is_bwd_d) {
            if (utils::one_of(data_type::f32, dst_data_type, wei_data_type))
                return false;
            return true;
        }
        if (is_bwd_w) {
            bool ok = true;
            ok &= (src_data_type == data_type::bf16);
            ok &= (dst_data_type == data_type::bf16);
            ok &= utils::one_of(wei_data_type, data_type::bf16, data_type::f32);
            if (with_bias) {
                ok &= utils::one_of(
                        bia_data_type, data_type::bf16, data_type::f32);
            }
            return ok;
        }
        return false;
    }

    bool is_s32_accumulator() const { return acc_data_type == data_type::s32; }

    int grf_size() const { return ngen::GRF::bytes(hw); }

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
        oss << "  OD TG block:                " << od_tg_blk << std::endl;
        oss << "  OH TG block:                " << oh_tg_blk << std::endl;
        oss << "  OW TG block:                " << ow_tg_blk << std::endl;
        oss << "  OC TG block:                " << oc_tg_blk << std::endl;
        oss << "  Kernel grid:                " << make_seq_print_helper(kernel_grid_dim, " x ") << std::endl;
        oss << "  Thread group:               " << make_seq_print_helper(tg_grid_dim, " x ") << std::endl;
        oss << "  FMA kind:                   " << fma_kind::to_string(fma_kind) << std::endl;
        oss << "  Use SLM for A:              " << to_string(use_a_slm) << std::endl;
        oss << "  Use SLM for B:              " << to_string(use_b_slm) << std::endl;
        oss << "  Pad SLM:                    " << to_string(pad_slm) << std::endl;
        oss << "  Assign SBIDs:               " << to_string(assign_sbids) << std::endl;
        oss << "  SLM buffers:                " << slm_bufs << std::endl;
        oss << "  GMEM to SLM, GRF buffers:   " << gmem_bufs << std::endl;
        oss << "  Reduce GRF usage:           " << to_string(reduce_grf_usage) << std::endl;
        oss << "  Reuse headers:              " << to_string(reuse_headers) << std::endl;
        oss << "  Allow GRF reorder:          " << to_string(allow_grf_reorder) << std::endl;
        oss << "  A sub-tiles:                " << a_sub_tiles << std::endl;
        oss << "  B sub-tiles:                " << b_sub_tiles << std::endl;
        // clang-format on
        return oss.str();
    }

    data_type_t a_data_type;
    data_type_t b_data_type;
    data_type_t c_data_type;
    data_type_t acc_data_type;

    ngen::HW hw = ngen::HW::Unknown;
    int simd_size; // SIMD width.
    int regs; // Number of registers.

    // Thread group dimensions (thread group grid).
    std::array<int, 3> tg_grid_dim;

    // Number of thread groups across dimensions (kernel grid).
    std::array<int, 3> kernel_grid_dim;

    int mb_tg_dim;
    int od_tg_dim;
    int oh_tg_dim;
    int ow_tg_dim;
    int iw_tg_dim;
    int oc_tg_dim;
    int ic_tg_dim;

    // Block sizes per thread group (convolution notation).
    int ic_tg_blk;
    int iw_tg_blk;
    int mb_tg_blk;
    int oc_tg_blk;
    int od_tg_blk;
    int oh_tg_blk;
    int ow_tg_blk;

    // Block sizes per iteration.
    int mb_blk;
    int ic_blk;
    int oc_blk;

    int oc_thr_blk;
    int ic_thr_blk;

    // Block sizes per thread group (GEMM notation).
    int m_tg_blk;
    int n_tg_blk;
    int k_tg_blk;

    bool do_b_reduction;

    fma_kind_t fma_kind; // Which instruction backend to use.

    bool use_a_slm; // Whether to use SLM for A.
    bool use_b_slm; // Whether to use SLM for B.
    bool pad_slm; // Whether to pad SLM to avoid write conflicts.
    bool assign_sbids; // Whether to manually assign SBID tokens.
    int slm_bufs; // Number of SLM buffers to use.
    int gmem_bufs; // Number of GRF buffers to use for GMEM -> SLM copy.
    bool do_loop_unroll; // Whether to fully unroll inner loops.
    bool reduce_grf_usage; // Whether to try to reduce GRF usage based on heuristics.
    bool allow_grf_reorder; // Whether to allow GRF reorders to FMA-friendly layouts.
    bool zero_out_output; // Whether to zero out outputs before the main kernel.
    bool do_atomic_update; // Whether to use atomics during C update.
    bool reuse_headers; // Whether to reuse header messages to reduce GRF usage.

    // Specific to BWD_W.
    bool do_post_bia_reorder; // Whether to perform extra reorder for weights.
    bool do_post_wei_reorder; // Whether to perform extra reorder for bias.

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
    status_t init_hw(engine_t *engine) {
        using namespace compute;

        auto compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        auto device_info = compute_engine->device_info();

        switch (device_info->gpu_arch()) {
            case gpu_arch_t::gen9: hw = ngen::HW::Gen9; break;
            case gpu_arch_t::xe_lp: hw = ngen::HW::Xe_LP; break;
#if DNNL_WITH_XE_HP
            case gpu_arch_t::xe_hp: hw = ngen::HW::Xe_HP; break;
#endif
#if DNNL_WITH_XE_HPG
            case gpu_arch_t::xe_hpg: hw = ngen::HW::Xe_HPG; break;
#endif
#if DNNL_WITH_XE_HPC
            case gpu_arch_t::xe_hpc: hw = ngen::HW::Xe_HPC; break;
#endif
            default: return status::unimplemented;
        }
        return status::success;
    }

    // Initializes A/B/C data types (GEMM notation: C += A * B) according to
    // the following convention:
    // FWD:        src -> A,      wei -> B,      dst -> C
    // BWD_D: diff_dst -> A,      wei -> B, diff_src -> C
    // BWD_W:      src -> A, diff_dst -> B, diff_wei -> C
    status_t init_abc_data_types() {
        if (is_fwd) {
            a_data_type = src_data_type;
            b_data_type = wei_data_type;
            c_data_type = dst_data_type;
        } else if (is_bwd_d) {
            a_data_type = dst_data_type;
            b_data_type = wei_data_type;
            c_data_type = src_data_type;
        } else if (is_bwd_w) {
            a_data_type = src_data_type;
            b_data_type = dst_data_type;
            // Always use f32 for accumulation/storing in the main kernel.
            c_data_type = data_type::f32;
        } else {
            ir_error_not_expected();
        }
        return status::success;
    }

    status_t init_acc_data_type() {
        auto a = a_data_type;
        auto b = b_data_type;
        auto c = c_data_type;
        if (utils::one_of(a, data_type::s8, data_type::u8)
                && utils::one_of(b, data_type::s8, data_type::u8)) {
            acc_data_type = data_type::s32;
            return status::success;
        }
        if (utils::everyone_is(data_type::f16, a, b)
                || utils::everyone_is(data_type::bf16, a, b)) {
            acc_data_type = data_type::f32;
            return status::success;
        }
        if (utils::everyone_is(data_type::f32, a, b, c)) {
            acc_data_type = data_type::f32;
            return status::success;
        }
        return status::unimplemented;
    }

    status_t init_fma_kind() {
        fma_kind = fma_kind::get_supported_kind(
                hw, a_data_type, b_data_type, acc_data_type);
#ifdef GEN_CONV_DEBUG
        fma_kind = fma_kind::from_string(ir_utils::getenv_str(
                "fma_kind", fma_kind::to_string(fma_kind)));
#endif
        if (fma_kind == fma_kind_t::unknown) return status::unimplemented;

        // Disable using mad instruction backend until performance parity is
        // reached with OpenCL kernels.
        if (fma_kind == fma_kind_t::mad && (!is_fwd || hw < ngen::HW::Xe_HP))
            return status::unimplemented;

        return status::success;
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
        // Downgrade dpasw -> dpas for some cases.
        if (fma_kind == fma_kind_t::dpasw) {
            // dpasw is executed by fused EUs (across X thread group
            // dimension). Do not use dpasw if X is uneven.
            if (tg_grid_dim[0] % 2 != 0) fma_kind = fma_kind_t::dpas;
            // dpasw can't be generated in case of direct load from GMEM and reorder.
            if (is_bwd_w && allow_grf_reorder && (!use_a_slm || !use_b_slm))
                fma_kind = fma_kind_t::dpas;
        }

        // Can't reuse headers with loop unroll and post-increment offset updates.
        if (reuse_headers) do_loop_unroll = false;

        if (!do_loop_unroll) {
            gmem_bufs = 1;
            // Double/triple SLM buffering is not supported when only one
            // matrix is SLM-buffered.
            if (use_a_slm != use_b_slm) slm_bufs = std::min(1, slm_bufs);
        }
    }

    bool try_reduce_grf_usage() {
        if (!reduce_grf_usage) return true;

        int max_regs = int(regs * 0.95);
        int regs = estimate_register_count();
        if (regs <= max_regs) return true;

        // Try to reduce disable GRF buffering.
        if (gmem_bufs != 1) {
            gmem_bufs = 1;
            int regs = estimate_register_count();
            if (regs <= max_regs) return true;
        }

        // Try to use sub-tiles for B.
        int n_thr_blk = utils::div_up(n_tg_blk, tg_grid_dim[0]);
        int max_b_sub_tiles
                = std::min((use_b_slm ? 4 : 2), n_thr_blk / simd_size);
#if DNNL_WITH_XE_HPC
        // XXX: avoid layout mismatch for B loads
        if (hw >= ngen::HW::Xe_HPC && is_bwd_w) max_b_sub_tiles = 2;
#endif
        while (b_sub_tiles < max_b_sub_tiles) {
            b_sub_tiles *= 2;
            int regs = estimate_register_count();
            if (regs <= max_regs) return true;
        }

        // Try to use double SLM buffering.
        if (slm_bufs == 3) {
            slm_bufs = 2;
            int regs = estimate_register_count();
            if (regs <= max_regs) return true;
        }

        // Try to use single SLM buffering.
        if (slm_bufs == 2) {
            slm_bufs = 1;
            int regs = estimate_register_count();
            if (regs <= max_regs) return true;
        }

        // Last resort settings to reduce GRF usage.
        reuse_headers = true;
        do_loop_unroll = false;

        return estimate_register_count() <= max_regs;
    }

    int estimate_register_count() const {
        int reg_bytes = ngen::GRF::bytes(hw);
        int gmem_msg_bytes = reg_bytes; // Assume 1 register per GMEM load.
        int slm_msg_bytes
                = 8 * reg_bytes; // Assume 8 registers per SLM load/store.

        int nthr = tg_grid_dim[0] * tg_grid_dim[1];
        int m_thr_blk = utils::div_up(m_tg_blk, tg_grid_dim[1]);
        int n_thr_blk = utils::div_up(n_tg_blk, tg_grid_dim[0]);
        int k_thr_blk = k_tg_blk;

        int a_size = int(types::data_type_size(a_data_type));
        int b_size = int(types::data_type_size(b_data_type));
        int acc_size = int(types::data_type_size(acc_data_type));

        // Registers for C += A * B operation.
        int a_bytes
                = utils::div_up(m_thr_blk * k_thr_blk * a_size, a_sub_tiles);
        int b_bytes
                = utils::div_up(k_thr_blk * n_thr_blk * b_size, b_sub_tiles);
        int acc_bytes = m_thr_blk * n_thr_blk * acc_size;

        int a_regs = utils::div_up(a_bytes, reg_bytes);
        int b_regs = utils::div_up(b_bytes, reg_bytes);
        int acc_regs = utils::div_up(acc_bytes, reg_bytes);

        int a_headers = utils::div_up(
                a_bytes, use_a_slm ? slm_msg_bytes : gmem_msg_bytes);
        int b_headers = utils::div_up(
                b_bytes, use_b_slm ? slm_msg_bytes : gmem_msg_bytes);

        if (fma_kind == fma_kind_t::dpasw) {
            // dpasw reuses registers between fused threads across tg0. M is
            // split across tg1, N is split across tg0 so dpasw allows to share
            // matrix A which is is (M x K).
            a_regs = utils::div_up(a_regs, 2);
            a_headers = utils::div_up(a_headers, 2);
        }

        // Temporary registers for GMEM -> SLM load.
        int a_g2s_bytes
                = (use_a_slm ? utils::div_up(m_tg_blk * k_tg_blk * a_size, nthr)
                             : 0);
        int b_g2s_bytes
                = (use_b_slm ? utils::div_up(k_tg_blk * n_tg_blk * b_size, nthr)
                             : 0);

        int a_g2s_regs = utils::div_up(a_g2s_bytes, reg_bytes);
        int b_g2s_regs = utils::div_up(b_g2s_bytes, reg_bytes);

        // Two sets of headers for GMEM -> GRF and GRF -> SLM.
        int a_g2s_headers = utils::div_up(a_g2s_bytes, gmem_msg_bytes)
                + utils::div_up(a_g2s_bytes, slm_msg_bytes);
        int b_g2s_headers = utils::div_up(b_g2s_bytes, gmem_msg_bytes)
                + utils::div_up(b_g2s_bytes, slm_msg_bytes);

        // Extra registers for GRF <-> GRF reorders.
        int reorder_regs = 0;

        // Assume A/B need reorders to temporary buffers.
        if (is_bwd_w) {
            if (use_a_slm) {
                a_g2s_regs *= 2;
            } else {
                a_regs *= 2;
            }
            if (use_b_slm) {
                b_g2s_regs *= 2;
            } else {
                b_regs *= 2;
            }
            // Hardcode for now, this is the upper bound for the temporary
            // buffer size for BWD_W.
            int bwd_w_reorder_regs = 16;
            reorder_regs += bwd_w_reorder_regs;
        }

        int g2s_regs = gmem_bufs * (a_g2s_regs + b_g2s_regs);
        int g2s_headers = a_g2s_headers + b_g2s_headers;

        int data_regs = a_regs + b_regs + acc_regs + g2s_regs;
        int header_regs = a_headers + b_headers + g2s_headers;
        if (reuse_headers) header_regs = 1;

        int estimated_regs = data_regs + reorder_regs + header_regs;

        return estimated_regs;
    }

    static std::string prepend_groups_to_tag(const std::string &tag) {
        auto ret = tag;
        for (auto &c : ret) {
            bool is_lower_dim = ('a' <= c && c < 'a' + DNNL_MAX_NDIMS);
            bool is_upper_dim = ('A' <= c && c < 'A' + DNNL_MAX_NDIMS);
            if (!is_lower_dim && !is_upper_dim) continue;
            c += 1;
        }
        return "a" + ret;
    }

    static layout_t init_layout(memory_desc_t &md, const std::string &tag) {
        if (md.format_kind != format_kind::any) return make_layout(md);
        auto ret = make_layout(md, tag);
        md = ret.to_dnnl(md.dims);
        return ret;
    }

    static layout_t make_layout(const memory_desc_t &md) {
        return layout_t(md, /*do_normalize=*/false);
    }

    static layout_t make_layout(
            const memory_desc_t &md, const std::string &tag) {
        return layout_t(md, tag, /*do_normalize=*/false);
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

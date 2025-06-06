/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
* Copyright 2022 Arm Ltd. and affiliates
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

#ifndef CPU_MATMUL_MATMUL_UTILS_HPP
#define CPU_MATMUL_MATMUL_UTILS_HPP

#include "common/memory_desc_wrapper.hpp"
#include "common/tag_traits.hpp"
#include "common/utils.hpp"

#include "cpu/binary_injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace matmul {

struct matmul_helper_t {
    using mdw_t = const memory_desc_wrapper;

    matmul_helper_t(mdw_t &src_md, mdw_t &weights_md, mdw_t &dst_md)
        : src_md_(src_md), weights_md_(weights_md), dst_md_(dst_md) {}

    int ndims() const { return dst_md_.ndims(); }
    bool batched() const { return ndims() > 2; }

    dim_t batch() const { return get_batch_size(dst_md_); };
    dim_t src_batch() const { return get_batch_size(src_md_); };
    dim_t wei_batch() const { return get_batch_size(weights_md_); };

    dim_t M() const { return dst_md_.dims()[ndims() - 2]; }
    dim_t N() const { return dst_md_.dims()[ndims() - 1]; }
    dim_t K() const { return src_md_.dims()[ndims() - 1]; }

    char transA() const {
        const auto &strides = &src_md_.blocking_desc().strides[ndims() - 2];
        return (strides[1] == 1 && src_md_.dims()[ndims() - 2] > 1) ? 'N' : 'T';
    }

    char transB() const {
        const auto &strides = &weights_md_.blocking_desc().strides[ndims() - 2];
        return (strides[1] == 1 && weights_md_.dims()[ndims() - 2] > 1) ? 'N'
                                                                        : 'T';
    }

    char transC() const {
        const auto &strides = &dst_md_.blocking_desc().strides[ndims() - 2];
        return strides[1] == 1 ? 'N' : 'T';
    }

    dim_t lda() const {
        const auto &strides = &src_md_.blocking_desc().strides[ndims() - 2];
        return strides[transA() == 'N' ? 0 : 1];
    }

    dim_t get_a_stride(int dim) const {
        if (dim >= ndims() || dim < 0) return 0;
        return src_md_.blocking_desc().strides[dim];
    }

    dim_t ldb() const {
        const auto &strides = &weights_md_.blocking_desc().strides[ndims() - 2];
        return strides[transB() == 'N' ? 0 : 1];
    }

    dim_t get_b_stride(int dim) const {
        if (dim >= ndims() || dim < 0) return 0;
        return weights_md_.blocking_desc().strides[dim];
    }

    dim_t ldc() const {
        const auto &strides = &dst_md_.blocking_desc().strides[ndims() - 2];
        return strides[transC() == 'N' ? 0 : 1];
    }

    dim_t get_c_stride(int dim) const {
        if (dim >= ndims() || dim < 0) return 0;
        return dst_md_.blocking_desc().strides[dim];
    }

    bool use_single_gemm_call_optimization(const post_ops_t &post_ops) {
        using namespace binary_injector_utils;
        bool is_binary_po_per_oc;
        bool is_binary_po_per_oc_sp;
        bool is_binary_po_channel_bcast;
        std::tie(is_binary_po_per_oc, is_binary_po_per_oc_sp,
                is_binary_po_channel_bcast)
                = bcast_strategies_present_tup(post_ops.entry_, dst_md_,
                        broadcasting_strategy_t::per_oc,
                        broadcasting_strategy_t::per_oc_spatial,
                        broadcasting_strategy_t::per_mb_spatial);

        const bool can_use_po_with_fused_batch = !is_binary_po_channel_bcast
                && IMPLICATION(is_binary_po_per_oc || is_binary_po_per_oc_sp,
                        ndims() == 2);

        // single GeMM call can be made, avoid parallelization over GeMM calls
        return can_use_po_with_fused_batch && can_fuse_src_batch_dims();
    }

    // Read note (3-4) in the "can_fuse_src_batch_dims()" method below.
    bool is_src_dst_layout_batch_fusable() const {
        // determine batch dims layout
        dims_t src_strides;
        const int batch_ndims = ndims() - 2;
        utils::array_copy(
                src_strides, src_md_.blocking_desc().strides, batch_ndims);

        // compute ou_dims. It is required to get correct perm
        dims_t blocks = {0};
        src_md_.compute_blocks(blocks);
        dims_t ou_dims;
        for (int i = 0; i < batch_ndims; ++i)
            ou_dims[i] = src_md_.padded_dims()[i] / blocks[i];

        dims_t perm;
        for (int i = 0; i < batch_ndims; ++i)
            perm[i] = i;

        // permute batch dim idx by sorting based on strides.
        utils::simultaneous_sort(src_strides, ou_dims, perm, batch_ndims,
                [](stride_t a, stride_t b) { return a - b; });

        dim_t src_stride = lda() * (transA() == 'N' ? M() : K());
        dim_t dst_stride = ldc() * (transC() == 'N' ? M() : N());

        for (int i = 0; i < batch_ndims; ++i) {
            const dim_t dim_idx = perm[i];
            if (src_md_.blocking_desc().strides[dim_idx] != src_stride
                    || dst_md_.blocking_desc().strides[dim_idx] != dst_stride)
                return false;
            src_stride = src_stride * src_md_.dims()[dim_idx];
            dst_stride = dst_stride * dst_md_.dims()[dim_idx];
        }

        return true;
    }

    // TODO: consolidate these functions with ones in simple_reorder.hpp, as they
    // are copy-pasted, and address TODOs from there.
    static status_t get_quant_md(memory_desc_t &md, const int ndims,
            const dims_t in_dims, const int quant_mask, const dim_t g0,
            const dim_t g1, const data_type_t dt) {
        if (dt == data_type::undef || quant_mask < 0) {
            md = glob_zero_md;
            return status::success;
        }

        dims_t quant_dims {};
        utils::copy_dims_with_mask(quant_dims, in_dims, ndims, quant_mask,
                /* fill_with_ones = */ true);
        if (ndims >= 2) {
            quant_dims[ndims - 1] /= g1;
            quant_dims[ndims - 2] /= g0;
        }

        CHECK(memory_desc_init_by_tag(
                md, ndims, quant_dims, dt, get_abx_tag(ndims)));
        return status::success;
    }

    static dim_t get_quant_off(const dims_t &input_idx, const int ndims,
            const int quant_mask, const dim_t g0, const dim_t g1,
            const memory_desc_t &quant_md) {
        if (types::is_zero_md(&quant_md)) return 0;

        dims_t quant_idx {};
        utils::array_copy(quant_idx, input_idx, ndims);
        utils::apply_mask_on_dims(quant_idx, ndims, quant_mask);
        // Note: an `idx` must divide by a group value as grouped quantization
        // applies to consecutive points.
        // Using quant dimensions in `l_dims_by_l_offset` will lead to wrapping
        // around dimensions instead of applying consecutively.
        if (ndims >= 2) {
            quant_idx[ndims - 1] /= g1;
            quant_idx[ndims - 2] /= g0;
        }

        const memory_desc_wrapper q_mdw(quant_md);
        return q_mdw.off_v(quant_idx);
    }

private:
    mdw_t src_md_;
    mdw_t weights_md_;
    mdw_t dst_md_;

    // TODO similar optimization is also possible for wei batch fusion.
    bool can_fuse_src_batch_dims() const {
        /* Note:
            We can fuse src batch dims so that a single GeMM can be used if
            0. always for batch = 1 case
            1. src is not transposed
            2. wei batch dims are all 1's
            3. The strides in batch dims are trivial (allowing permutations).
            4. src and dst layout are identical. Example:
                src layout : {batch dim_idx permutations}xMxK
                dst layout : {identical batch dim_idx perm}xMxN;

            For example,
            src_layout : aXdXcXbXmXk
            wei_layout: 1X1X1X1xkxn or 1X1X1X1xnxk
            dst_layout : aXdXcXbXmXn

            A single GeMM call can be used instead with m = a*d*c*b*m
        */
        // Note 0:
        if (batch() == 1) return true;

        // Note 1:
        if (transA() == 'T') return false;

        // Note 2:
        if (wei_batch() != 1) return false;

        // Note 3-4:
        return is_src_dst_layout_batch_fusable();
    }
    dim_t get_batch_size(const mdw_t &tensor_md) const {
        int batch_dims = ndims() - 2;
        dim_t batch_size = 1;
        for (int b_idx = 0; b_idx < batch_dims; b_idx++) {
            dim_t batch_dim = tensor_md.dims()[b_idx];
            if (DNNL_RUNTIME_DIM_VAL == batch_dim) return DNNL_RUNTIME_DIM_VAL;

            batch_size *= batch_dim;
        }

        return batch_size;
    }
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif

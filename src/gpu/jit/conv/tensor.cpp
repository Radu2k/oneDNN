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

#include <cctype>
#include <regex>

#include "gpu/jit/conv/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

layout_t::layout_t(const type_t &type, const expr_t &offset,
        const std::string &format, const std::vector<dim_t> &dims)
    : type_(type), offset_(offset) {
    auto parts = parse_format(format, int(dims.size()));
    ndims_ = 0;
    for (auto &p : parts) {
        int dim_idx = p.first;
        dim_t block = p.second;
        ndims_ = std::max(ndims_, dim_idx + 1);
        if (block == 0 && dims.empty())
            ir_error_not_expected()
                    << "Dimensions are missing. Can't deduce them from "
                       "the format.";
    }
    if (!dims.empty() && ndims_ != int(dims.size())) {
        ir_error_not_expected() << "Format and dimensions do not match.";
    }

    dim_t stride = 1;
    // Iterate from left to right (innermost to outermost).
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
        int dim_idx = it->first;
        dim_t block = it->second;
        if (block == 0) {
            dim_t full_block = 1;
            for (auto &b : blocks_)
                if (b.dim_idx == dim_idx) full_block *= b.block;

            block = utils::div_up(dims[dim_idx], full_block);
        }

        blocks_.emplace_back(dim_idx, block, stride);
        stride = block * stride;
    }

    sanity_check();
}

layout_t::layout_t(const memory_desc_wrapper &mdw)
    : type_(mdw.data_type()), offset_(mdw.offset0()) {
    ir_assert(mdw.is_blocking_desc()) << "Expected blocking memory descriptor.";

    ndims_ = mdw.ndims();
    auto &blocking = mdw.blocking_desc();
    auto *padded_dims = mdw.padded_dims();

    dim_t stride = 1;
    std::vector<dim_t> full_blocks(ndims_, 1);
    for (int i = blocking.inner_nblks - 1; i >= 0; i--) {
        int dim_idx = blocking.inner_idxs[i];
        dim_t block = blocking.inner_blks[i];
        blocks_.emplace_back(dim_idx, block, stride);
        stride *= block;
        full_blocks[dim_idx] *= block;
    }

    for (int i = 0; i < ndims_; i++) {
        dim_t block = padded_dims[i] / full_blocks[i];
        blocks_.emplace_back(i, block, blocking.strides[i]);
    }

    // Sort outer blocks by their stride.
    std::sort(blocks_.begin() + blocking.inner_nblks, blocks_.end(),
            [](const block_t &a, const block_t &b) {
                if (a.stride == b.stride) return a.dim_idx > b.dim_idx;
                return a.stride < b.stride;
            });

    sanity_check();
}

memory_desc_t layout_t::to_dnnl(const dim_t *dims_hint) const {
    memory_desc_t md = {};
    md.ndims = ndims();
    std::copy(dims_hint, dims_hint + ndims(), md.dims);
    md.data_type = jit::to_dnnl(type_);
    md.offset0 = to_cpp<dim_t>(offset_);
    md.format_kind = format_kind::blocked;

    auto &blk = md.format_desc.blocking;
    bool seen[DNNL_MAX_NDIMS] = {};

    bool in_inner_block = false;
    dim_t prev_stride = 0;

    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        auto &b = *it;
        if (!seen[b.dim_idx]) {
            // Outer block.
            ir_assert(!in_inner_block);
            MAYBE_UNUSED(in_inner_block);
            blk.strides[b.dim_idx] = b.stride;
            md.padded_dims[b.dim_idx] = b.block;
        } else {
            // Inner block.
            md.padded_dims[b.dim_idx] *= b.block;
            blk.inner_idxs[blk.inner_nblks] = b.dim_idx;
            blk.inner_blks[blk.inner_nblks] = b.block;
            blk.inner_nblks++;
            if (prev_stride > 0) {
                // Inner block must be dense.
                ir_assert(prev_stride == b.block * b.stride);
            }
            prev_stride = b.stride;
            in_inner_block = true;
        }
        seen[b.dim_idx] = true;
    }

    return md;
}

format_tag_t layout_t::to_format_tag() const {
    auto desc = desc_str(/*dnnl_style=*/true);
#define CASE(tag) \
    if (desc == #tag) return format_tag::tag

    CASE(ABc32a16b);
    CASE(ABc32a32b);
    CASE(ABc4a8b8a2b);
    CASE(ABc4a8b8a4b);
    CASE(ABcd32a16b);
    CASE(ABcd32a32b);
    CASE(ABcd4a8b8a2b);
    CASE(ABcd4a8b8a4b);
    CASE(ABcde32a16b);
    CASE(ABcde32a32b);
    CASE(ABcde4a8b8a2b);
    CASE(ABcde4a8b8a4b);
    CASE(BAc16b16a);
    CASE(BAcd16b16a);
    CASE(BAcde16b16a);
    CASE(BAc4b8a8b2a);
    CASE(BAcd4b8a8b2a);
    CASE(BAcde4b8a8b2a);
    CASE(aBc16b);
    CASE(aBc32b);
    CASE(aBcd16b);
    CASE(aBcd32b);
    CASE(aBcde16b);
    CASE(aBcde32b);
    CASE(acb);
    CASE(acdb);
    CASE(acdeb);
    CASE(ABc2a8b16a2b);
    CASE(ABc2a8b16a4b);
    CASE(ABcd2a8b16a2b);
    CASE(ABcd2a8b16a4b);
    CASE(ABcde2a8b16a4b);
    CASE(ABcde2a8b16a2b);
    CASE(BAc2b8a16b2a);
    CASE(BAc2b8a16b4a);
    CASE(BAcd2b8a16b2a);
    CASE(BAcd2b8a16b4a);
    CASE(BAcde2b8a16b4a);
    CASE(BAcde2b8a16b2a);
    CASE(ABc16b16a);
    CASE(ABcd16b16a);
    CASE(ABcde16b16a);

#undef CASE

    ir_error_not_expected() << "Unknown tag: " << desc;
    return format_tag::undef;
}

layout_t layout_t::map(const tensor_t &tensor) const {
    if (ndims() != tensor.ndims())
        ir_error_not_expected() << "Dimensions do not match.";

    std::vector<dim_t> remaining_dims = tensor.dims();
    std::vector<block_t> mapped_blocks;

    for (auto &eb : enumerated_blocks()) {
        block_t &b = eb.second;
        bool b_is_outermost = is_outermost(eb);

        dim_t block = b.block;
        dim_t &rem_dim = remaining_dims[b.dim_idx];
        if (rem_dim == 1) {
            if (b_is_outermost) {
                // This is to have similarity between the current and
                // mapped layouts.
                mapped_blocks.emplace_back(b.dim_idx, 1, b.stride);
            }
            continue;
        }
        if (b_is_outermost) {
            block = rem_dim;
        } else if (rem_dim % block != 0) {
            // Try to split the current block and start mapping from
            // scratch.
            if (block % rem_dim == 0)
                return split_block(eb, rem_dim, block / rem_dim).map(tensor);

            ir_error_not_expected() << "Can't map tensor layout.";
        }
        rem_dim /= block;
        mapped_blocks.emplace_back(b.dim_idx, block, b.stride);
    }

    for (auto &d : remaining_dims) {
        ir_assert(d == 1) << "Can't map tensor layout.";
        MAYBE_UNUSED(d);
    }

    return layout_t(type(), ndims(), operator()(tensor.start()), mapped_blocks);
}

layout_t layout_t::reinterpret(const type_t &new_type) const {
    int old_size = type().size();
    int new_size = new_type.size();
    if (new_size == old_size) return *this;

    expr_t new_offset = 0;
    if (!has_zero_offset()) {
        ir_assert(is_const(offset_)) << "Expected constant offset.";
        int64_t off = to_cpp<int64_t>(offset_) * old_size;
        ir_assert(off % new_size == 0);
        new_offset = off / new_size;
    }

    if (old_size % new_size != 0 && new_size % old_size != 0)
        ir_error_not_expected();

    auto new_blocks = blocks_;
    if (new_size < old_size) {
        int factor = (old_size / new_size);
        auto &b0 = new_blocks.front();
        b0.block *= factor;
        // Recompute strides.
        for (auto &b : new_blocks) {
            if (&b == &b0) continue;
            b.stride *= factor;
        }
    } else {
        int factor = (new_size / old_size);
        auto &b0 = new_blocks.front();
        if (b0.block % factor != 0) ir_error_not_expected();
        b0.block /= factor;
        // Recompute strides.
        for (auto &b : new_blocks) {
            if (&b == &b0) continue;
            if (b.stride % factor != 0) ir_error_not_expected();
            b.stride /= factor;
        }
    }

    return layout_t(new_type, ndims(), new_offset, new_blocks);
}

layout_t layout_t::split_block(
        const std::pair<int, block_t> &eb, dim_t block0, dim_t block1) const {
    int block_idx = eb.first;
    auto &b = eb.second;
    ir_assert(b.block == block0 * block1) << "Incompatible block sizes.";
    MAYBE_UNUSED(b);

    auto new_blocks = blocks_;

    block_t &b0 = new_blocks[block_idx];
    block_t b1 = b0;

    b0.block = block0;
    b1.block = block1;
    b1.stride = b0.stride * block0;

    new_blocks.insert(new_blocks.begin() + block_idx + 1, b1);

    return layout_t(type(), ndims(), offset(), new_blocks);
}

layout_t layout_t::split_into_multi_blocks(
        const std::vector<dim_t> &multi_blocks) const {
    return split_into_multi_blocks_impl(multi_blocks, nullptr);
}

layout_t layout_t::split_into_multi_blocks_with_hint(
        std::vector<dim_t> &multi_blocks) const {
    return split_into_multi_blocks_impl(multi_blocks, &multi_blocks);
}

tensor_t layout_t::split_into_dense_tile(
        dim_t tile_elems, dim_t outer_block) const {
    stride_t dense_stride = 1;
    dim_t cur_tile_elems = 1;
    dim_t cur_outer_block = 1;
    bool in_tile = (tile_elems != 1);
    std::vector<dim_t> tile_dims(ndims(), 1);
    for (auto &b : blocks()) {
        if (b.block == 1) continue;
        if (in_tile) {
            if (b.stride.is_unknown()) return tensor_t();
            if (dense_stride != b.stride) return tensor_t();
            dense_stride = b.block * b.stride;
            cur_tile_elems *= b.block;
            tile_dims[b.dim_idx] *= b.block;
            ir_assert(cur_tile_elems <= tile_elems);
            if (cur_tile_elems == tile_elems) in_tile = false;
        } else {
            if (outer_block == 1) break;
            cur_outer_block *= b.block;
            tile_dims[b.dim_idx] *= b.block;
            ir_assert(cur_outer_block <= outer_block);
            if (cur_outer_block == outer_block) break;
        }
    }
    if (cur_tile_elems != tile_elems) return tensor_t();
    if (cur_outer_block != outer_block) return tensor_t();
    return tensor_t(tile_dims);
}

tensor_t layout_t::split_into_max_tile(
        dim_t max_tile_elems, bool is_dense_tile) const {
    stride_t dense_stride = 1;
    std::vector<dim_t> tile_dims(ndims(), 1);
    dim_t cur_elems = 1;
    for (auto &eb : enumerated_blocks()) {
        auto &b = eb.second;
        if (b.block == 1) continue;
        if (b.block * cur_elems <= max_tile_elems) {
            if (is_dense_tile) {
                if (b.stride.is_unknown()) return tensor_t();
                if (dense_stride != b.stride) return tensor_t();
                dense_stride = b.block * b.stride;
            }
            cur_elems *= b.block;
            tile_dims[b.dim_idx] *= b.block;
            continue;
        }
        dim_t max_block = utils::max_div(b.block, max_tile_elems / cur_elems);
        if (max_block == 1) break;
        auto tmp_layout = split_block(eb, max_block, b.block / max_block);
        return tmp_layout.split_into_max_tile(max_tile_elems, is_dense_tile);
    }
    return tensor_t(tile_dims);
}

void layout_t::align_layouts(layout_t &a, layout_t &b) {
    for (int i = 0; i < a.ndims(); i++) {
        auto a_blocks = a.blocks();
        auto b_blocks = b.blocks();

        int a_max = int(a_blocks.size());
        int b_max = int(b_blocks.size());
        int a_idx = 0;
        int b_idx = 0;

        for (;;) {
            while (a_idx < a_max && a_blocks[a_idx].dim_idx != i)
                a_idx++;
            while (b_idx < b_max && b_blocks[b_idx].dim_idx != i)
                b_idx++;

            if (a_idx >= a_max || b_idx >= b_max) break;

            auto &ab = a_blocks[a_idx];
            auto &bb = b_blocks[b_idx];
            dim_t common_block = math::gcd(ab.block, bb.block);
            if (ab.block == common_block && bb.block == common_block) {
                a_idx++;
                b_idx++;
                continue;
            }

            if (ab.block != common_block) {
                a = a.split_block(
                        {a_idx, ab}, common_block, ab.block / common_block);
            }
            if (bb.block != common_block) {
                b = b.split_block(
                        {b_idx, bb}, common_block, bb.block / common_block);
            }
            break;
        }
    }
}

std::vector<std::pair<int, dim_t>> layout_t::parse_format(
        const std::string &format, int ndims_hint) {
    bool seen_letters[DNNL_MAX_NDIMS] = {};
    int letter_ndims = 0;
    for (char c = 'a'; c < 'a' + DNNL_MAX_NDIMS; c++) {
        if (format.find(c) != std::string::npos) {
            seen_letters[c - 'a'] = true;
            MAYBE_UNUSED(seen_letters);
            letter_ndims++;
        }
    }

    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        ir_assert(seen_letters[i] == (i < letter_ndims));
    }

    std::string s = format;
    std::regex r(R"((\d*)([a-zA-Z]))");
    std::smatch sm;

    std::vector<std::pair<int, dim_t>> parts;
    while (regex_search(s, sm, r)) {
        auto c = sm.str(2)[0];
        int dim_idx = (c == 'x') ? -1 : (std::tolower(c) - 'a');
        dim_t block = sm.str(1).empty() ? 0 : std::stol(sm.str(1));
        if (dim_idx != -1) {
            parts.emplace_back(dim_idx, block);
        } else {
            ir_assert(ndims_hint >= letter_ndims);
            for (int i = letter_ndims; i < ndims_hint; i++) {
                parts.emplace_back(i, 0);
            }
        }
        s = sm.suffix();
    }

    return parts;
}

void layout_t::sanity_check() const {
#ifdef NDEBUG
    return;
#endif
    if (is_empty()) return;

    for (auto &b : blocks_) {
        ir_assert(b.block > 0) << "Incorrect block size.";
        MAYBE_UNUSED(b);
    }
}

layout_t layout_t::split_into_multi_blocks_impl(
        const std::vector<dim_t> &multi_blocks,
        std::vector<dim_t> *out_multi_blocks) const {
    if (is_empty()) return *this;

    bool allow_smaller_blocks = bool(out_multi_blocks);
    layout_t tmp(*this);
    std::vector<dim_t> rem_elems = multi_blocks;
    std::vector<dim_t> cur_elems(rem_elems.size(), 1);
    for (auto &eb : tmp.enumerated_blocks()) {
        auto &b = eb.second;
        for (int i = 0; i < int(rem_elems.size()); i++) {
            auto &e = rem_elems[i];
            if (e == 1) continue;
            if (b.block > e) {
                // Try to split this block.
                int next_block = utils::max_div(b.block, e);
                return tmp.split_block(eb, next_block, b.block / next_block)
                        .split_into_multi_blocks_impl(
                                multi_blocks, out_multi_blocks);
            }
            if (e % b.block != 0) {
                if (!allow_smaller_blocks) return layout_t();
            }
            e /= b.block;
            cur_elems[i] *= b.block;
            break;
        }
    }
    for (int i = 0; i < int(cur_elems.size()); i++) {
        if (cur_elems[i] != multi_blocks[i]) {
            if (!allow_smaller_blocks) return layout_t();
        }
        if (out_multi_blocks) (*out_multi_blocks)[i] = cur_elems[i];
    }
    return tmp;
}

expr_t grid_splitter_t::pop_block(int size) {
    ir_assert(size > 1);
    ir_assert(can_pop_block(size));

    int new_stride = cur_stride_ * size;

    auto idx_expr = grid_.idx(cur_idx_);
    if (cur_stride_ != 1) idx_expr /= cur_stride_;
    if (new_stride != grid_.dim(cur_idx_)) idx_expr %= size;

    cur_stride_ = new_stride;
    if (cur_stride_ == grid_.dim(cur_idx_)) {
        // Move to the next dimension.
        cur_idx_--;
        cur_stride_ = 1;
    }
    return idx_expr;
}

mask_vector_t mask_vector_t::reinterpret(const type_t &new_type) const {
    dim_t bytes = elems() * type_.size();
    ir_assert(bytes % new_type.size() == 0) << "Can't reinterpret.";

    std::vector<int> new_masks(bytes / new_type.size());
    for (dim_t i = 0; i < bytes; i += new_type.size()) {
        int mask_id = std::numeric_limits<int>::max();
        for (int j = 0; j < new_type.size(); j++) {
            int cur_mask_id = masks_[(i + j) / type_.size()];
            if (mask_id >= int(masks_.size())) {
                mask_id = cur_mask_id;
            } else if (mask_id != cur_mask_id) {
                // Mask is not consistent, can't reinterpret.
                return mask_vector_t();
            }
        }
        ir_assert(0 <= mask_id && mask_id < int(masks_.size()));
        new_masks[i / new_type.size()] = mask_id;
    }
    return mask_vector_t(new_type, new_masks, mask2ids_, id2masks_);
}

expr_t mask_vector_t::to_expr(int nmasks) const {
    if (elems_ % nmasks != 0) return expr_t();

    std::vector<expr_t> vec(nmasks);
    for (int i = 0; i < elems_; i++) {
        auto &channel_mask = vec[i % nmasks];
        auto &cur_mask = id2masks_[masks_[i]];
        if (channel_mask.is_empty()) {
            channel_mask = cur_mask;
            continue;
        }
        if (!channel_mask.is_equal(cur_mask)) return expr_t();
    }
    auto e = shuffle_t::make(vec);
    e = jit::simplify(e);
    e = jit::simplify_propagate_shuffle(e);
    return e;
}

stride_t tdim_info_t::compute_stride(
        const expr_t &e, int idx, const expr_t &var) {
    // e == var -> fixed stride.
    if (e.is_same(var)) return stride_t(1);

    auto vars = find_objects<var_t>(e);

    auto e0 = e;
    auto e1 = substitute(e, var, var + 1);
    auto e_stride = simplify(e1 - e0);

    if (is_const(e_stride)) return stride_t(to_cpp<dim_t>(e_stride));

    // Stride is not a constant.
    return stride_t::unknown();
}

view_t view_t::create_sub_view(
        const tensor_t &sub_tensor, bool relative_vstart) const {
    ir_assert(sub_tensor.ndims() == nvdims()) << "Dimensions don't match.";

    if (!relative_vstart) { ir_assert(has_zero_vdirect_start()); }

    auto ret = *this;
    ret.vdims_ = sub_tensor.dims();
    for (int i = 0; i < nvdims(); i++) {
        auto &i_start = sub_tensor.start()[i];
        if (is_zero(i_start)) continue;
        auto &s = (is_direct_ ? ret.vdirect_start_[i] : ret.vstart_[i]);
        if (relative_vstart) {
            s += i_start;
            s = simplify(s);
        } else {
            if (is_direct_) {
                s = simplify(i_start - ret.vstart_[i]);
            } else {
                s = i_start;
            }
        }
    }
    return ret;
}

view_t view_t::split(const grid_info_t &grid) const {
    std::vector<dim_t> tile_dims(nvdims(), 1);
    dim_t elems = velems();
    ir_assert(elems % grid.elems() == 0) << "Can't split across grid.";

    dim_t cur_elems_per_tile = 1;
    dim_t elems_per_tile = elems / grid.elems();
    auto vlayout = create_pseudo_vlayout();
    for (auto &b : vlayout.blocks()) {
        dim_t block = std::min(b.block, elems_per_tile / cur_elems_per_tile);
        tile_dims[b.dim_idx] *= block;
        cur_elems_per_tile *= block;
    }
    ir_assert(cur_elems_per_tile == elems_per_tile)
            << "Can't split across grid.";

    return split(vlayout, tensor_t(tile_dims), grid);
}

view_t view_t::split(const mnk_tensor_t &mnk_tile, const grid_info_t &grid,
        std::vector<block_t> *outer_blocks) const {
    ir_assert(velems() == grid.elems() * mnk_tile.elems())
            << "Can't split across grid.";

    std::vector<dim_t> tile_dims(nvdims(), 1);
    std::unordered_map<mnk_kind_t, dim_t, ir_utils::enum_hash_t<mnk_kind_t>>
            rem_elems_per_mnk_kind;
    for (auto mnk_kind : mnk_tile.mnk_kinds())
        rem_elems_per_mnk_kind[mnk_kind] = mnk_tile.dim(mnk_kind);
    auto vlayout = create_pseudo_vlayout();
    for (auto &b : vlayout.blocks()) {
        auto mnk_kind = vmnk_kinds_[b.dim_idx];
        if (!mnk_tile.has(mnk_kind)) continue;

        dim_t &rem = rem_elems_per_mnk_kind[mnk_kind];
        dim_t block = std::min(b.block, rem);
        ir_assert(rem % block == 0);
        rem /= block;
        tile_dims[b.dim_idx] *= block;
    }
    for (auto &kv : rem_elems_per_mnk_kind) {
        ir_assert(kv.second == 1);
        MAYBE_UNUSED(kv);
    }

    return split(vlayout, tensor_t(tile_dims), grid, outer_blocks);
}

view_t view_t::substitute(const expr_t &from, const expr_t &to) const {
    view_t ret = *this;
    for (int i = 0; i < nvdims(); i++) {
        ret.vstart_[i] = jit::substitute(ret.vstart_[i], from, to);
        ret.vstart_[i] = simplify(ret.vstart_[i]);
        if (is_direct_) {
            ret.vdirect_start_[i]
                    = jit::substitute(ret.vdirect_start_[i], from, to);
            ret.vdirect_start_[i] = simplify(ret.vdirect_start_[i]);
        }
    }
    return ret;
}

layout_t view_t::create_pseudo_vlayout(const layout_t &tlayout) const {
    ir_assert(!is_direct_);
    ir_assert(!tlayout.is_empty());

    std::vector<dim_t> rem_vdims = vdims_;
    std::vector<block_t> blocks;

    for (auto &teb : tlayout.enumerated_blocks()) {
        block_t &tb = teb.second;
        bool tb_is_outermost = tlayout.is_outermost(teb);
        dim_t tblock = tb.block;

        auto &tinfo = tdims_[tb.dim_idx];
        if (tb_is_outermost) {
            bool is_first = true;
            for (int i = tinfo.nvargs() - 1; i >= 0; i--) {
                int vidx = tinfo.vidx(i);
                if (rem_vdims[vidx] == 1) continue;

                // When expression contains 2+ variables, use unknown
                // stride unless the view variable is the innermost.
                stride_t stride
                        = (is_first ? tinfo.vstride(i) : stride_t::unknown());
                blocks.emplace_back(
                        vidx, rem_vdims[vidx], stride * stride_t(tb.stride));
                rem_vdims[vidx] = 1;
                is_first = false;
            }
            continue;
        }

        ir_assert(tinfo.is_identity()) << "Can't create pseudo-layout.";

        int vidx = tinfo.vidx(0);
        dim_t &rem_vdim = rem_vdims[vidx];
        if (rem_vdim == 1) continue;

        if (tb_is_outermost) {
            tblock = rem_vdim;
            rem_vdim = 1;
        } else if (rem_vdim % tblock == 0) {
            rem_vdim /= tblock;
        } else if (rem_vdim % tblock != 0) {
            // Try to split the current block and start from scratch.
            if (tblock % rem_vdim == 0) {
                auto tmp_layout
                        = tlayout.split_block(teb, rem_vdim, tblock / rem_vdim);
                return create_pseudo_vlayout(tmp_layout);
            }

            ir_error_not_expected() << "Can't create pseudo-layout.";
        }
        blocks.emplace_back(tb.dim_idx, tblock, tb.stride);
    }

    for (auto &d : rem_vdims) {
        ir_assert(d == 1) << "Can't create pseudo-layout.";
        MAYBE_UNUSED(d);
    }

    return layout_t(tlayout.type(), nvdims(), 0, blocks);
}

view_t view_t::split(const layout_t &vlayout, const tensor_t &vtile,
        const grid_info_t &grid, std::vector<block_t> *outer_blocks) const {
    ir_assert(nvdims() == vtile.ndims())
            << "Number of dimensions doesn't match.";
    ir_assert(vtile.has_zero_start());

    if (outer_blocks) outer_blocks->resize(0);

    if (grid.elems() == 1) return *this;

    dim_t total_elems = velems();
    dim_t tile_elems = vtile.elems();

    grid_splitter_t grid_splitter(grid);
    ir_assert(tile_elems * grid.elems() == total_elems)
            << "Tile/grid dimensions do not match.";
    MAYBE_UNUSED(total_elems);
    MAYBE_UNUSED(tile_elems);

    std::vector<dim_t> dims(vtile.ndims(), 1);
    std::vector<expr_t> start(vtile.ndims(), 0);
    std::vector<dim_t> rem_dims = vtile.dims();
    for (auto &eb : vlayout.enumerated_blocks()) {
        auto &b = eb.second;
        if (b.block == 1) continue;

        dim_t &e = rem_dims[b.dim_idx];
        if (e > 1) {
            if (e % b.block == 0) {
                e /= b.block;
            } else if (b.block % e == 0) {
                auto tmp_layout = vlayout.split_block(eb, e, b.block / e);
                return split(tmp_layout, vtile, grid, outer_blocks);
            } else {
                ir_error_not_expected() << "Can't split across grid.";
            }
        } else {
            dim_t next_chunk = math::gcd(b.block, grid_splitter.cur_block());
            if (b.block == next_chunk) {
                auto idx = grid_splitter.pop_block(next_chunk);
                start[b.dim_idx] += idx * dims[b.dim_idx];
                if (outer_blocks) outer_blocks->push_back(b);
            } else if (b.block % next_chunk == 0) {
                auto tmp_layout = vlayout.split_block(
                        eb, next_chunk, b.block / next_chunk);
                return split(tmp_layout, vtile, grid, outer_blocks);
            } else {
                ir_error_not_expected() << "Can't split across grid.";
            }
        }
        dims[b.dim_idx] *= b.block;
    }
    return create_sub_view(tensor_t(vtile.dims(), start));
}

void view_t::create_mask_vector(mask_vector_t &mask_vec,
        const layout_t &_vlayout, int vidx, std::vector<dim_t> &vargs) const {
    if (vidx == _vlayout.ndims()) {
        std::vector<expr_t> vvalues = vstart_;
        for (int i = 0; i < nvdims(); i++)
            vvalues[i] += vargs[i];
        auto targs = cvt_vargs_to_targs<dim_t, expr_t>(vargs);
        expr_t mask = bool_imm_t::make(true);
        for (int i = 0; i < ntdims(); i++) {
            auto &tdim = tdims_[i];
            if (tdim.mask().is_empty()) continue;
            mask &= tdim.mask(targs[i], vvars_, vvalues);
        }
        mask_vec.set_mask(_vlayout(vargs), mask);
        return;
    }

    for (int i = 0; i < vdims()[vidx]; i++) {
        vargs[vidx] = i;
        create_mask_vector(mask_vec, _vlayout, vidx + 1, vargs);
    }
}

void mnk_mapper_t::push_block(
        const block_t &_b, const view_t &ab_view, const view_t &c_view) {
    auto &ab_var = ab_view.vvars()[_b.dim_idx];
    auto mnk_kind = ab_view.vmnk_kinds()[_b.dim_idx];
    if (!utils::one_of(mnk_kind, mnk_kind_t::m, mnk_kind_t::n)) return;

    auto b = _b;
    b.dim_idx = c_view.vvar_index(ab_var);
    switch (mnk_kind) {
        case mnk_kind_t::m: m_blocks_.push_back(b); break;
        case mnk_kind_t::n: n_blocks_.push_back(b); break;
        default: ir_error_not_expected();
    }
}

layout_t mnk_mapper_t::map_to_mnk(
        const view_t &view, const std::vector<mnk_kind_t> &mnk_kinds) const {
    ir_assert(view.is_direct())
            << "Only direct views can be mapped to mnk layouts.";
    auto layout = view.create_pseudo_vlayout();
    return map_to_mnk(layout, view, mnk_kinds);
}

layout_t mnk_mapper_t::map_to_mnk(const layout_t &layout, const view_t &view,
        const std::vector<mnk_kind_t> &mnk_kinds) const {
    std::vector<block_t> blocks;
    for (auto &b : layout.blocks()) {
        auto mnk_kind = view.vmnk_kinds()[b.dim_idx];
        bool found = false;
        for (int i = 0; i < int(mnk_kinds.size()); i++) {
            if (mnk_kinds[i] == mnk_kind) {
                blocks.emplace_back(i, b.block, b.stride);
                found = true;
                break;
            }
        }
        if (!found) ir_error_not_expected() << "MNK dimension not found.";
    }
    return layout_t(view.type(), int(mnk_kinds.size()), 0, blocks);
}

layout_t mnk_mapper_t::map_from_mnk(
        const layout_t &mnk_layout, int prb_ndims) const {
    ir_assert(mnk_layout.ndims() <= 2);
    ir_assert(mnk_layout.has_zero_offset());
    std::vector<block_t> blocks;
    std::vector<block_t> tmp_m_blocks = m_blocks_;
    std::vector<block_t> tmp_n_blocks = n_blocks_;
    for (auto &b : mnk_layout.blocks()) {
        auto &mnk_blocks = (b.dim_idx == 0 ? tmp_m_blocks : tmp_n_blocks);
        bool ok = pop_block(mnk_blocks, blocks, b);
        ir_assert(ok) << "Can't map from mnk layout to problem layout.";
        MAYBE_UNUSED(ok);
    }
    pop_size_1_blocks(tmp_m_blocks);
    pop_size_1_blocks(tmp_n_blocks);
    ir_assert(tmp_m_blocks.empty());
    ir_assert(tmp_n_blocks.empty());

    // Fix strides to make them dense.
    dim_t dense_stride = 1;
    for (auto &b : blocks) {
        b.stride = stride_t(dense_stride);
        dense_stride *= b.block;
    }

    return layout_t(mnk_layout.type(), prb_ndims, 0, blocks);
}

bool mnk_mapper_t::pop_block(std::vector<block_t> &mnk_blocks,
        std::vector<block_t> &prb_blocks, const block_t &mnk_block) const {
    if (mnk_block.block == 1) return true;

    pop_size_1_blocks(mnk_blocks);
    if (mnk_blocks.empty()) return false;

    auto &next_block = mnk_blocks.front();
    dim_t common_block = math::gcd(next_block.block, mnk_block.block);
    if (common_block == mnk_block.block) {
        prb_blocks.emplace_back(
                next_block.dim_idx, common_block, next_block.stride);
        next_block.block /= common_block;
        next_block.stride *= common_block;
        return true;
    } else if (common_block == next_block.block) {
        prb_blocks.emplace_back(
                next_block.dim_idx, common_block, next_block.stride);
        mnk_blocks.erase(mnk_blocks.begin());
        auto tmp_block = mnk_block;
        tmp_block.block /= common_block;
        return pop_block(mnk_blocks, prb_blocks, tmp_block);
    }
    return false;
}

layout_t dim_assignment_t::map(const layout_t &layout) const {
    std::vector<block_t> new_blocks;
    for (auto &b : layout.blocks()) {
        int new_idx = assignments_[b.dim_idx];
        if (new_idx == -1) continue; // Drop this block.
        auto new_b = b;
        new_b.dim_idx = new_idx;
        new_blocks.push_back(new_b);
    }
    new_blocks = layout_t::normalize_blocks(
            new_ndims(), new_blocks, /*keep_size_1_blocks=*/true);
    auto ret
            = layout_t(layout.type(), new_ndims(), layout.offset(), new_blocks);
    ir_assert(layout.elems() == ret.elems())
            << "Assignment doesn't preserve number of elements.";
    return ret;
}

layout_t normalize_spatial(
        const layout_t &layout, int old_sp_ndims, bool reduced_to_1d) {
    int old_ndims = layout.ndims();
    int new_ndims = old_ndims - old_sp_ndims + 3;

    dim_assignment_t to_3d(old_ndims, new_ndims);
    for (int i = 0; i < old_ndims; i++) {
        if (i < old_ndims - old_sp_ndims) {
            // Non-spatial dimensions.
            to_3d.assign(i, i);
        } else {
            // Spatial dimensions.
            int sp_idx = 3 - (old_ndims - i);
            if (reduced_to_1d) sp_idx = 2;
            to_3d.assign(i, new_ndims - (3 - sp_idx));
        }
    }
    return to_3d.map(layout);
}

std::vector<dim_t> normalize_spatial(
        const std::vector<dim_t> &dims, int old_sp_ndims, bool reduced_to_1d) {
    layout_t dummy_layout(type_t::u8(), 0, dims);
    return normalize_spatial(dummy_layout, old_sp_ndims, reduced_to_1d).dims();
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

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

#ifndef GPU_JIT_CONV_TENSOR_HPP
#define GPU_JIT_CONV_TENSOR_HPP

#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>

#include "common/memory_desc_wrapper.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class tensor_t {
public:
    tensor_t() = default;

    tensor_t(const std::vector<dim_t> &dims)
        : tensor_t(dims, std::vector<expr_t>()) {}

    tensor_t(const std::vector<dim_t> &dims, const std::vector<expr_t> &start)
        : dims_(dims), start_(start) {
        if (start_.empty()) start_.resize(dims.size(), 0);
    }

    tensor_t(const std::vector<dim_t> &dims, const std::vector<dim_t> &start)
        : tensor_t(dims) {
        start_.resize(start.size());
        for (size_t i = 0; i < start.size(); i++)
            start_[i] = start[i];
    }

    dim_t operator()(int idx) const { return dims_[idx]; }

    const expr_t &start(int idx) const { return start_[idx]; }

    int ndims() const { return int(dims_.size()); }

    dim_t elems() const {
        dim_t ret = 1;
        for (int i = 0; i < ndims(); i++)
            ret *= dims_[i];
        return ret;
    }

    const std::vector<dim_t> &dims() const { return dims_; }

    const std::vector<expr_t> &start() const { return start_; }

    bool is_empty() const { return dims_.empty(); }

    bool is_equal(const tensor_t &other) const {
        if (ndims() != other.ndims()) return false;
        for (int i = 0; i < ndims(); i++) {
            if (dims_[i] != other.dims_[i]) return false;
            if (!start_[i].is_equal(other.start_[i])) return false;
        }
        return true;
    }

    std::string str() const {
        using ir_utils::operator<<;

        if (is_empty()) return "(nil)";
        std::ostringstream oss;
        oss << ir_utils::make_seq_print_helper(dims_, "x");
        if (!has_zero_start()) oss << " start: [" << start_ << "]";
        return oss.str();
    }

    IR_DEFINE_DUMP()

    bool has_zero_start() const {
        for (int i = 0; i < ndims(); i++)
            if (!is_zero(start_[i])) return false;
        return true;
    }

    dim_t to_1d_offset(const std::vector<dim_t> &args) const {
        ir_assert(has_zero_start());

        dim_t off = 0;
        for (int i = 0; i < ndims(); i++) {
            off *= dims_[i];
            off += args[i];
        }
        return off;
    }

private:
    std::vector<dim_t> dims_;
    std::vector<expr_t> start_;
};

inline std::ostream &operator<<(std::ostream &out, const tensor_t &tensor) {
    out << tensor.str();
    return out;
}

enum class stride_kind_t {
    undef,
    fixed,
    unknown,
};

class stride_t {
public:
    stride_t() = default;

    stride_t(dim_t stride) : stride_t(stride_kind_t::fixed, stride) {}

    bool operator==(const stride_t &other) const {
        return (kind_ == other.kind_) && (stride_ == other.stride_);
    }

    bool operator!=(const stride_t &other) const { return !operator==(other); }

    size_t get_hash() const { return ir_utils::get_hash(kind_, stride_); }

    operator dim_t() const {
        ir_assert(kind_ == stride_kind_t::fixed);
        return stride_;
    }

    bool is_fixed() const { return kind_ == stride_kind_t::fixed; }

    bool is_unknown() const { return kind_ == stride_kind_t::unknown; }

    stride_t &operator*=(const stride_t &other) {
        if (is_fixed() && other.is_fixed()) {
            stride_ *= other.stride_;
        } else {
            set_unknown();
        }
        return *this;
    }

    stride_t &operator/=(const stride_t &other) {
        if (is_fixed() && other.is_fixed()) {
            stride_ /= other.stride_;
        } else {
            set_unknown();
        }
        return *this;
    }

    std::string str() const {
        std::ostringstream oss;
        if (is_fixed()) {
            oss << stride_;
        } else {
            oss << "(unknown)";
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

    static stride_t unknown() { return stride_t(stride_kind_t::unknown); }

private:
    stride_t(stride_kind_t kind, dim_t stride = 0)
        : kind_(kind), stride_(stride) {}

    void set_unknown() {
        kind_ = stride_kind_t::unknown;
        stride_ = 0;
    }

    stride_kind_t kind_ = stride_kind_t::undef;
    dim_t stride_ = 0;
};

inline std::ostream &operator<<(std::ostream &out, const stride_t &stride) {
    out << stride.str();
    return out;
}

inline stride_t operator*(const stride_t &a, const stride_t &b) {
    stride_t tmp = a;
    return tmp *= b;
}

inline stride_t operator*(const stride_t &a, dim_t b) {
    return a * stride_t(b);
}

inline stride_t operator*(dim_t a, const stride_t &b) {
    return stride_t(a) * b;
}

struct block_t {
    block_t() = default;

    block_t(int dim_idx, dim_t block, const stride_t &stride)
        : dim_idx(dim_idx), block(block), stride(stride) {}

    bool is_equal(const block_t &other) const {
        return (dim_idx == other.dim_idx) && (block == other.block)
                && (stride == other.stride);
    }

    size_t get_hash() const {
        return ir_utils::get_hash(dim_idx, block, stride);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "block_t(dim_idx = " << dim_idx;
        oss << ", block = " << block;
        oss << ", stride = " << stride;
        oss << ")";
        return oss.str();
    }

    IR_DEFINE_DUMP()

    int dim_idx; // Dimension index.
    dim_t block; // Block size.
    stride_t stride; // Stride between elements of the block.
};

inline std::ostream &operator<<(std::ostream &out, const block_t &b) {
    out << b.str();
    return out;
}

class layout_t {
public:
    static const int max_ndims = 6;

    layout_t() : type_(type_t::undef()), ndims_(0), offset_(0) {
        sanity_check();
    }

    layout_t(const type_t &type, const expr_t &offset,
            const std::string &format, const std::vector<dim_t> &dims = {});

    layout_t(const memory_desc_wrapper &mdw, const std::string &format)
        : layout_t(mdw.data_type(), mdw.offset0(), format,
                std::vector<dim_t>(
                        mdw.padded_dims(), mdw.padded_dims() + mdw.ndims())) {}

    layout_t(const memory_desc_wrapper &mdw);

    layout_t(const type_t &type, const expr_t &offset,
            const std::vector<dim_t> &dims)
        : type_(type), ndims_(int(dims.size())), offset_(offset) {
        dim_t stride = 1;
        for (int i = ndims_ - 1; i >= 0; i--) {
            blocks_.emplace_back(i, dims[i], stride);
            stride *= dims[i];
        }
        sanity_check();
    }

    layout_t(const type_t &type, int ndims, const expr_t &offset,
            const std::vector<block_t> &blocks)
        : type_(type), ndims_(ndims), offset_(offset), blocks_(blocks) {
        sanity_check();
    }

    layout_t(const type_t &type, const expr_t &offset, const layout_t &other)
        : layout_t(type, other.ndims(), offset, other.blocks()) {}

    bool is_empty() const { return ndims_ == 0; }

    int ndims() const { return ndims_; }

    dim_t elems() const {
        dim_t ret = 1;
        for (auto &b : blocks_)
            ret *= b.block;
        return ret;
    }

    // Storage size in bytes.
    dim_t size() const {
        if (is_empty()) return 0;
        dim_t max_stride = 1;
        for (auto &b : blocks_) {
            max_stride = std::max(max_stride, dim_t(b.block * b.stride));
        }
        return max_stride * type().size();
    }

    template <typename T = expr_t>
    T offset(
            const std::vector<T> &args = {}, bool ignore_offset = false) const {
        if (args.empty()) return expr_cast<T>(offset_);

        ir_assert(int(args.size()) == ndims()) << "Dimensions do not match.";

        T off = 0;
        auto _args = args;
        for (auto &eb : enumerated_blocks()) {
            auto &b = eb.second;
            auto &idx = _args[b.dim_idx];
            if (ir_utils::is_equal(idx, T(0))) continue;

            // Do not use modulus for outermost blocks.
            auto i = is_outermost(eb) ? idx : (idx % b.block);
            off = i * dim_t(b.stride) + off;
            idx /= b.block;
        }
        if (ignore_offset) return off;

        T off0 = expr_cast<T>(offset_);
        return off0 + off;
    }

    const type_t &type() const { return type_; }

    std::vector<dim_t> dims() const {
        std::vector<dim_t> dims(ndims(), 1);
        for (auto &b : blocks_) {
            dims[b.dim_idx] *= b.block;
        }
        return dims;
    }

    dim_t dim(int dim_idx) const {
        dim_t ret = 1;
        for (auto &b : blocks_) {
            if (b.dim_idx == dim_idx) ret *= b.block;
        }
        return ret;
    }

    const std::vector<block_t> &blocks() const { return blocks_; }

    void set_offset(const expr_t &offset) { offset_ = offset; }

    bool is_strictly_equal(
            const layout_t &other, bool compare_offset = true) const {
        if (!type_.is_equal(other.type_)) return false;
        if (compare_offset && !offset_.is_equal(other.offset_)) return false;
        if (!ir_utils::is_equal(blocks_, other.blocks_)) return false;
        return true;
    }

    bool operator==(const layout_t &other) const { return is_equal(other); }

    bool operator!=(const layout_t &other) const { return !operator==(other); }

    bool is_equal(const layout_t &other, bool compare_offset = true) const {
        return normalize().is_strictly_equal(other.normalize(), compare_offset);
    }

    size_t get_hash() const {
        return ir_utils::get_hash(type_, ndims_, offset_, blocks_);
    }

    template <typename T>
    T operator()(const std::vector<T> &args) const {
        return offset(args);
    }

    template <typename T = expr_t>
    T offset_in_bytes(
            const std::vector<T> &args = {}, bool ignore_offset = false) const {
        return offset(args, ignore_offset) * type().size();
    }

    std::string desc_str(bool dnnl_style = false) const {
        if (is_empty()) return "(nil)";
        if (!dnnl_style && blocks_.empty()) return "(scalar)";
        std::string ret;
        stride_t dense_stride(1);
        std::vector<bool> seen(ndims());
        for (auto &eb : enumerated_blocks()) {
            auto &b = eb.second;
            std::string b_str;
            if (dnnl_style && is_outermost(eb)) {
                b_str.append(1, (seen[b.dim_idx] ? 'A' : 'a') + b.dim_idx);
            } else {
                b_str = std::to_string(b.block);
                b_str.append(1, 'a' + b.dim_idx);
            }
            if (!dnnl_style) {
                if (b.stride.is_unknown()) {
                    b_str.append(1, '?');
                } else if (b.stride != dense_stride) {
                    b_str.append(1, '*');
                }
            }
            ret = b_str + ret;
            dense_stride = b.stride * b.block;
            seen[b.dim_idx] = true;
        }
        return ret;
    }

    std::string str() const {
        if (is_empty()) return "(nil)";
        std::ostringstream oss;
        oss << desc_str();
        if (!has_zero_offset()) oss << " offset: " << offset_;
        return oss.str();
    }

    IR_DEFINE_DUMP()

    memory_desc_t to_dnnl(const dim_t *dims_hint) const;

    // Returns a vector of <block index, block> pairs.
    // The innermost block (first) has index 0.
    std::vector<std::pair<int, block_t>> enumerated_blocks() const {
        std::vector<std::pair<int, block_t>> ret;
        for (int i = 0; i < int(blocks_.size()); i++) {
            ret.emplace_back(i, blocks_[i]);
        }
        return ret;
    }

    std::vector<dim_t> strides(int dim_idx) const {
        std::vector<dim_t> ret;
        for (auto &b : blocks_)
            if (b.dim_idx == dim_idx) ret.push_back(b.stride);
        return ret;
    }

    // eb is <block index, block> pair, see enumerated_blocks().
    bool is_outermost(const std::pair<int, block_t> &eb) const {
        return is_outermost(eb, blocks_);
    }

    bool is_plain() const {
        std::vector<bool> seen(ndims());
        for (auto &b : blocks_) {
            if (seen[b.dim_idx]) return false;
            seen[b.dim_idx] = true;
        }
        return true;
    }

    bool has_zero_offset() const { return offset_.is_equal(expr_t(0)); }

    bool has_unknown_strides() const {
        for (auto &b : blocks_)
            if (b.stride.is_unknown()) return true;
        return false;
    }

    // Returns a canonical representation of the layout:
    // - Consecutive dense blocks are merged
    // - Size one blocks are:
    //   - Removed (if keep_size_1_blocks is false)
    //   - Reordered according to the heuristic (if keep_size_1_blocks is true)
    // Optionally removes size one blocks and merges consecutive dense blocks
    // representing the same dimension.
    layout_t normalize(bool keep_size_1_blocks = false) const {
        auto blocks = normalize_blocks(ndims(), blocks_, keep_size_1_blocks);
        return layout_t(type(), ndims(), offset(), blocks);
    }

    layout_t transpose() const {
        if (ndims() != 2) ir_error_not_expected();

        // Flip: 0 -> 1, 1 -> 0.
        auto blocks = blocks_;
        for (auto &b : blocks)
            b.dim_idx ^= 1;

        return layout_t(type(), ndims(), offset(), blocks);
    }

    // Returns a new (sub-)layout that fully contains the passed sub-tensor.
    // Strides are kept unchanged.
    // Assumption: the original layout can be tiled by the passed sub-tensor.
    // For example: XaYb4a2b can be tiled into 2x2 sub-tensors but it's not
    // possible to tile it into 3x2 sub-tensors.
    layout_t map(const tensor_t &tensor) const;

    layout_t reinterpret(const type_t &new_type) const;

    layout_t retype(const type_t &new_type) const {
        auto ret = *this;
        ret.type_ = new_type;
        return ret;
    }

    bool is_dense() const {
        stride_t stride = 1;
        for (auto &b : blocks_) {
            if (b.stride != stride) return false;
            stride *= b.block;
        }
        return true;
    }

    // Returns a packed layout where all blocks are contiguous, without gaps.
    layout_t make_dense() const {
        dim_t stride = 1;
        auto new_blocks = blocks_;
        for (auto &b : new_blocks) {
            b.stride = stride;
            stride *= b.block;
        }
        return layout_t(type(), ndims(), 0, new_blocks);
    }

    // Returns an equivalent layout where the specified block is split into two.
    // block0 - inner block size.
    // block1 - outer block size.
    layout_t split_block(const std::pair<int, block_t> &eb, dim_t block0,
            dim_t block1) const;

    // Splits blocks so that they can be used to form `multi_blocks` without
    // crossing the block boundaries. `multi_blocks` are ordered from innermost
    // to outermost. Returns an empty layout if such a split is not possible.
    // Example (all blocks are ordered from innermost to outermost):
    //     Input blocks:  [4, 4, 2]
    //     Multi-blocks:  [8, 2]
    //     Output blocks: [4, 2, 2, 2]
    layout_t split_into_multi_blocks(
            const std::vector<dim_t> &multi_blocks) const;

    layout_t split_into_multi_blocks_with_hint(
            std::vector<dim_t> &multi_blocks) const;

    layout_t add_outer_block(
            int dim_idx, dim_t block, dim_t stride = -1) const {
        if (stride == -1) stride = elems();
        ir_assert(stride >= elems());
        ir_assert(dim_idx < ndims());
        auto new_blocks = blocks();
        new_blocks.emplace_back(dim_idx, block, stride);
        return layout_t(type(), ndims(), offset(), new_blocks);
    }

    tensor_t split_into_dense_tile(dim_t tile_elems, dim_t outer_block) const;

    // Returns a tensor corresponding to the biggest innermost sub-layout so that
    // 1) It consists of consecutive blocks only.
    // 2) It contains less or equal than max_tile_elems elements.
    // 3) It is dense if is_dense_tile is true.
    tensor_t split_into_max_tile(
            dim_t max_tile_elems, bool is_dense_tile) const;

    // Iterates through tiles of the layout, calling `f` with relative offsets
    // for each tile. The iteration order is defined by the layout blocks -
    // absolute 1D offsets are increasing between callback calls.
    template <typename F>
    void for_each_tile(const tensor_t &tile, const F &f) const {
        ir_assert(tile.ndims() == ndims());
        ir_assert(tile.has_zero_start());
        for (int i = 0; i < ndims(); i++) {
            ir_assert(dim(i) % tile.dims()[i] == 0);
        }

        int nblocks = int(blocks().size());
        std::vector<dim_t> sub_blocks(nblocks);
        for (int i = 0; i < nblocks; i++)
            sub_blocks[i] = blocks()[i].block;

        for (int i = 0; i < ndims(); i++) {
            dim_t dim = tile.dims()[i];
            for (auto &eb : enumerated_blocks()) {
                auto &b = eb.second;
                if (b.dim_idx != i) continue;
                int block_idx = eb.first;
                if (b.block >= dim) {
                    ir_assert(b.block % dim == 0);
                    sub_blocks[block_idx] = b.block / dim;
                    break;
                }
                sub_blocks[block_idx] = 1;
                ir_assert(dim % b.block == 0);
                dim /= b.block;
            }
        }

        int ntiles = int(elems() / tile.elems());

        std::vector<dim_t> sub_block_idxs(nblocks);
        for (int i = 0; i < ntiles; i++) {
            // Convert sub-block indices to dimension indices.
            std::vector<dim_t> dims(ndims(), 1);
            std::vector<dim_t> start(ndims());
            for (int j = 0; j < nblocks; j++) {
                auto &b = blocks()[j];
                dim_t k = sub_block_idxs[j]
                        * (blocks()[j].block / sub_blocks[j]);
                start[b.dim_idx] += dims[b.dim_idx] * k;
                dims[b.dim_idx] *= b.block;
            }

            // Pass dimension offsets to the callback.
            f(start);

            // Move to the next vector of indices.
            for (int j = 0; j < nblocks; j++) {
                auto &idx = sub_block_idxs[j];
                if (idx + 1 < sub_blocks[j]) {
                    idx++;
                    break;
                }
                idx = 0;
            }
        }
    }

    // eb is <block index, block> pair, see enumerated_blocks().
    static bool is_outermost(const std::pair<int, block_t> &eb,
            const std::vector<block_t> &blocks) {
        int dim_idx = eb.second.dim_idx;
        for (int i = 0; i < int(blocks.size()); i++) {
            if (blocks[i].dim_idx == dim_idx && i > eb.first) return false;
        }
        return true;
    }

    // Assume that layouts are normalized.
    static void align_layouts(layout_t &a, layout_t &b);

    static std::vector<block_t> normalize_blocks(int ndims,
            const std::vector<block_t> &blocks,
            bool keep_size_1_blocks = false) {
        auto new_blocks = blocks;

        // Remove blocks of size 1.
        for (auto it = new_blocks.begin(); it != new_blocks.end();) {
            if (it->block == 1) {
                it = new_blocks.erase(it);
            } else {
                ++it;
            }
        }
        // Merge same dimension blocks.
        block_t prev_b;
        prev_b.dim_idx = -1;
        for (auto it = new_blocks.begin(); it != new_blocks.end();) {
            if (it->dim_idx == prev_b.dim_idx
                    && it->stride == (prev_b.stride * prev_b.block)) {
                auto &b = *(it - 1);
                b.block *= it->block;
                prev_b = b;
                it = new_blocks.erase(it);
            } else {
                prev_b = *it;
                ++it;
            }
        }
        // No need to keep size one blocks, return.
        if (!keep_size_1_blocks) return new_blocks;

        bool seen[max_ndims] = {false};
        for (auto &b : new_blocks)
            seen[b.dim_idx] = true;

        stride_t stride = (new_blocks.empty()
                        ? stride_t(1)
                        : new_blocks.back().stride * new_blocks.back().block);

        // Insert size one blocks according to the following heuristic:
        // TODO: Add documentation.
        for (int i = ndims - 1; i >= 0; i--) {
            if (seen[i]) continue;
            new_blocks.emplace_back(i, 1, stride);
        }

        return new_blocks;
    }

private:
    // Returns vector of <dimension index, block size> pairs.
    std::vector<std::pair<int, dim_t>> parse_format(
            const std::string &format, int ndims_hint);

    void sanity_check() const;

    layout_t split_into_multi_blocks_impl(
            const std::vector<dim_t> &multi_blocks,
            std::vector<dim_t> *out_multi_blocks) const;

    // Data type of the layout.
    type_t type_;

    // Number of dimensions.
    int ndims_;

    // Offset to the start of the layout (in elements of type).
    expr_t offset_;

    // Blocks ordered from innermost to outermost.
    std::vector<block_t> blocks_;
};

inline std::ostream &operator<<(std::ostream &out, const layout_t &layout) {
    out << layout.str();
    return out;
}

class grid_info_t {
public:
    grid_info_t() = default;
    grid_info_t(int ndims) : dims_(ndims), offs_(ndims), idxs_(ndims) {}
    grid_info_t(const std::vector<int> &dims, const std::vector<expr_t> &idxs)
        : grid_info_t(dims, {}, idxs) {}
    grid_info_t(const std::vector<int> &dims, const std::vector<int> &offs,
            const std::vector<expr_t> &idxs)
        : dims_(dims), offs_(offs), idxs_(idxs) {
        if (offs_.empty()) offs_.resize(dims.size());
        ir_assert(dims_.size() == offs_.size());
        ir_assert(dims_.size() == idxs_.size());
    }

    bool operator==(const grid_info_t &other) const {
        if (ndims() != other.ndims()) return false;
        for (int i = 0; i < ndims(); i++) {
            if (dim(i) != other.dim(i)) return false;
            if (off(i) != other.off(i)) return false;
            if (!idx(i).is_equal(other.idx(i))) return false;
        }
        return true;
    }

    bool is_empty() const { return dims_.empty(); }

    int &dim(int dim_idx) { return dims_[dim_idx]; }
    int &off(int dim_idx) { return offs_[dim_idx]; }
    expr_t &idx(int dim_idx) { return idxs_[dim_idx]; }

    const int &dim(int dim_idx) const { return dims_[dim_idx]; }
    const int &off(int dim_idx) const { return offs_[dim_idx]; }
    const expr_t &idx(int dim_idx) const { return idxs_[dim_idx]; }

    int ndims() const { return int(dims_.size()); }
    int elems() const {
        return utils::array_product(dims_.data(), dims_.size());
    }

    grid_info_t sub_grid(std::initializer_list<int> old_dim_idxs) const {
        grid_info_t ret(int(old_dim_idxs.size()));
        int new_dim_idx = 0;
        for (auto old_dim_idx : old_dim_idxs) {
            ret.dim(new_dim_idx) = dim(old_dim_idx);
            ret.off(new_dim_idx) = off(old_dim_idx);
            ret.idx(new_dim_idx) = idx(old_dim_idx);
            new_dim_idx++;
        }
        return ret;
    }

    grid_info_t slice(int dim_idx, int new_off, int new_dim,
            const expr_t &new_idx, expr_t &new_idx_value) const {
        ir_assert(dim_idx >= 0 && dim_idx < ndims());
        ir_assert(new_dim > 0 && new_off >= 0);
        ir_assert(new_off + new_dim <= dims_[dim_idx]);

        grid_info_t ret = *this;
        ret.offs_[dim_idx] += new_off;
        ret.dims_[dim_idx] = new_dim;
        if (new_off > 0) {
            new_idx_value = ret.idxs_[dim_idx] - new_off;
            ret.idxs_[dim_idx] = new_idx;
        } else {
            new_idx_value = expr_t();
        }
        ret.parent_dims_ = (parent_dims_.empty() ? dims_ : parent_dims_);
        return ret;
    }

    grid_info_t halven(const expr_t &new_idx, int &dim_idx,
            expr_t &new_idx_value, bool first = true) const {
        for (int i = ndims() - 1; i >= 0; i--) {
            if (dim(i) == 1 || dim(i) % 2 != 0) continue;
            dim_idx = i;
            if (first) return slice(i, 0, dim(i) / 2, new_idx, new_idx_value);
            return slice(i, dim(i) / 2, dim(i) / 2, new_idx, new_idx_value);
        }
        return grid_info_t();
    }

    expr_t slice_condition() const {
        if (parent_dims_.empty()) return expr_t();
        expr_t ret(true);
        for (int i = 0; i < ndims(); i++) {
            auto &idx = idxs_[i];
            if (offs_[i] > 0) ret &= (idx >= 0);
            if (offs_[i] + dims_[i] < parent_dims_[i]) ret &= (idx < dims_[i]);
        }
        if (ret.is_equal(expr_t(true))) return expr_t();
        return ret;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << ir_utils::make_seq_print_helper(dims_, "x");
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    std::vector<int> dims_;
    std::vector<int> offs_;
    std::vector<expr_t> idxs_;

    std::vector<int> parent_dims_;
};

inline std::ostream &operator<<(
        std::ostream &out, const grid_info_t &grid_info) {
    out << grid_info.str();
    return out;
}

class grid_splitter_t {
public:
    grid_splitter_t(const grid_info_t &grid)
        : grid_(grid), cur_idx_(grid.ndims() - 1), cur_stride_(1) {
        while (cur_idx_ >= 0 && grid_.dim(cur_idx_) == 1)
            cur_idx_--;
        ir_assert(cur_idx_ >= 0);
    }

    int cur_block() const {
        if (is_empty()) return 1;

        return grid_.dim(cur_idx_) / cur_stride_;
    }

    bool is_empty() const { return cur_idx_ == -1; }

    bool can_pop_block(int size) const {
        if (is_empty()) return false;
        return cur_block() % size == 0;
    }

    expr_t pop_block(int size);

private:
    grid_info_t grid_;

    int cur_idx_;
    int cur_stride_;
};

// Used to describe semantics of a dimension in the GEMM context.
enum class mnk_kind_t { undef, m, n, k };

class mnk_tensor_t {
public:
    mnk_tensor_t(const std::vector<mnk_kind_t> &mnk_kinds,
            const std::vector<dim_t> &dims)
        : mnk_kinds_(mnk_kinds), dims_(dims) {
        ir_assert(mnk_kinds.size() == dims.size()) << "Inconsistent sizes.";
    }

    int ndims() const { return int(mnk_kinds_.size()); }

    const std::vector<mnk_kind_t> &mnk_kinds() const { return mnk_kinds_; }

    dim_t elems() const {
        dim_t ret = 1;
        for (auto d : dims_)
            ret *= d;
        return ret;
    }

    dim_t dim(mnk_kind_t mnk_kind) const {
        ir_assert(has(mnk_kind)) << "Dimension not found.";
        for (int i = 0; i < ndims(); i++) {
            if (mnk_kinds_[i] == mnk_kind) return dims_[i];
        }
        return -1;
    }

    bool has(mnk_kind_t mnk_kind) const {
        for (int i = 0; i < ndims(); i++) {
            if (mnk_kinds_[i] == mnk_kind) return true;
        }
        return false;
    }

private:
    std::vector<mnk_kind_t> mnk_kinds_;
    std::vector<dim_t> dims_;
};

class mask_vector_t {
public:
    mask_vector_t() = default;

    mask_vector_t(const type_t &type, const std::vector<int> &masks,
            const object_eq_map_t<expr_t, int> &mask2ids,
            const std::vector<expr_t> &id2masks)
        : type_(type)
        , elems_(dim_t(masks.size()))
        , masks_(masks)
        , mask2ids_(mask2ids)
        , id2masks_(id2masks) {}

    mask_vector_t(const type_t &type, dim_t elems)
        : type_(type), elems_(elems), masks_(elems, -1) {}

    const type_t &type() const { return type_; }

    dim_t elems() const { return elems_; }

    void set_mask(dim_t off, const expr_t &mask) {
        ir_assert(0 <= off && off < elems()) << "Incorrect offset.";
        if (mask.is_empty()) return;

        auto ret = mask2ids_.insert({mask, int(mask2ids_.size())});
        int id = ret.first->second;
        masks_[off] = id;

        if (ret.second) id2masks_.push_back(mask);
    }

    void simplify(const constraint_set_t &cset) {
        for (auto &mask : id2masks_) {
            auto new_mask = jit::simplify(mask, cset);
            // Some complex expressions need more than one simplify() call.
            int max_tries = 5;
            for (int i = 0; i < max_tries; i++) {
                mask = new_mask;
                new_mask = jit::simplify(new_mask, cset);
                if (new_mask.is_equal(mask)) break;
            }
        }
        mask2ids_.clear();
        for (int i = 0; i < int(id2masks_.size()); i++) {
            auto ret = mask2ids_.insert({id2masks_[i], i});
            if (!ret.second) {
                for (auto &m : masks_)
                    if (m == i) m = ret.first->second;
            }
        }
    }

    std::string str() const {
        std::ostringstream oss;
        for (int i = 0; i < int(elems()); i++) {
            if (i != 0) oss << std::endl;
            oss << "mask #" << i << ": ";
            if (masks_[i] == -1) {
                oss << "(nil)";
            } else {
                oss << id2masks_[masks_[i]];
            }
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

    bool is_empty() const { return elems_ == 0; }

    bool is_true() const {
        if (id2masks_.size() != 1) return false;
        return id2masks_[0].is_equal(expr_t(true));
    }

    mask_vector_t reinterpret(const type_t &new_type) const;

    expr_t to_expr(int nmasks) const;

private:
    type_t type_;
    dim_t elems_ = 0;
    std::vector<int> masks_;

    object_eq_map_t<expr_t, int> mask2ids_;
    std::vector<expr_t> id2masks_;
};

inline std::ostream &operator<<(
        std::ostream &out, const mask_vector_t &mask_vector) {
    out << mask_vector.str();
    return out;
}

class tdim_info_t {
public:
    tdim_info_t() = default;

    tdim_info_t(const expr_t &expr, const expr_t &mask)
        : expr_(expr), mask_(mask) {}

    int nvargs() const { return nvargs_; }

    const expr_t &expr() const { return expr_; }

    const expr_t &mask() const { return mask_; }

    expr_t mask(const expr_t &tvalue, const std::vector<expr_t> &vvars,
            const std::vector<expr_t> &vvalues) const {
        auto ret = substitute(mask_, placeholder_var(), tvalue);
        for (int i = 0; i < int(vvars.size()); i++) {
            if (contains_object(ret, vvars[i])) {
                ret = substitute(ret, vvars[i], vvalues[i]);
            }
        }
        return ret;
    }

    int vidx(int arg_idx) const {
        ir_assert(arg_idx < nvargs());
        return vidxs_[arg_idx];
    }

    stride_t vstride(int arg_idx) const {
        ir_assert(arg_idx < nvargs());
        return vstrides_[arg_idx];
    }

    bool is_empty() const { return expr_.is_empty(); }

    bool is_identity() const { return is_var(expr_); }

    bool is_fixed_stride(int arg_idx) const {
        ir_assert(arg_idx < nvargs());
        return vstrides_[arg_idx].is_fixed();
    }

    void add_vvar(int vidx, const expr_t &varg) {
        ir_assert(nvargs_ + 1 <= max_nvargs);
        vidxs_[nvargs_] = vidx;
        vstrides_[nvargs_] = compute_stride(expr_, nvargs_, varg);
        nvargs_++;
    }

    static const expr_t &placeholder_var() {
        static expr_t ph_var = var_t::make(type_t::s32(), "_ph");
        return ph_var;
    }

private:
    static const int max_nvargs = 2;

    static stride_t compute_stride(const expr_t &e, int idx, const expr_t &var);

    expr_t expr_;

    int nvargs_ = 0;
    std::array<stride_t, max_nvargs> vstrides_;
    std::array<int, max_nvargs> vidxs_;
    expr_t mask_;
};

class view_t {
public:
    view_t() = default;

    view_t(const std::vector<expr_t> &vvars, int ntdims)
        : vvars_(vvars)
        , vdims_(vvars.size())
        , vstart_(vvars.size())
        , vmnk_kinds_(vvars.size())
        , tdims_(ntdims) {}

    // Constructs a direct view.
    view_t(const view_t &other, const layout_t &vlayout)
        : is_direct_(true)
        , vvars_(other.vvars_)
        , vdims_(other.vdims_)
        , vstart_(other.vstart_)
        , vdirect_start_(other.nvdims(), 0)
        , vmnk_kinds_(other.vmnk_kinds_)
        , tlayout_(vlayout) {
        if (other.is_direct()) {
            for (int i = 0; i < nvdims(); i++) {
                auto &s = other.vdirect_start_[i];
                if (!is_zero(s)) vstart_[i] += s;
            }
        }
    }

    view_t(const layout_t &tlayout, const std::vector<expr_t> &vvars,
            uint32_t bound_check_mask = 0)
        : view_t(vvars, tlayout.ndims()) {
        vdirect_start_.resize(nvdims(), 0);
        for (int i = 0; i < tlayout.ndims(); i++) {
            expr_t i_mask;
            if ((bound_check_mask & (1 << i)) != 0)
                i_mask = (placeholder_var() < tlayout.dim(i));
            set_vdim(vvars_[i], tlayout.dim(i), 0);
            set_tdim(i, vvars_[i], i_mask);
        }
        set_tlayout(tlayout);
    }

    bool is_direct() const { return is_direct_; }

    const std::vector<expr_t> &vvars() const { return vvars_; }

    const std::vector<dim_t> &vdims() const { return vdims_; }

    expr_t vstart(int vidx) const {
        if (!is_direct_ || is_zero(vdirect_start_[vidx])) return vstart_[vidx];
        return vstart_[vidx] + vdirect_start_[vidx];
    }

    const std::vector<mnk_kind_t> &vmnk_kinds() const { return vmnk_kinds_; }

    const layout_t tlayout() const { return tlayout_; }

    int nvdims() const { return int(vdims_.size()); }

    int ntdims() const {
        if (is_direct_) return nvdims();
        return int(tdims_.size());
    }

    dim_t velems() const {
        dim_t ret = 1;
        for (int i = 0; i < nvdims(); i++)
            ret *= vdims_[i];
        return ret;
    }

    const expr_t &vvar(int idx) const {
        ir_assert(idx < nvdims());
        return vvars_[idx];
    }

    void set_tdim(int tidx, const expr_t &_texpr, expr_t mask = {}) {
        ir_assert(!is_direct_ && tdims_[tidx].is_empty());

        auto texpr = simplify(_texpr);
        ir_assert(!is_const(texpr)) << "Tensor dimension can't be a constant.";

        tdim_info_t tdim(texpr, mask);
        for (int i = 0; i < nvdims(); i++) {
            if (contains_object(texpr, vvars_[i])) tdim.add_vvar(i, vvars_[i]);
        }
        ir_assert(tdim.nvargs() > 0)
                << "Tensor dimension must have at least one "
                   "view dimension that maps to it.";
        tdims_[tidx] = tdim;
    }

    void set_vdim(const expr_t &varg, dim_t vdim, const expr_t &vstart,
            mnk_kind_t vmnk_kind = mnk_kind_t::undef) {
        ir_assert(!is_direct_);
        int vidx = vvar_index(varg);
        ir_assert(vstart_[vidx].is_empty());
        vstart_[vidx] = vstart;
        vdims_[vidx] = vdim;
        vmnk_kinds_[vidx] = vmnk_kind;
    }

    void set_tlayout(const layout_t &tlayout) {
        ir_assert(!is_direct_);
        tlayout_ = tlayout;
    }

    std::string str() const {
        using ir_utils::operator<<;

        if (is_empty()) return "(nil)";
        std::ostringstream oss;
        oss << ir_utils::make_seq_print_helper(vdims_, "x");
        if (!has_zero_vstart()) oss << " vstart: [" << vstart_ << "]";
        if (!has_zero_vdirect_start())
            oss << " vdirect_start: [" << vdirect_start_ << "]";
        oss << " tlayout: " << tlayout_;
        return oss.str();
    }

    IR_DEFINE_DUMP()

    bool is_empty() const { return vdims_.empty(); }

    bool has_zero_vstart() const {
        for (int i = 0; i < nvdims(); i++)
            if (!is_zero(vstart_[i])) return false;
        return true;
    }

    bool has_zero_vdirect_start() const {
        if (vdirect_start_.empty()) return true;

        for (int i = 0; i < nvdims(); i++)
            if (!is_zero(vdirect_start_[i])) return false;
        return true;
    }

    bool has_tmask(int tidx) const {
        ir_assert(tidx >= 0 && tidx < ntdims());
        return !tdims_[tidx].mask().is_empty();
    }

    const type_t &type() const { return tlayout_.type(); }

    expr_t offset(const std::vector<expr_t> &vargs = {},
            bool ignore_offset = false) const {
        auto targs = cvt_vargs_to_targs(vargs);
        return tlayout_.offset(targs, ignore_offset);
    }

    expr_t offset_in_bytes(const std::vector<expr_t> &vargs = {},
            bool ignore_offset = false) const {
        return offset(vargs, ignore_offset) * type().size();
    }

    tensor_t vtensor(bool force_zero_start = false) const {
        if (force_zero_start) return tensor_t(vdims());

        std::vector<expr_t> start(nvdims());
        for (int i = 0; i < nvdims(); i++)
            start[i] = vstart(i);
        return tensor_t(vdims(), start);
    }

    int vvar_index(const expr_t &vvar) const {
        for (size_t i = 0; i < vvars_.size(); i++)
            if (vvar.is_same(vvars_[i])) return int(i);
        ir_error_not_expected() << "Can't find view dimension.";
        return -1;
    }

    template <typename T>
    T operator()(const std::vector<T> &vargs) const {
        auto targs = cvt_vargs_to_targs(vargs);
        return tlayout_(targs);
    }

    view_t create_sub_view(
            const tensor_t &sub_tensor, bool relative_vstart = true) const;

    view_t retype(const type_t &new_type) const {
        auto ret = *this;
        ret.tlayout_ = tlayout_.retype(new_type);
        return ret;
    }

    view_t make_dense() const {
        auto ret = *this;
        ret.tlayout_ = tlayout_.make_dense();
        return ret;
    }

    bool can_convert_to_vlayout() const {
        if (is_direct_) return true;
        if (nvdims() != ntdims()) return false;
        for (int i = 0; i < nvdims(); i++) {
            if (!tdims_[i].expr().is_same(vvars_[i])) return false;
            if (!tdims_[i].is_fixed_stride(0)) return false;
        }
        return true;
    }

    // FIXME: Offset of the returned layout is always 0.
    layout_t create_pseudo_vlayout() const {
        if (is_direct_) return create_vlayout(/*force_zero_offset=*/true);
        return create_pseudo_vlayout(tlayout_);
    }

    layout_t create_dense_vlayout() const {
        return create_pseudo_vlayout().make_dense();
    }

    layout_t create_vlayout(bool force_zero_offset = false) const {
        ir_assert(can_convert_to_vlayout()) << "Can't convert view to layout.";
        if (force_zero_offset) return tlayout_.map(tensor_t(vdims_));
        if (is_direct_) return tlayout_.map(tensor_t(vdims_, vdirect_start_));
        return tlayout_.map(tensor_t(vdims_, vstart_));
    }

    dim_t vlayout_size() const { return create_vlayout().size(); }

    bool has_same_vlayout(
            const view_t &other, bool compare_offset = true) const {
        return create_vlayout().is_equal(
                other.create_vlayout(), compare_offset);
    }

    view_t split(const grid_info_t &grid) const;

    // Outer blocks are filled from innermost to outermost.
    view_t split(const mnk_tensor_t &mnk_tile, const grid_info_t &grid,
            std::vector<block_t> *outer_blocks = nullptr) const;

    // Tile is assumed to be dense.
    tensor_t split_into_dense_tile(
            dim_t &tile_elems, dim_t &outer_block) const {
        auto vlayout = create_pseudo_vlayout();
        std::vector<dim_t> blocks = {tile_elems, outer_block};
        vlayout = vlayout.split_into_multi_blocks_with_hint(blocks);
        if (vlayout.is_empty()) return tensor_t();
        tile_elems = blocks[0];
        outer_block = blocks[1];
        return vlayout.split_into_dense_tile(tile_elems, outer_block);
    }

    // Returns a tensor corresponding to the biggest innermost sub-layout so that
    // 1) It consists of consecutive blocks only.
    // 2) It contains less or equal than max_tile_elems elements.
    // 3) It is dense if is_dense_tile is true.
    tensor_t split_into_max_tile(
            dim_t max_tile_elems, bool is_dense_tile) const {
        auto vlayout = create_pseudo_vlayout();
        return vlayout.split_into_max_tile(max_tile_elems, is_dense_tile);
    }

    template <typename F>
    void for_each_tile(const tensor_t &tile, const F &f) const {
        auto vlayout = create_dense_vlayout();
        vlayout.for_each_tile(tile, f);
    }

    view_t substitute(const expr_t &from, const expr_t &to) const;

    mask_vector_t create_mask_vector() const {
        ir_assert(!is_direct_) << "Direct views don't use masks.";
        auto _vlayout = create_dense_vlayout();
        mask_vector_t mask_vec(type(), _vlayout.elems());
        std::vector<dim_t> vargs(nvdims());
        create_mask_vector(mask_vec, _vlayout, 0, vargs);
        return mask_vec;
    }

    static const expr_t &placeholder_var() {
        return tdim_info_t::placeholder_var();
    }

private:
    template <typename SrcT = expr_t, typename DstT = SrcT>
    std::vector<DstT> cvt_vargs_to_targs(
            const std::vector<SrcT> &_vargs = {}) const {
        std::vector<expr_t> vargs = expr_cast<expr_t>(_vargs);
        if (vargs.empty()) vargs.resize(nvdims(), 0);

        for (int i = 0; i < nvdims(); i++) {
            if (!is_direct_ && !is_zero(vstart_[i])) vargs[i] += vstart_[i];
            if (is_direct_ && !is_zero(vdirect_start_[i]))
                vargs[i] += vdirect_start_[i];
        }

        if (is_direct_) return expr_cast<DstT>(vargs);

        std::vector<expr_t> targs(ntdims());
        for (int i = 0; i < ntdims(); i++) {
            targs[i] = tdims_[i].expr();
            for (int j = 0; j < nvdims(); j++) {
                targs[i] = jit::substitute(targs[i], vvars_[j], vargs[j]);
            }
        }
        return expr_cast<DstT>(targs);
    }

    layout_t create_pseudo_vlayout(const layout_t &tlayout) const;

    view_t split(const layout_t &vlayout, const tensor_t &vtile,
            const grid_info_t &grid,
            std::vector<block_t> *outer_blocks = nullptr) const;

    void create_mask_vector(mask_vector_t &mask_vec, const layout_t &_vlayout,
            int vidx, std::vector<dim_t> &vargs) const;

    bool is_direct_ = false;
    std::vector<expr_t> vvars_;
    std::vector<dim_t> vdims_;
    std::vector<expr_t> vstart_;
    std::vector<expr_t> vdirect_start_;
    std::vector<mnk_kind_t> vmnk_kinds_;

    std::vector<tdim_info_t> tdims_;
    layout_t tlayout_;
};

inline std::ostream &operator<<(std::ostream &out, const view_t &view) {
    out << view.str();
    return out;
}

class mnk_mapper_t {
public:
    mnk_mapper_t() = default;

    void push_view(const view_t &ab_view, const view_t &c_view) {
        ir_assert(ab_view.is_direct());
        auto layout = ab_view.create_pseudo_vlayout();
        for (auto &b : layout.blocks()) {
            push_block(b, ab_view, c_view);
        }
    }

    void push_blocks(const std::vector<block_t> &blocks, const view_t &ab_view,
            const view_t &c_view) {
        for (auto &b : blocks)
            push_block(b, ab_view, c_view);
    }

    void push_block(
            const block_t &_b, const view_t &ab_view, const view_t &c_view);

    layout_t map_to_mnk(
            const view_t &view, const std::vector<mnk_kind_t> &mnk_kinds) const;

    layout_t map_to_mnk(const layout_t &layout, const view_t &view,
            const std::vector<mnk_kind_t> &mnk_kinds) const;

    layout_t map_from_mnk(const layout_t &mnk_layout, int prb_ndims) const;

private:
    static void pop_size_1_blocks(std::vector<block_t> &blocks) {
        while (!blocks.empty() && blocks.front().block == 1) {
            blocks.erase(blocks.begin());
        }
    }

    bool pop_block(std::vector<block_t> &mnk_blocks,
            std::vector<block_t> &prb_blocks, const block_t &mnk_block) const;

    // Ordered from innermost to outermost.
    std::vector<block_t> m_blocks_;
    std::vector<block_t> n_blocks_;
};

class dim_assignment_t {
public:
    dim_assignment_t() = default;

    dim_assignment_t(int old_ndims, int new_ndims)
        : old_ndims_(old_ndims)
        , new_ndims_(new_ndims)
        , assignments_(old_ndims, -1) {}

    void assign(int old_idx, int new_idx) {
        ir_assert(0 <= old_idx && old_idx < old_ndims_);
        ir_assert(0 <= new_idx && new_idx < new_ndims_);
        assignments_[old_idx] = new_idx;
    }

    void assign(const std::vector<int> &old_idxes, int new_idx) {
        for (auto old_idx : old_idxes) {
            assign(old_idx, new_idx);
        }
    }

    int operator[](int old_idx) const {
        ir_assert(old_idx >= 0 && old_idx < old_ndims());
        return assignments_[old_idx];
    }

    int old_ndims() const { return old_ndims_; }

    int new_ndims() const { return new_ndims_; }

    bool is_empty() const { return old_ndims_ == 0 && new_ndims_ == 0; }

    layout_t map(const layout_t &layout) const;

private:
    int old_ndims_ = 0;
    int new_ndims_ = 0;

    // assignments_[old_idx] = new_idx.
    std::vector<int> assignments_;
};

// Adds size one spatial dimensions according to input parameters. Spatial
// dimensions are assumed to be the last dimensions.
layout_t normalize_spatial(
        const layout_t &layout, int old_sp_ndims, bool reduced_to_1d);

std::vector<dim_t> normalize_spatial(
        const std::vector<dim_t> &dims, int old_sp_ndims, bool reduced_to_1d);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

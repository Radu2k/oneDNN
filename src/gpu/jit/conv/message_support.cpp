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

#include "gpu/jit/conv/message_support.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

expr_t create_mask_expr(
        const constraint_set_t &cset, const send_t &send, const view_t &view) {
    if (view.is_direct()) return expr_t();

    auto mask_vec = view.create_mask_vector();
    mask_vec.simplify(cset);

    ir_assert(mask_vec.elems() * mask_vec.type().size() == send.eff_size())
            << "Unexpected mask.";

    type_t mask_type = (send.mask_granularity() == mask_granularity_t::per_dword
                    ? type_t::dword()
                    : send.data_type.with_elems(send.data_elems));
    auto mask_vec_retyped = mask_vec.reinterpret(mask_type);

    // Couldn't reinterpret.
    if (mask_vec_retyped.is_empty()) return expr_t();

    return mask_vec_retyped.to_expr(send.eff_mask_count);
}

stmt_t create_send_stmt(const constraint_set_t &cset, const send_t &send,
        const expr_t &mem_buf, const expr_t &reg_buf, const view_t &view) {
    using namespace ir_utils;
    ir_assert(mem_buf.type().is_ptr()) << mem_buf;
    ir_assert(reg_buf.type().is_ptr()) << reg_buf;

    expr_t mask_expr = create_mask_expr(cset, send, view);

    // Do not use mask if all its elements are true.
    if (!mask_expr.is_empty()
            && mask_expr.is_equal(shuffle_t::make_broadcast(
                    expr_t(true), mask_expr.type().elems()))) {
        mask_expr = expr_t();
    }

    // TODO: Check alignment.
    switch (send.type) {
        case message_type_t::block: {
            expr_t off_in_bytes = simplify(view.offset_in_bytes());
            return send(mem_buf, off_in_bytes, reg_buf, mask_expr);
        }
        case message_type_t::scattered: {
            int elems_per_slot = send.block_size() / view.type().size();
            auto slot_tile = view.split_into_max_tile(
                    elems_per_slot, /*is_dense=*/true);
            std::vector<expr_t> off_vec;
            view.for_each_tile(slot_tile, [&](const std::vector<dim_t> &start) {
                auto estart = expr_cast<expr_t>(start);
                off_vec.push_back(simplify(view.offset_in_bytes(estart)));
            });
            return send(mem_buf, shuffle_t::make(off_vec), reg_buf, mask_expr);
        }
        default: ir_error_not_expected();
    }
    return stmt_t();
}

// Returns register layout corresponding to the message.
layout_t create_raw_register_layout(const send_t &send) {
    std::vector<block_t> spec_blocks;

    // Message reads (and maybe transforms) (slots x data_elems) tensor.
    spec_blocks.emplace_back(1, send.data_elems, send.data_elems_stride());
    spec_blocks.emplace_back(0, send.slots, send.slots_stride());

    if (send.is_transposing()) std::swap(spec_blocks[0], spec_blocks[1]);

    // TODO: Use extra outer block to force stride when writeback buffer size
    // is rounded up.
    return layout_t(send.data_type, 2, 0, spec_blocks);
}

// Returns dense memory layout corresponding to the message.
layout_t create_raw_memory_layout(const send_t &send) {
    if (send.type == message_type_t::block
            && send.eff_mask_count != send.mask_count()) {
        std::vector<dim_t> dims = {1, send.eff_mask_count};
        return layout_t(type_t::dword(), 0, dims);
    }
    std::vector<dim_t> dims = {send.eff_slots(), send.data_elems};
    return layout_t(send.data_type, 0, dims);
}

// Converts memory layout to register layout according to the message.
layout_t create_register_layout_for_message(
        const send_t &send, const layout_t &_mem_layout) {
    // Message tensor: (slots x data_elems) XaYb.
    auto msg_mem_layout
            = create_raw_memory_layout(send).reinterpret(_mem_layout.type());
    ir_assert(msg_mem_layout.is_plain());
    int slots = int(msg_mem_layout.dims()[0]);
    int data_elems = int(msg_mem_layout.dims()[1]);

    // Split memory layout according to the message specification.
    auto mem_layout = _mem_layout.split_into_multi_blocks({data_elems, slots});

    std::vector<block_t> b_slots;
    std::vector<block_t> b_data_elems;
    std::vector<block_t> b_rest;

    int rem_slots = slots;
    int rem_data_elems = data_elems;
    for (auto &b : mem_layout.blocks()) {
        auto b_copy = b;
        b_copy.stride = 0;
        if (rem_data_elems > 1) {
            ir_assert(rem_data_elems % b.block == 0);
            rem_data_elems /= b.block;
            b_data_elems.push_back(b_copy);
            continue;
        }
        if (rem_slots > 1) {
            ir_assert(rem_slots % b.block == 0);
            rem_slots /= b.block;
            b_slots.push_back(b_copy);
            continue;
        }
        b_rest.push_back(b_copy);
    }

    ir_assert(rem_slots == 1);
    ir_assert(rem_data_elems == 1);

    int ndims = mem_layout.ndims();
    layout_t msg_reg_layout
            = create_raw_register_layout(send).reinterpret(_mem_layout.type());

    // Dummy block for stride between slots.
    block_t dummy_slots_block(ndims, 1, msg_reg_layout.strides(0)[0]);

    // Dummy block for stride between data elements.
    block_t dummy_data_elems_block(ndims + 1, 1, msg_reg_layout.strides(1)[0]);

    // Dummy block for stride between messages.
    block_t dummy_msg_block(
            ndims + 2, 1, send.register_size() / msg_reg_layout.type().size());

    // Sanity check to ensure it's divisible.
    ir_assert(
            dim_t(dummy_msg_block.stride * dim_t(msg_reg_layout.type().size()))
            == send.register_size());

    // Prepare blocks for the translated layout.
    std::vector<block_t> v;
    if (send.is_transposing()) {
        v.insert(v.end(), dummy_slots_block);
        v.insert(v.end(), b_slots.begin(), b_slots.end());
        v.insert(v.end(), dummy_data_elems_block);
        v.insert(v.end(), b_data_elems.begin(), b_data_elems.end());
    } else {
        v.insert(v.end(), dummy_data_elems_block);
        v.insert(v.end(), b_data_elems.begin(), b_data_elems.end());
        v.insert(v.end(), dummy_slots_block);
        v.insert(v.end(), b_slots.begin(), b_slots.end());
    }
    v.insert(v.end(), dummy_msg_block);
    v.insert(v.end(), b_rest.begin(), b_rest.end());

    // Fix strides.
    auto &v0 = v.front();
    dim_t stride = v0.stride;
    for (auto &b : v) {
        if (&b != &v0 && dim_t(b.stride) == 0) b.stride = stride;
        stride = b.block * b.stride;
    }

    // Remove dummy dimensions.
    auto it = std::remove_if(v.begin(), v.end(),
            [&](const block_t &b) { return b.dim_idx >= ndims; });
    v.erase(it, v.end());

    return layout_t(mem_layout.type(), mem_layout.ndims(), 0, v);
}

view_t create_register_view_for_message(
        const send_t &send, const view_t &mem_view, int &reg_buf_size) {
    auto mem_layout = mem_view.create_dense_vlayout();
    auto reg_layout = create_register_layout_for_message(send, mem_layout);
    reg_buf_size = utils::rnd_up(reg_layout.size(), reg_bytes);
    return view_t(mem_view, reg_layout);
}

bool has_compatible_mask(
        const constraint_set_t &cset, const send_t &send, const view_t &view) {
    if (view.is_direct()) return true;
    auto mask_expr = create_mask_expr(cset, send, view);
    return !mask_expr.is_empty();
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

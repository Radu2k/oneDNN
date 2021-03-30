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

#ifndef GPU_JIT_CONV_MESSAGE_SUPPORT_HPP
#define GPU_JIT_CONV_MESSAGE_SUPPORT_HPP

#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/tensor.hpp"
#include "gpu/jit/conv/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// TODO: Make a send_t parameter.
const int reg_bytes = 32;

enum class message_type_t {
    block,
    scattered,
};

enum class mask_granularity_t {
    undef,
    per_slot,
    per_dword,
};

// Function representing send messages.
class send_t : public func_impl_t {
public:
    IR_DECL_DERIVED_TYPE_ID(send_t, func_impl_t)

    static func_t make(ngen_proxy::Access access_type, message_type_t type,
            const type_t &data_type, int data_elems, int slots,
            const type_t &alignment, ngen_proxy::AddressModel address_model,
            int eff_mask_count = -1) {
        return func_t(new send_t(access_type, type, data_type, data_elems,
                slots, alignment, address_model, eff_mask_count));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (access_type == other.access_type) && (type == other.type)
                && (data_type == other.data_type)
                && (data_elems == other.data_elems) && (slots == other.slots)
                && (alignment == other.alignment)
                && (address_model == other.address_model)
                && (eff_mask_count == other.eff_mask_count);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(access_type, type, data_type, data_elems,
                slots, alignment, address_model);
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << "send." << slots << "x" << data_elems << "x" << data_type.str();
        return oss.str();
    }

    IR_DEFINE_ARG_GET(mem_buf, 0)
    IR_DEFINE_ARG_GET(mem_off, 1)
    IR_DEFINE_ARG_GET(reg_buf, 2)
    IR_DEFINE_ARG_GET(mask, 3)

    stmt_t operator()(const expr_t &mem_buf, const expr_t &mem_off,
            const expr_t &reg_buf, const expr_t &mask) const {
        return call({mem_buf, mem_off, reg_buf, mask});
    }

    bool is_read() const { return access_type == ngen_proxy::Access::Read; }

    bool is_write() const { return access_type == ngen_proxy::Access::Write; }

    // Size of elements to read/write in bytes.
    int size() const { return block_size() * slots; }

    int eff_size() const { return eff_block_size() * eff_slots(); }

    mask_granularity_t mask_granularity() const {
        switch (type) {
            case message_type_t::block: return mask_granularity_t::per_dword;
            case message_type_t::scattered: return mask_granularity_t::per_slot;
            default: ir_error_not_expected();
        }
        return mask_granularity_t::undef;
    }

    bool is_per_dword_mask() {
        return mask_granularity() == mask_granularity_t::per_dword;
    }
    bool is_per_slot_mask() {
        return mask_granularity() == mask_granularity_t::per_slot;
    }

    int mask_count() const {
        switch (type) {
            case message_type_t::block:
                return std::min(16, block_size() / int(sizeof(uint32_t)));
            case message_type_t::scattered: return slots;
            default: ir_error_not_expected();
        }
        return -1;
    }

    int eff_slots() const {
        if (eff_mask_count == mask_count()) return slots;
        if (mask_granularity() == mask_granularity_t::per_slot)
            return eff_mask_count;
        return slots;
    }

    // Stride between slots in elements of data_type (in memory).
    int slots_stride() const {
        if (type == message_type_t::block) return data_elems;
        if (data_type == type_t::byte()) return 4;
        return 1;
    }

    // Stride between data elements in elements of data_type (in memory).
    int data_elems_stride() const {
        if (type == message_type_t::block) return 1;
        if (data_type == type_t::byte()) return 1;
        return slots;
    }

    // Size of the innermost dense block in bytes (in memory).
    int block_size() const { return data_type.size() * data_elems; }

    // Effective size of the innermost dense block in bytes (in memory).
    int eff_block_size() const {
        if (eff_mask_count == mask_count()) return block_size();
        if (mask_granularity() == mask_granularity_t::per_dword) {
            int max_mask_count = 16;
            // Do not allow strided blocks.
            ir_assert(block_size() <= max_mask_count * int(sizeof(uint32_t)));
            MAYBE_UNUSED(max_mask_count);
            return eff_mask_count * int(sizeof(uint32_t));
        }
        return data_type.size() * data_elems;
    }

    // Size of the register buffer in bytes.
    int register_size() const {
        int sz;
        if (is_transposing()) {
            sz = data_elems * data_elems_stride();
        } else {
            sz = slots * slots_stride();
        }
        sz *= data_type.size();
        // Round up to the full register length.
        return utils::rnd_up(sz, reg_bytes);
    }

    // Size of address elements.
    int address_size() const {
        return (address_model == ngen_proxy::AddressModel::ModelA64) ? 8 : 4;
    }

    type_t address_type(bool is_signed = false, int elems = 1) const {
        int bits = address_size() * 8;
        return is_signed ? type_t::s(bits, elems) : type_t::u(bits, elems);
    }

    // Size of header in bytes.
    int header_size() const {
        if (type == message_type_t::block) return reg_bytes;
        return utils::rnd_up(address_size() * slots, reg_bytes);
    }

    bool is_transposing() const { return data_elems_stride() > slots_stride(); }

    func_t adjust(int new_block_size, int new_slots) const {
        ir_assert(new_block_size > 0 && new_slots > 0);
        if (new_block_size == block_size() && new_slots == slots) return this;
        int max_mask_count = 16;
        int new_mask_count = -1;
        switch (mask_granularity()) {
            case mask_granularity_t::per_dword: {
                if (block_size() > max_mask_count * int(sizeof(uint32_t)))
                    return func_t();
                if (new_block_size > max_mask_count * int(sizeof(uint32_t)))
                    return func_t();
                if (new_block_size % int(sizeof(uint32_t)) != 0)
                    return func_t();
                new_mask_count = new_block_size / int(sizeof(uint32_t));
                break;
            }
            case mask_granularity_t::per_slot:
                if (new_block_size != block_size()) return func_t();
                if (new_slots > max_mask_count) return func_t();
                new_mask_count = new_slots;
                break;
            default: ir_error_not_expected(); return func_t();
        }
        return send_t::make(access_type, type, data_type, data_elems, slots,
                alignment, address_model, new_mask_count);
    }

    // Generates a statement to store (and maybe convert) the offset to the
    // message header according to the message description.
    stmt_t create_offset_store(const expr_t &header_buf, const expr_t &mem_buf,
            const expr_t &mem_off, bool is_signed_offset = false) const {
        bool is_block = (type == message_type_t::block);
        bool is_a64 = (address_model == ngen_proxy::AddressModel::ModelA64);
        bool is_bts = (address_model == ngen_proxy::AddressModel::ModelBTS);
        bool is_slm = (address_model == ngen_proxy::AddressModel::ModelSLM);

        expr_t header_sub_buf;
        expr_t off;
        if (is_block && (is_slm || is_bts)) {
            // Convert byte offset to dwords/owords/hwords offset.
            off = mem_off / data_type.size();
            header_sub_buf = header_buf[2 * sizeof(uint32_t)];
        } else if (is_a64) {
            // Convert buffer to 64-bit integer.
            off = cast(mem_buf, type_t::u64());
            if (mem_off.type().is_vector())
                off = shuffle_t::make_broadcast(off, mem_off.type().elems());
            off += mem_off;
            header_sub_buf = header_buf[0];
        } else {
            ir_error_not_expected();
        }
        off = cast(off, address_type(is_signed_offset, off.type().elems()));
        return store_t::make(header_sub_buf, 0, off);
    }

    static func_t scattered_byte_read(int elems, int slots) {
        return send_t::make(ngen_proxy::Access::Read, message_type_t::scattered,
                type_t::byte(), elems, slots, type_t::undef(),
                ngen_proxy::AddressModel::ModelA64);
    }

    static func_t scattered_byte_write(int elems, int slots) {
        return send_t::make(ngen_proxy::Access::Write,
                message_type_t::scattered, type_t::byte(), elems, slots,
                type_t::undef(), ngen_proxy::AddressModel::ModelA64);
    }

    static func_t scattered_dword_read(int elems, int slots) {
        return send_t::make(ngen_proxy::Access::Read, message_type_t::scattered,
                type_t::dword(), elems, slots, type_t::undef(),
                ngen_proxy::AddressModel::ModelA64);
    }

    static func_t scattered_dword_write(int elems, int slots) {
        return send_t::make(ngen_proxy::Access::Write,
                message_type_t::scattered, type_t::dword(), elems, slots,
                type_t::undef(), ngen_proxy::AddressModel::ModelA64);
    }

    static func_t scattered_qword_read(int elems, int slots) {
        return send_t::make(ngen_proxy::Access::Read, message_type_t::scattered,
                type_t::qword(), elems, slots, type_t::undef(),
                ngen_proxy::AddressModel::ModelA64);
    }

    static func_t scattered_qword_write(int elems, int slots) {
        return send_t::make(ngen_proxy::Access::Write,
                message_type_t::scattered, type_t::qword(), elems, slots,
                type_t::undef(), ngen_proxy::AddressModel::ModelA64);
    }

    static func_t block_oword_read(int elems) {
        return send_t::make(ngen_proxy::Access::Read, message_type_t::block,
                type_t::oword(), elems, 1, type_t::undef(),
                ngen_proxy::AddressModel::ModelA64);
    }

    static func_t block_oword_write(int elems) {
        return send_t::make(ngen_proxy::Access::Write, message_type_t::block,
                type_t::oword(), elems, 1, type_t::undef(),
                ngen_proxy::AddressModel::ModelA64);
    }

    static func_t block_oword_read_slm(int elems) {
        return send_t::make(ngen_proxy::Access::Read, message_type_t::block,
                type_t::oword(), elems, 1, type_t::undef(),
                ngen_proxy::AddressModel::ModelSLM);
    }

    static func_t block_oword_write_slm(int elems) {
        return send_t::make(ngen_proxy::Access::Write, message_type_t::block,
                type_t::oword(), elems, 1, type_t::undef(),
                ngen_proxy::AddressModel::ModelSLM);
    }

    static func_t block_hword_read(int elems) {
        return send_t::make(ngen_proxy::Access::Read, message_type_t::block,
                type_t::hword(), elems, 1, type_t::undef(),
                ngen_proxy::AddressModel::ModelA64);
    }

    static std::vector<func_t> get_all() {
        static std::vector<func_t> list;
        static std::once_flag flag;
        std::call_once(flag, [&]() {
            for (int elems : {1, 2, 4, 8, 16}) {
                if (elems <= 8) {
                    list.push_back(block_oword_read(elems));
                    list.push_back(block_oword_write(elems));
                    list.push_back(block_hword_read(elems));
                }
                list.push_back(block_oword_read_slm(elems));
                list.push_back(block_oword_write_slm(elems));
            }
            for (int slots : {8, 16}) {
                for (int elems : {1, 2, 4}) {
                    list.push_back(scattered_byte_read(elems, slots));
                    list.push_back(scattered_byte_write(elems, slots));
                    list.push_back(scattered_dword_read(elems, slots));
                    list.push_back(scattered_dword_write(elems, slots));
                    if (slots * elems <= 32) {
                        list.push_back(scattered_qword_read(elems, slots));
                        list.push_back(scattered_qword_write(elems, slots));
                    }
                }
            }
            // Sort by total size in descending order.
            std::sort(list.begin(), list.end(),
                    [](const func_t &_a, const func_t &_b) {
                        auto &a = _a.as<send_t>();
                        auto &b = _b.as<send_t>();
                        size_t a_sz = a.size();
                        size_t b_sz = b.size();
                        // Put block messages first.
                        if (a.type != b.type)
                            return a.type == message_type_t::block;
                        return a_sz > b_sz;
                    });
        });
        return list;
    }

    template <typename FilterFunc>
    static std::vector<func_t> get_all(const FilterFunc &filter) {
        std::vector<func_t> ret = get_all();
        ret.erase(std::remove_if(ret.begin(), ret.end(),
                          [&](const func_t &s) { return !filter(s); }),
                ret.end());
        return ret;
    }

    ngen_proxy::Access access_type;
    message_type_t type;
    type_t data_type;
    int data_elems;
    int slots;
    type_t alignment;
    ngen_proxy::AddressModel address_model;
    int eff_mask_count;

private:
    send_t(ngen_proxy::Access access_type, message_type_t type,
            const type_t &data_type, int data_elems, int slots,
            const type_t &alignment, ngen_proxy::AddressModel address_model,
            int eff_mask_count)
        : access_type(access_type)
        , type(type)
        , data_type(data_type)
        , data_elems(data_elems)
        , slots(slots)
        , alignment(alignment)
        , address_model(address_model)
        , eff_mask_count(eff_mask_count == -1 ? mask_count() : eff_mask_count) {
        ir_assert(eff_mask_count <= mask_count());
    }
};

// Creates send statement for a memory view.
stmt_t create_send_stmt(const constraint_set_t &cset, const send_t &send,
        const expr_t &mem_buf, const expr_t &reg_buf, const view_t &view);

// Translates a memory view to a register view after send.
view_t create_register_view_for_message(
        const send_t &send, const view_t &mem_view, int &reg_buf_size);

// Checks if send supports the mask defined by a view.
bool has_compatible_mask(
        const constraint_set_t &cset, const send_t &send, const view_t &view);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

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

#include "gpu/jit/conv/kernel_builder.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/fma_support.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/message_support.hpp"
#include "gpu/jit/conv/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Helper class to permute registers. Used to restore registers after applying
// DPAS -> DPASW transformation.
class grf_permutator_t : public ir_mutator_t {
public:
    void set_permute(const expr_t &old_grf, const expr_t &new_grf) {
        auto &old_base = old_grf.as<ptr_t>().base;
        auto &new_base = new_grf.as<ptr_t>().base;
        ir_assert(old_base.is_same(new_base));
        MAYBE_UNUSED(new_base);

        if (grf_buf_base_.is_empty()) grf_buf_base_ = old_base;
        ir_assert(old_base.is_same(grf_buf_base_));

        int old_off = to_cpp<int>(old_grf.as<ptr_t>().off);
        int new_off = to_cpp<int>(new_grf.as<ptr_t>().off);

        auto ret = permutation_.insert({old_off, new_off});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
    }

    object_t _mutate(const load_t *obj) override {
        if (!obj->buf.is_same(grf_buf_base_) || !obj->has_default_stride())
            return ir_mutator_t::_mutate(obj);

        // Ensure no cross-register accesses.
        ir_assert(obj->type.size() <= reg_bytes);

        int load_off = to_cpp<int>(obj->off);
        if (permutation_.count(load_off) == 0)
            return ir_mutator_t::_mutate(obj);

        int new_off = permutation_[load_off];
        ir_assert(new_off % obj->type.size() == 0);

        return load_t::make(obj->type, obj->buf, new_off);
    }

private:
    expr_t grf_buf_base_;
    std::unordered_map<int, int> permutation_;
};

class dpasw_injector_t {
public:
    dpasw_injector_t(const stmt_t &load_mul_stmt, const stmt_t &c_store_stmt,
            alloc_updater_t &alloc_updater, const expr_t &tg_idx0)
        : load_mul_stmt_(load_mul_stmt)
        , c_store_stmt_(c_store_stmt)
        , alloc_updater_(alloc_updater)
        , tg_idx0_(tg_idx0) {}

    const stmt_t &load_mul_stmt() const { return load_mul_stmt_; }

    const stmt_t &c_store_stmt() const { return c_store_stmt_; }

    void inject() {
        expr_t src2_base;
        extract_dpas_calls(src2_base);

        grf_permutator_t grf_perm;

        int dpas_count = int(dpas_infos_.size());
        for (int i = 0; i < dpas_count;) {
            if (i + 1 < dpas_count) {
                auto &a = dpas_infos_[i];
                auto &b = dpas_infos_[i + 1];
                if (try_convert_to_dpasw(a, b, grf_perm)) {
                    i += 2;
                    continue;
                }
            }
            try_convert_to_dpasw(dpas_infos_[i], grf_perm);
            ++i;
        }
        int src2_off = 0;
        for (auto &si : send_infos_) {
            if (!si.reg_buf_base().is_equal(src2_base)) continue;
            auto src2_sub = src2_base[src2_off];
            auto new_call = (si.promote_to_dpasw ? si.new_call : si.call);
            if (!new_call.is_empty()) {
                new_call = substitute(
                        new_call, send_t::arg_reg_buf(new_call), src2_sub, 1);
            }
            load_mul_stmt_ = substitute(load_mul_stmt_, si.call, new_call, 1);
            for (auto &d : si.dpas_consumers) {
                auto &di = find_dpas_info(d);
                ir_assert(si.promote_to_dpasw == di.promote_to_dpasw)
                        << "Both send and dpas must be updated.";
                if (di.update_applied) continue;
                auto new_call = (di.promote_to_dpasw ? di.new_call : di.call);
                new_call = substitute(new_call, dpas_t::arg_src2(new_call),
                        src2_sub[di.src2_relative_off], 1);
                load_mul_stmt_
                        = substitute(load_mul_stmt_, di.call, new_call, 1);
                di.update_applied = true;
            }
            if (!new_call.is_empty()) {
                auto &new_send = new_call.as<func_call_t>().func.as<send_t>();
                src2_off += new_send.register_size();
            }
        }

        // Apply permutation to C store.
        c_store_stmt_ = grf_perm.mutate(c_store_stmt_);

        int new_src2_size = src2_off;
        alloc_updater_.resize(src2_base, new_src2_size);
    }

private:
    struct send_info_t {
        send_info_t() = default;

        send_info_t(const stmt_t &call) : call(call) {}

        const send_t &send() const {
            return call.as<func_call_t>().func.as<send_t>();
        }

        const std::vector<expr_t> &args() const {
            return call.as<func_call_t>().args;
        }

        const expr_t &reg_buf() const { return send_t::arg_reg_buf(call); }

        const expr_t &reg_buf_base() const {
            return reg_buf().as<ptr_t>().base;
        }

        int reg_buf_size() const { return send().register_size(); }

        void set_new_call(const stmt_t &s) {
            if (!promote_to_dpasw) {
                promote_to_dpasw = true;
                new_call = s;
                return;
            }
            ir_assert(new_call.is_equal(s));
        }

        stmt_t call;
        std::vector<stmt_t> dpas_consumers;

        bool promote_to_dpasw = false;
        stmt_t new_call;
    };

    struct dpas_info_t {
        dpas_info_t() = default;

        dpas_info_t(const stmt_t &call) : call(call) {}

        const dpas_t &dpas() const {
            return call.as<func_call_t>().func.as<dpas_t>();
        }

        const std::vector<expr_t> &args() const {
            return call.as<func_call_t>().args;
        }

        const expr_t &src1_buf() const { return dpas_t::arg_src1(call); }

        const expr_t &src2_buf() const { return dpas_t::arg_src2(call); }

        int src2_size() const { return dpas().src2_size(); }

        void set_new_call(const stmt_t &s, int src2_relative_off) {
            if (!promote_to_dpasw) {
                promote_to_dpasw = true;
                this->src2_relative_off = src2_relative_off;
                new_call = s;
                return;
            }
            ir_assert(this->src2_relative_off == src2_relative_off);
            ir_assert(new_call.is_equal(s));
        }

        stmt_t call;
        stmt_t send_producer;

        bool promote_to_dpasw = false;
        bool update_applied = false;
        int src2_relative_off = 0;
        stmt_t new_call;
    };

    send_info_t &find_send_info(const stmt_t &s) {
        for (auto &si : send_infos_)
            if (si.call.is_same(s)) return si;
        ir_error_not_expected();
        return send_infos_.front();
    }

    dpas_info_t &find_dpas_info(const stmt_t &s) {
        for (auto &si : dpas_infos_)
            if (si.call.is_same(s)) return si;
        ir_error_not_expected();
        return dpas_infos_.front();
    }

    void extract_dpas_calls(expr_t &src2_base) {
        object_eq_map_t<expr_t, stmt_t> buf2send;

        auto is_send = [](const stmt_t &s, send_info_t &info) {
            if (!is_func_call<send_t>(s)) return false;
            info = send_info_t(s);
            return true;
        };

        auto is_dpas = [](const stmt_t &s, dpas_info_t &info) {
            if (!is_func_call<dpas_t>(s)) return false;
            info = dpas_info_t(s);
            return true;
        };

        auto set_src2_base = [&](const expr_t &ptr) {
            auto &ptr_base = ptr.as<ptr_t>().base;
            if (src2_base.is_empty()) {
                src2_base = ptr_base;
                return;
            }
            // This may need a fix in the future.
            ir_assert(src2_base.is_same(ptr_base));
        };

        // Iterate through dpas and send calls.
        auto stmt_vec = flatten_statements(load_mul_stmt_);
        for (auto &s : stmt_vec) {
            send_info_t send_info;
            if (is_send(s, send_info)) {
                buf2send[send_info.reg_buf()] = s;
                send_infos_.push_back(send_info);
                continue;
            }
            dpas_info_t dpas_info;
            if (is_dpas(s, dpas_info)) {
                set_src2_base(dpas_info.src2_buf());
                auto &buf = dpas_info.src2_buf();
                auto it = buf2send.find(buf);
                if (it == buf2send.end()) continue;
                auto &send_info = find_send_info(it->second);
                // Ensure read size matches DPAS src2 size.
                // FIXME: This may not be always the case.
                ir_assert(send_info.reg_buf_size() == dpas_info.src2_size());
                dpas_info.send_producer = send_info.call;
                send_info.dpas_consumers.push_back(s);
                dpas_infos_.push_back(dpas_info);
            }
        }
    }

    // Checks for the following pattern:
    //    dpas.sxr(a_dst, a_src0, src1, src2)
    //    dpas.sxr(b_dst, b_src0, src1, src2 + s * r * 4)
    static bool can_convert_to_dpasw(
            const dpas_info_t &a, const dpas_info_t &b) {
        if (!a.dpas().is_equal(&b.dpas())) return false;
        if (!a.src1_buf().is_equal(b.src1_buf())) return false;

        auto src2_off0 = to_cpp<int>(a.src2_buf().as<ptr_t>().off);
        auto src2_off1 = to_cpp<int>(b.src2_buf().as<ptr_t>().off);

        if (src2_off1 - src2_off0 != a.src2_size()) return false;

        return true;
    }

    bool try_convert_to_dpasw(
            dpas_info_t &a, dpas_info_t &b, grf_permutator_t &grf_perm) {
        // Check if DPAS -> DPASW transformation is possible.
        if (!can_convert_to_dpasw(a, b)) return false;

        // Perform the transformation:
        // Before:
        //   send(slm, a_off, src2[0])
        //   send(slm, b_off, src2[s * r * 4])
        //   dpas.sxr(a_dst, a_src0, src1, src2[0])
        //   dpas.sxr(b_dst, b_src0, src1, src2[s * r * 4])
        // After:
        //   send(slm, a_off + (tg_idx0 % 2) * (b_off - a_off), src2)
        //   dpasw.sxr(p_a_dst, p_a_src0, src1, src2[0])
        //   dpasw.sxr(p_b_dst, p_b_src0, src1, src2[s * r * 4 / 2])
        // Where:
        //   p_a_dst[:] = a_dst[0:rcount / 2] + b_dst[0:rcount / 2]
        //   p_b_dst[:] = a_dst[rcount / 2:rcount] + b_dst[rcount / 2:rcount]
        ir_assert(a.dpas().is_equal(&b.dpas()));
        auto _dpasw = dpas_t::make_dpasw(a.dpas());
        auto &dpasw = _dpasw.as<dpas_t>();

        auto a_args = a.args();
        auto b_args = b.args();
        dpas_t::arg_src2(b_args) -= dpasw.src2_size();

        a.set_new_call(dpasw.call(a.args()), 0);
        b.set_new_call(dpasw.call(b_args), dpasw.src2_size());

        // Record permutation for registers to apply it for the destination
        // store later.
        int rcount = a.dpas().rcount;
        for (int j = 0; j < rcount; j++) {
            int k = j % (rcount / 2);
            auto a_old = dpas_t::arg_dst(a_args) + reg_bytes * j;
            auto b_old = dpas_t::arg_dst(b_args) + reg_bytes * j;
            expr_t grf_new;
            if (j < rcount / 2) {
                grf_new = dpas_t::arg_dst(a_args)[reg_bytes * k];
            } else {
                grf_new = dpas_t::arg_dst(b_args)[reg_bytes * k];
            }
            grf_perm.set_permute(a_old, grf_new);
            grf_perm.set_permute(b_old, grf_new + reg_bytes * rcount / 2);
        }

        auto &a_send = find_send_info(a.send_producer);
        auto &b_send = find_send_info(b.send_producer);

        auto &a_mem_off = send_t::arg_mem_off(a_send.call);
        auto &b_mem_off = send_t::arg_mem_off(b_send.call);
        auto ab_addr_diff = simplify(b_mem_off - a_mem_off);
        ir_assert(is_const(ab_addr_diff));

        auto new_send_args = a_send.args();
        send_t::arg_mem_off(new_send_args)
                += (tg_idx0_ % 2) * to_cpp<int64_t>(ab_addr_diff);

        a_send.set_new_call(a_send.send().call(new_send_args));
        b_send.set_new_call(stmt_t());

        for (auto &d : b_send.dpas_consumers) {
            a_send.dpas_consumers.push_back(d);
            b.send_producer = a_send.call;
        }
        b_send.dpas_consumers.clear();
        return true;
    }

    static bool can_convert_to_dpasw(const dpas_info_t &a) {
        return a.dpas().rcount % 2 == 0;
    }

    static func_t create_half_send(const send_t &send) {
        for (auto &_s : send_t::get_all()) {
            auto &s = _s.as<send_t>();
            if (s.access_type != send.access_type) continue;
            if (s.type != send.type) continue;
            if (s.slots != send.slots) continue;
            if (s.alignment != send.alignment) continue;
            if (s.address_model != send.address_model) continue;
            if (s.data_elems * 2 != send.data_elems) continue;
            return _s;
        }
        ir_error_not_expected()
                << "Can't find send reading half of the original send.";
        return func_t();
    }

    bool try_convert_to_dpasw(dpas_info_t &a, grf_permutator_t &grf_perm) {
        if (!can_convert_to_dpasw(a)) return false;

        // Perform the transformation:
        // Before:
        //   send(slm, a_off, src2[0])
        //   dpas.sxr(a_dst, a_src0, src1, src2[0])
        // After:
        //   send(slm, a_off + (tg_idx0 % 2) * (s * r * 4 / 2), src2)
        //   dpasw.sxr(a_dst, a_src0, src1, src2[0])

        auto _dpasw = dpas_t::make_dpasw(a.dpas());
        auto &dpasw = _dpasw.as<dpas_t>();

        a.set_new_call(dpasw.call(a.args()), 0);

        // Real permutation is not required but it needs to be set anyway.
        int rcount = a.dpas().rcount;
        for (int j = 0; j < rcount; j++) {
            auto grf = dpas_t::arg_dst(a.args()) + reg_bytes * j;
            grf_perm.set_permute(grf, grf);
        }

        auto &a_send = find_send_info(a.send_producer);
        auto new_send_args = a_send.args();
        send_t::arg_mem_off(new_send_args)
                += (tg_idx0_ % 2) * to_cpp<int64_t>(a.src2_size() / 2);
        a_send.set_new_call(
                create_half_send(a_send.send()).call(new_send_args));

        return true;
    }

    stmt_t load_mul_stmt_;
    stmt_t c_store_stmt_;
    alloc_updater_t &alloc_updater_;
    expr_t tg_idx0_;

    std::vector<dpas_info_t> dpas_infos_;
    std::vector<send_info_t> send_infos_;
};

// Transforms DPAS to DPASW.
void inject_dpasw(stmt_t &load_mul_stmt, stmt_t &c_store_stmt,
        alloc_updater_t &alloc_updater, const expr_t &tg_idx0) {
    dpasw_injector_t injector(
            load_mul_stmt, c_store_stmt, alloc_updater, tg_idx0);
    injector.inject();

    load_mul_stmt = injector.load_mul_stmt();
    c_store_stmt = injector.c_store_stmt();
}

// Adds {Atomic} modifier to DPAS/DPASW instructions when applicable.
stmt_t inject_atomic(const stmt_t &stmt) {
    stmt_t ret = stmt;
    auto stmt_vec = flatten_statements(stmt);
    for (size_t i = 0; i < stmt_vec.size(); i++) {
        bool ok = true;
        ok &= is_func_call<dpas_t>(stmt_vec[i]);
        ok &= (i + 1 < stmt_vec.size()
                && is_func_call<dpas_t>(stmt_vec[i + 1]));
        if (ok) {
            auto &cur_src1 = dpas_t::arg_src1(stmt_vec[i]);
            auto &next_src1 = dpas_t::arg_src1(stmt_vec[i + 1]);
            // Compare src1, apply {Atomic} if they are equal.
            if (cur_src1.is_equal(next_src1)) {
                auto &s = stmt_vec[i];
                auto atomic_attr = instruction_modifier_attr_t::make(
                        ngen_proxy::InstructionModifier().with_atomic());
                ret = substitute(ret, s, atomic_attr.apply_to(s));
            }
        }
    }
    return ret;
}

// Trace for debugging purposes.
void trace_pass(const char *pass_name, const stmt_t &stmt) {
    ir_trace() << "=== After " << pass_name << std::endl;
    ir_trace() << stmt << std::endl;
}

class external_var_visitor_t : public scope_visitor_t {
public:
    void _visit(const var_t *obj) {
        if (!is_expr_defined(obj)) external_vars.insert(obj);
    }

    object_set_t<expr_t> external_vars;
};

stmt_t inject_external_var_let(const stmt_t &_stmt) {
    auto stmt = _stmt;
    external_var_visitor_t v;
    v.visit(stmt);

    for (auto &var : v.external_vars)
        stmt = let_t::make(var, {}, stmt);

    trace_pass("inject_external_var_let", stmt);
    return stmt;
}

// Merges all SLM buffers into a single one.
stmt_t merge_slm_buffers(const stmt_t &_stmt) {
    stmt_t stmt = _stmt;
    int off = 0;
    alloc_manager_t alloc_mgr(stmt);
    expr_t slm_base = make_buffer("slm");
    for (auto &buf : alloc_mgr.buffers()) {
        if (alloc_mgr.alloc_kind(buf) != alloc_kind_t::slm) continue;
        alloc_updater_t alloc_updater;
        alloc_updater.remove(buf);
        stmt = alloc_updater.update(stmt);
        stmt = substitute(stmt, buf, slm_base + off);
        off += alloc_mgr.alloc_size(buf);
    }
    auto ret = alloc_t::make(slm_base, off, alloc_kind_t::slm, {}, stmt);
    trace_pass("merge_slm_buffers", ret);
    return ret;
}

class buffer_offset_lifter_t : public ir_mutator_t {
public:
    object_t _mutate(const func_call_t *obj) {
        if (!obj->func.is<send_t>()) return ir_mutator_t::_mutate(obj);

        auto &mem_buf = send_t::arg_mem_buf(obj);
        if (!mem_buf.is<ptr_t>()) return ir_mutator_t::_mutate(obj);

        auto &base = mem_buf.as<ptr_t>().base;
        auto &off = mem_buf.as<ptr_t>().off;

        std::vector<expr_t> new_args = obj->args;
        send_t::arg_mem_buf(new_args) = base;
        send_t::arg_mem_off(new_args) += off;
        return obj->func.call(new_args);
    }
};

stmt_t lift_buffer_offsets_in_send(const stmt_t &s) {
    buffer_offset_lifter_t lifter;
    auto ret = lifter.mutate(s);
    trace_pass("lift_buffer_offsets_in_send", ret);
    return ret;
}

class send_injector_t : public ir_mutator_t {
public:
    send_injector_t(ir_context_t &ir_ctx, const constraint_set_t &cset)
        : ir_ctx_(ir_ctx), cset_(cset) {}

    object_t _mutate(const func_call_t *obj) {
        auto *send = obj->func.as_ptr<send_t>();
        if (!send) return ir_mutator_t::_mutate(obj);

        auto &mem_buf = send_t::arg_mem_buf(obj);
        auto &mem_off = send_t::arg_mem_off(obj);
        auto &reg_buf = send_t::arg_reg_buf(obj);
        auto &mask = send_t::arg_mask(obj);

        ir_assert(is_var(mem_buf)) << mem_buf;

        auto header_buf = ir_ctx_.create_tmp_var(type_t::byte_ptr(), "h");
        auto off_store = simplify_store(
                send->create_offset_store(header_buf, mem_buf, mem_off));

        auto new_call = (*send)(mem_buf, header_buf, reg_buf, mask);
        auto body = stmt_seq_t::make(off_store, new_call);

        // Allocate header.
        return alloc_t::make(
                header_buf, send->header_size(), alloc_kind_t::grf, {}, body);
    }

private:
    stmt_t simplify_store(const stmt_t &_store) const {
        auto &store = _store.as<store_t>();

        auto value = store.value;
        value = simplify(value, cset_);

        // Convert to N-ary form and back to expand multiplications. This
        // helps to find more common subexpressions during the pass.
        value = nary_op_canonicalize(value);
        value = nary_op_back_transform(value);

        return store_t::make(store.buf, store.off, value, store.stride);
    }

    ir_context_t &ir_ctx_;
    const constraint_set_t &cset_;
};

stmt_t inject_send(
        const stmt_t &s, ir_context_t &ir_ctx, const constraint_set_t &cset) {
    auto ret = send_injector_t(ir_ctx, cset).mutate(s);
    trace_pass("inject_send", ret);
    return ret;
}

class alloc_lifter_t : public ir_mutator_t {
public:
    object_t _mutate(const alloc_t *obj) override {
        if (!in_compute_loop_) return ir_mutator_t::_mutate(obj);
        // Remove alloc and insert it before the compute loop.
        allocs_.push_back(obj);
        return obj->body;
    }

    object_t _mutate(const stmt_group_t *obj) override {
        bool is_compute_loop = (obj->label == stmt_label_t::compute_loop());
        if (is_compute_loop) in_compute_loop_ = true;
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (is_compute_loop) {
            in_compute_loop_ = false;
            // Outermost loop.
            for (auto it = allocs_.rbegin(); it != allocs_.rend(); ++it) {
                auto &a = it->as<alloc_t>();
                new_obj = alloc_t::make(a.buf, a.size, a.kind, a.attr, new_obj);
            }
            allocs_.resize(0);
        }
        return new_obj;
    }

private:
    bool in_compute_loop_ = false;
    std::vector<stmt_t> allocs_;
};

// Lifts alloc statements out of loops.
stmt_t lift_alloc(const stmt_t &s) {
    auto ret = alloc_lifter_t().mutate(s);
    trace_pass("lift_alloc", ret);
    return ret;
}

// Common subexpression elimination support.

// Represents an expression-candidate to eliminate.
class cse_expr_t {
public:
    cse_expr_t(const expr_t &expr, const ir_path_t &path, int refs = 1,
            const expr_t &cse_var = {})
        : expr(expr), path(path), refs(refs), cse_var(cse_var) {
        ir_trace() << "cse_pass: add expression: " << expr << std::endl;
    }

    void add_usage(const ir_path_t &other_path, bool do_increment = true) {
        if (do_increment) refs++;
        path.merge(other_path);
        ir_trace() << "cse_pass: add usage: " << expr
                   << ", total refs: " << refs << std::endl;
    }

    // Expression to eliminate via let.
    expr_t expr;
    // Path to the innermost IR node where the expression can be defined.
    ir_path_t path;
    // Number of references to the expression.
    int refs;
    // Variable assigned to the expression (if decided to eliminate).
    expr_t cse_var;
};

// Stores information about all expressions subject to CSEing.
class cse_context_t {
public:
    cse_context_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    ir_context_t &ir_ctx() { return ir_ctx_; }

    bool has(const expr_t &e) const { return cse_exprs_.count(e) != 0; }

    cse_expr_t &find_cse_expr(const expr_t &e) {
        ir_assert(has(e)) << e;
        return cse_exprs_.at(e);
    }

    const cse_expr_t &find_cse_expr(const expr_t &e) const {
        ir_assert(has(e)) << e;
        return cse_exprs_.at(e);
    }

    bool has_var(const expr_t &e) const {
        return !find_cse_expr(e).cse_var.is_empty();
    }

    int get_refs(const expr_t &e) const {
        if (!has(e)) return 0;
        return find_cse_expr(e).refs;
    }

    void register_expr(const expr_t &e, const ir_path_t &path) {
        if (e.type().is_bool()) return; // Ignore booleans.
        auto ret = cse_exprs_.insert({e, cse_expr_t(e, path)});
        ir_assert(ret.second) << e;
        MAYBE_UNUSED(ret);
    }

    void register_expr(const cse_expr_t &cse_expr) {
        auto ret = cse_exprs_.insert({cse_expr.expr, cse_expr});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
    }

    expr_t get_or_assign_var(const expr_t &e) {
        auto &cse_expr = find_cse_expr(e);
        if (cse_expr.cse_var.is_empty()) {
            cse_expr.cse_var = ir_ctx_.create_tmp_var(e.type());
            ir_trace() << "cse_pass: assigning var: " << e << " -> "
                       << cse_expr.cse_var << std::endl;
        }
        return cse_expr.cse_var;
    }

    const expr_t &get_var(const expr_t &e) const {
        return find_cse_expr(e).cse_var;
    }

    const ir_path_t &get_path(const expr_t &e) const {
        return find_cse_expr(e).path;
    }

    void add_usage(
            const expr_t &e, const ir_path_t &path, bool do_increment = true) {
        if (e.type().is_bool()) return; // Ignore booleans.
        return find_cse_expr(e).add_usage(path, do_increment);
    }

    void update_expr(const expr_t &old_expr, const expr_t &new_expr) {
        auto it = cse_exprs_.find(old_expr);
        ir_assert(it != cse_exprs_.end()) << old_expr;
        auto &old_cse_expr = it->second;
        auto new_cse_expr = cse_expr_t(new_expr, old_cse_expr.path,
                old_cse_expr.refs, old_cse_expr.cse_var);
        auto ret = cse_exprs_.insert({new_expr, new_cse_expr});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
        cse_exprs_.erase(it);
    }

    template <typename F>
    void for_each(const F &f) const {
        for (auto &kv : cse_exprs_)
            f(kv.first);
    }

private:
    ir_context_t &ir_ctx_;
    object_eq_map_t<expr_t, cse_expr_t> cse_exprs_;
};

// Collects statistics about expressions for common subexpression elimination.
class cse_visitor_t : public ir_visitor_t {
public:
    cse_visitor_t(cse_context_t &ctx) : ctx_(ctx) {}

    void _visit(const binary_op_t *obj) override { visit_expr(obj); }
    void _visit(const shuffle_t *obj) override {
        if (is_const_broadcast(obj)) return;
        visit_expr(obj);
    }
    void _visit(const unary_op_t *obj) override { visit_expr(obj); }

#define HANDLE_IR_OBJECT(type) \
    void _visit(const type *obj) override { visit_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

private:
    template <typename T>
    void visit_expr(const T *obj) {
        // Exclude loads as they may have side effects.
        if (count_objects<load_t>(obj) > 0) {
            ir_visitor_t::_visit(obj);
            return;
        }

        if (propagate_path_) {
            if (ctx_.has(obj))
                ctx_.add_usage(obj, root_path_, /*do_increment=*/false);
            ir_visitor_t::_visit(obj);
            return;
        }
        if (ctx_.has(obj)) {
            ctx_.add_usage(obj, root_path_);
            propagate_path_ = true;
            ir_visitor_t::_visit(obj);
            propagate_path_ = false;
            return;
        }
        ir_visitor_t::_visit(obj);
        ctx_.register_expr(obj, root_path_);
    }

    template <typename T>
    void visit_stmt(const T *obj) {
        if (std::is_same<T, for_t>::value) {
            visit_for((const object_impl_t *)obj);
            return;
        }
        if (std::is_same<T, let_t>::value) {
            visit_let((const object_impl_t *)obj);
            return;
        }
        root_path_.push(obj);
        ir_visitor_t::_visit(obj);
        root_path_.pop();
    }

    void visit_for(const object_impl_t *_obj) {
        auto *obj = (const for_t *)_obj;

        visit(obj->var);
        visit(obj->init);
        visit(obj->bound);
        root_path_.push(obj);
        visit(obj->body);
        root_path_.pop();
    }

    void visit_let(const object_impl_t *_obj) {
        auto *obj = (const let_t *)_obj;

        visit(obj->var);
        visit(obj->value);
        root_path_.push(obj);
        visit(obj->body);
        root_path_.pop();
    }

    cse_context_t &ctx_;
    ir_path_t root_path_;

    bool propagate_path_ = false;
};

// Verifies all IR paths are correct (for debugging purposes).
class cse_verifier_t : public scope_visitor_t {
public:
    cse_verifier_t(cse_context_t &ctx) : ctx_(ctx) {}

    ~cse_verifier_t() { ir_assert(to_check_.empty()); }

    void _visit(const binary_op_t *obj) override { visit_expr(obj); }
    void _visit(const shuffle_t *obj) override { return visit_expr(obj); }
    void _visit(const unary_op_t *obj) override { visit_expr(obj); }

#define HANDLE_IR_OBJECT(type) \
    void _visit(const type *obj) override { visit_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

    void verify(const stmt_t &s) {
        // Phase 0: collect IR paths for expressions.
        phase_ = 0;
        visit(s);

        // Phase 1: verify all expressions are defined at their path.
        phase_ = 1;
        visit(s);
    }

private:
    template <typename T>
    void visit_expr(const T *obj) {
        // Expressions are not used during phase 1.
        if (phase_ == 1) return;
        if (ctx_.has(obj)) {
            auto &path = ctx_.get_path(obj);
            to_check_[path.back()].push_back(obj);
        }
        scope_visitor_t::_visit(obj);
    }

    template <typename T>
    void visit_stmt(const T *obj) {
        scope_visitor_t::_visit(obj);

        // Statements are not used during phase 0.
        if (phase_ == 0) return;

        // Phase 1: check that all attached expressions are defined at this
        // statement.
        auto it = to_check_.find(obj);
        if (it != to_check_.end()) {
            for (auto &e : it->second) {
                ir_assert(is_expr_defined(e));
                MAYBE_UNUSED(e);
            }
            to_check_.erase(it);
        }
    }

    cse_context_t &ctx_;

    int phase_ = 0;
    object_map_t<stmt_t, std::vector<expr_t>> to_check_;
};

// Generates let statements for expressions being eliminated.
class cse_let_generator_t : public ir_visitor_t {
public:
    cse_let_generator_t(const cse_context_t &ctx, const stmt_t &stmt)
        : ctx_(ctx), stmt_(stmt) {}

    void _visit(const binary_op_t *obj) override { visit_expr(obj); }
    void _visit(const shuffle_t *obj) override { visit_expr(obj); }
    void _visit(const unary_op_t *obj) override { visit_expr(obj); }
    void _visit(const var_t *obj) override {
        auto it = all_vars_.find(obj);
        if (it == all_vars_.end()) return;
        if (seen_vars_.count(obj) == 0) generate_for_expr(it->second);
    }

    stmt_t generate() {
        ctx_.for_each([&](const expr_t &e) {
            auto &cse_var = ctx_.get_var(e);
            auto ret = all_vars_.insert({cse_var, e});
            ir_assert(ret.second);
            MAYBE_UNUSED(ret);
        });
        ctx_.for_each([&](const expr_t &e) { generate_for_expr(e); });
        for (auto it = lets_.rbegin(); it != lets_.rend(); ++it) {
            auto &let = it->as<let_t>();
            stmt_ = let_t::make(let.var, let.value, stmt_);
        }
        return stmt_;
    }

private:
    void generate_for_expr(const expr_t &e) {
        auto &cse_var = ctx_.get_var(e);
        if (seen_vars_.count(cse_var) == 1) return;
        visit(e);
    }

    template <typename T>
    void visit_expr(const T *obj) {
        ir_visitor_t::_visit(obj);
        if (ctx_.has(obj) && ctx_.has_var(obj)) {
            auto &var = ctx_.get_var(obj);
            auto ret = seen_vars_.insert(var);
            if (ret.second) lets_.push_back(let_t::make(var, obj));
        }
    }

    const cse_context_t &ctx_;
    stmt_t stmt_;

    object_map_t<expr_t, expr_t> all_vars_; // Var -> expression.
    object_set_t<expr_t> seen_vars_;

    std::vector<stmt_t> lets_;
};

// Eliminiates expressions from the statement.
class cse_mutator_t : public ir_mutator_t {
public:
    cse_mutator_t(cse_context_t &ctx) : ctx_(ctx) {}

    object_t _mutate(const binary_op_t *obj) override {
        return mutate_expr(obj);
    }
    object_t _mutate(const shuffle_t *obj) override { return mutate_expr(obj); }
    object_t _mutate(const unary_op_t *obj) override {
        return mutate_expr(obj);
    }

#define HANDLE_IR_OBJECT(type) \
    object_t _mutate(const type *obj) override { return mutate_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

private:
    template <typename T>
    object_t mutate_expr(const T *obj) {
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (ctx_.has(obj) && !new_obj.is_equal(obj)) {
            ctx_.update_expr(obj, new_obj);
        }
        if (ctx_.get_refs(new_obj) > 1) {
            bool has_var = ctx_.has_var(new_obj);
            auto var = ctx_.get_or_assign_var(new_obj);
            auto &path = ctx_.get_path(new_obj);
            if (!has_var) to_update_[path.back()].push_back(new_obj);
            return std::move(var);
        }
        return new_obj;
    }

    template <typename T>
    object_t mutate_stmt(const T *obj) {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto it = to_update_.find(obj);
        if (it == to_update_.end()) return new_obj;

        cse_context_t local_ctx(ctx_.ir_ctx());
        for (auto &e : it->second) {
            local_ctx.register_expr(ctx_.find_cse_expr(e));
        }
        to_update_.erase(it);

        auto body = get_stmt_body(new_obj);
        cse_let_generator_t g(local_ctx, body);
        body = g.generate();
        new_obj = replace_stmt_body(new_obj, body);
        return new_obj;
    }

    cse_context_t &ctx_;
    object_map_t<stmt_t, std::vector<expr_t>> to_update_;
};

stmt_t eliminate_common_subexprs(const stmt_t &_stmt, ir_context_t &ir_ctx) {
    auto stmt = _stmt;

    cse_context_t ctx(ir_ctx);

    // Collect statistics.
    cse_visitor_t visitor(ctx);
    visitor.visit(stmt);

#ifndef NDEBUG
    // Verify that collected IR paths are correct (cse_expr_t objects are
    // defined at those paths).
    cse_verifier_t verifier(ctx);
    verifier.verify(stmt);
#endif

    // Eliminate subexpressions.
    cse_mutator_t mutator(ctx);
    stmt = mutator.mutate(stmt);

    trace_pass("eliminate_common_subexprs", stmt);
    return stmt;
}

class hoist_exprs_mutator_t : public ir_mutator_t {
public:
    hoist_exprs_mutator_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    ~hoist_exprs_mutator_t() { ir_assert(let_vars_.empty()); }

    object_t _mutate(const func_call_t *obj) override {
        if (!obj->func.is<send_t>()) return ir_mutator_t::_mutate(obj);

        std::vector<expr_t> new_args;
        for (auto &e : obj->args) {
            new_args.push_back(hoist_expr(e));
        }

        if (ir_utils::is_equal(new_args, obj->args)) return obj;

        return func_call_t::make(obj->func, new_args, obj->attr);
    }

    object_t _mutate(const stmt_group_t *obj) override {
        if (obj->body.is<for_t>()) {
            loops_.emplace_back(obj->body.as<for_t>().var);
            auto body = ir_mutator_t::_mutate(obj->body.as_ptr<for_t>());
            if (body.is_same(obj->body)) return obj;
            auto new_obj = stmt_group_t::make(obj->label, body);
            return injects_lets_and_pop_loop(new_obj);
        }
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const store_t *obj) override {
        auto value = hoist_expr(obj->value);
        if (value.is_equal(obj->value)) return obj;
        return store_t::make(obj->buf, obj->off, value, obj->stride);
    }

    object_t _mutate(const for_t *obj) override {
        loops_.emplace_back(obj->var);
        auto new_obj = ir_mutator_t::_mutate(obj);
        return injects_lets_and_pop_loop(new_obj);
    }

    object_t _mutate(const let_t *obj) override {
        bool fully_hoisted = false;
        auto new_value = hoist_expr(obj->value, obj->var, &fully_hoisted);
        if (fully_hoisted) return mutate(obj->body);
        register_let(obj->var, new_value);
        auto new_obj = let_t::make(
                obj->var, new_value, ir_mutator_t::mutate(obj->body));
        unregister_let(obj->var);
        return std::move(new_obj);
    }

private:
    struct loop_info_t {
        loop_info_t(const expr_t &var) : var(var) {}

        expr_t var;
        int var_count = 0;
        std::vector<stmt_t> lets;
    };

    expr_t hoist_expr(const expr_t &expr, const expr_t &expr_var = {},
            bool *fully_hoisted = nullptr) {
        if (expr.is_empty()) return expr;
        if (expr.type().is_ptr()) return expr;
        if (expr.type().is_bool()) return expr;
        if (is_const(expr) || is_shuffle_const(expr) || is_var(expr))
            return expr;

        auto cur_expr = nary_op_canonicalize(expr);

        auto is_nary_add = [](const expr_t &e) {
            auto *nary = e.as_ptr<nary_op_t>();
            return nary && (nary->op_kind == op_kind_t::_add);
        };

        for (size_t i = 0; i < loops_.size(); i++) {
            std::vector<expr_t> invariant_args;
            std::vector<expr_t> other_args;
            std::vector<expr_t> nary_args;
            if (is_nary_add(cur_expr)) {
                nary_args = cvt_expr_to_nary_op_args(cur_expr);
            } else {
                nary_args.push_back(cur_expr);
            }
            for (auto &_a : nary_args) {
                auto a = nary_op_back_transform(_a);
                bool is_inv_arg = true;
                for (size_t j = i; j < loops_.size(); j++) {
                    if (!is_invariant(a, loops_[j].var)) is_inv_arg = false;
                }
                if (is_inv_arg) {
                    invariant_args.push_back(_a);
                } else {
                    other_args.push_back(_a);
                }
            }
            // Nothing to hoist for this loop, continue.
            if (invariant_args.empty()) continue;
            if (invariant_args.size() == 1 && is_var(invariant_args[0]))
                continue;

            // Introduce new variable for the invariant sub-expression.
            auto inv_expr = nary_op_back_transform(
                    make_nary_op(op_kind_t::_add, invariant_args));
            expr_t inv_var;
            if (!expr_var.is_empty() && other_args.empty()) {
                // If nothing to hoist further, reuse the old variable and
                // return.
                inv_var = expr_var;
            } else {
                inv_var = ir_ctx_.create_tmp_var(inv_expr.type());
            }
            auto let = let_t::make(inv_var, inv_expr);
            register_let(inv_var, inv_expr);
            loops_[i].lets.push_back(let);

            if (other_args.empty()) {
                if (fully_hoisted) *fully_hoisted = true;
                return inv_var;
            }

            other_args.push_back(inv_var);
            cur_expr = make_nary_op(op_kind_t::_add, other_args);
        }
        return nary_op_back_transform(cur_expr);
    }

    stmt_t injects_lets_and_pop_loop(const stmt_t &_s) {
        stmt_t s = _s;
        // Inject let statements if any.
        auto &lets = loops_.back().lets;
        for (auto it = lets.rbegin(); it != lets.rend(); ++it) {
            auto &let = it->as<let_t>();
            s = let_t::make(let.var, let.value, s);
            unregister_let(let.var);
        }
        loops_.pop_back();
        return s;
    }

    void register_let(const expr_t &var, const expr_t &value) {
        let_vars_.insert({var, value});
    }

    void unregister_let(const expr_t &var) { let_vars_.erase(var); }

    bool is_invariant(const expr_t &e, const expr_t &var) const {
        if (contains_object(e, var)) return false;

        // Check value if this is a let variable.
        auto it = let_vars_.find(e);
        if (it != let_vars_.end()) return is_invariant(it->second, var);

        if (is_var(e)) return true;

        // Check transitive dependencies.
        auto vars = find_unique_objects<var_t>(e);
        for (auto &v : vars) {
            if (!is_invariant(v, var)) return false;
        }
        return true;
    }

    ir_context_t &ir_ctx_;
    std::vector<loop_info_t> loops_;

    object_map_t<expr_t, expr_t> let_vars_;
};

// Moves invariant expressions out of loops.
stmt_t hoist_exprs(const stmt_t &s, ir_context_t &ir_ctx) {
    auto ret = hoist_exprs_mutator_t(ir_ctx).mutate(s);
    trace_pass("hoist_exprs", ret);
    return ret;
}

class loop_strength_reducer_t : public ir_mutator_t {
public:
    loop_strength_reducer_t() {
        // Create top-level dummy loop.
        loops_.emplace_back();
    }

    ~loop_strength_reducer_t() {
        // Sanity check, all stores must be applied.
        ir_assert(post_inc_stores.empty());
    }

    object_t _mutate(const for_t *obj) override {
        loops_.emplace_back(obj);
        auto new_obj = ir_mutator_t::_mutate(obj);
        return inject_stores_and_pop_loop(new_obj);
    }

    object_t _mutate(const let_t *obj) override {
        int loop_level = int(loops_.size()) - 1;
        auto ret = lets_.insert(
                {obj->var, let_info_t(obj->var, obj->value, loop_level)});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
        auto new_obj = ir_mutator_t::_mutate(obj);
        lets_.erase(obj->var);
        return new_obj;
    }

    object_t _mutate(const stmt_group_t *obj) override {
        if (obj->body.is<for_t>()) {
            loops_.emplace_back(obj->body);
            auto body = ir_mutator_t::_mutate(obj->body.as_ptr<for_t>());
            if (body.is_same(obj->body)) return obj;
            auto new_obj = stmt_group_t::make(obj->label, body);
            return inject_stores_and_pop_loop(new_obj);
        }
        return ir_mutator_t::_mutate(obj);
    }

    // Pattern to handle:
    //     for (...) {
    //         store(buf_ptr, ...) <- Write (producer).
    //         // ...
    //         stmt_t(..., buf_ptr, ...) <- Read (consumer).
    //     }
    object_t _mutate(const store_t *obj) override {
        if (loops_.size() == 1) return ir_mutator_t::_mutate(obj);

        // Try to reduce strength, moving the store up.
        int init_store_level = -1;
        stmt_t init_store_stmt = obj;
        post_inc_store_info_t post_inc_store(obj);
        for (int level = int(loops_.size()) - 1; level >= 1; level--) {
            auto &loop_info = loops_[level];
            int refs = count_object(loop_info.loop, obj->buf);
            // Producer and consumer - must be 2 references.
            if (refs != 2) break;

            // Try to insert the store before level-th loop.
            auto &store = init_store_stmt.as<store_t>();
            auto &store_value = store.value;
            auto &loop_var = loop_info.loop_var();

            auto cur_value = substitute_let(store_value, level);
            auto next_value = substitute(cur_value, loop_var, loop_var + 1);
            auto inc = simplify(next_value - cur_value);

            // Cannot eliminate loop variable, break.
            if (contains_object(inc, loop_var)) break;

            // Success, replace store by post-increment store.
            init_store_level = level;

            auto new_store_value
                    = substitute(cur_value, loop_var, loop_info.loop_init());
            init_store_stmt = store_t::make(store.buf, store.off,
                    simplify(new_store_value), store.stride);

            post_inc_store.update(loop_info, inc);
        }

        // Can't do anything, return as is.
        if (init_store_level == -1) return ir_mutator_t::_mutate(obj);

        // Move this store up, remove from here.
        loops_[init_store_level].init_stores.push_back(init_store_stmt);
        if (!post_inc_store.is_empty()) {
            auto ret = post_inc_stores.insert({obj->buf, post_inc_store});
            ir_assert(ret.second);
            MAYBE_UNUSED(ret);
        }
        return stmt_t();
    }

    object_t _mutate(const func_call_t *obj) override {
        for (auto &kv : post_inc_stores) {
            int refs = count_object(obj, kv.first);
            if (refs == 1) {
                auto ret = stmt_seq_t::make(obj, kv.second.stmt());
                post_inc_stores.erase(kv.first);
                return std::move(ret);
            }
        }
        return ir_mutator_t::_mutate(obj);
    }

private:
    struct loop_info_t {
        loop_info_t(const stmt_t &loop = {}) : loop(loop) {}

        const expr_t &loop_var() const { return loop.as<for_t>().var; }

        const expr_t &loop_init() const { return loop.as<for_t>().init; }

        const expr_t &loop_bound() const { return loop.as<for_t>().bound; }

        expr_t loop_extent() const { return loop_bound() - loop_init(); }

        // Loop being analyzed.
        stmt_t loop;
        // Stores to insert before the loop.
        std::vector<stmt_t> init_stores;

        std::vector<stmt_t> lets;
    };

    struct let_info_t {
        let_info_t(const expr_t &var, const expr_t &value, int loop_level)
            : var(var), value(value), loop_level(loop_level) {}

        expr_t var;
        expr_t value;
        int loop_level;
    };

    struct post_inc_store_info_t {
        post_inc_store_info_t(const store_t *obj)
            : store(obj), inc(0), last_iter_cond(true), compensation(0) {}

        stmt_t stmt() const {
            auto load
                    = load_t::make(store->value.type(), store->buf, store->off);
            return store_t::make(store->buf, store->off, load + inc);
        }

        bool is_empty() const { return is_zero(inc); }

        void update(const loop_info_t &loop, const expr_t &loop_inc) {
            inc = simplify(iif_t::make(
                    last_iter_cond, inc - compensation + loop_inc, inc));
            if (last_iter_cond.is_equal(expr_t(true))) {
                last_iter_cond = (loop.loop_var() == loop.loop_bound() - 1);
            } else {
                last_iter_cond = last_iter_cond
                        & (loop.loop_var() == loop.loop_bound() - 1);
            }
            compensation = simplify(loop.loop_extent() * loop_inc);
        }

        const store_t *store;
        expr_t inc;

        expr_t last_iter_cond;
        expr_t compensation;
    };

    // Recursively substitutes all variable from let statements located under
    // the given loop level.
    expr_t substitute_let(const expr_t &_e, int loop_level) const {
        auto e = _e;
        for (;;) {
            bool found = false;
            auto vars = find_unique_objects<var_t>(e);
            for (auto &v : vars) {
                auto it = lets_.find(v);
                if (it == lets_.end()) continue;
                auto &let_info = it->second;
                // Do not substitute top-level let variables.
                if (let_info.loop_level < loop_level) continue;
                found = true;
                e = substitute(e, v, let_info.value);
            }
            if (!found) break;
        }
        return e;
    }

    // Injects initial store statements if any.
    object_t inject_stores_and_pop_loop(const stmt_t &_s) {
        stmt_t s = _s;
        auto &stores = loops_.back().init_stores;
        for (auto it = stores.rbegin(); it != stores.rend(); ++it) {
            s = stmt_seq_t::make(*it, s);
        }
        loops_.pop_back();
        // The top-level dummy loop shouldn't be removed.
        ir_assert(loops_.size() >= 1);
        return std::move(s);
    }

    // Loops, ordered from outermost to innermost. The first loop is dummy, to
    // represent let statements in the top-level scope.
    std::vector<loop_info_t> loops_;

    // Buffers whose references are to be updated.
    object_map_t<expr_t, post_inc_store_info_t> post_inc_stores;

    // Let statements available at the current IR node.
    object_map_t<expr_t, let_info_t> lets_;
};

// Detects and converts expensive expression operations inside a loop to less
// expensive operations. Example:
// Before:
//     for (int j = 0; j < N; j++) {
//         int off = off_i + j * K;
//         a[off] = j;
//     }
// After:
//     int off = off_i;
//     for (int j = 0; j < N; j++) {
//         a[off] = j;
//         off += K;
//     }
stmt_t loop_strength_reduce(const stmt_t &s) {
    auto ret = loop_strength_reducer_t().mutate(s);
    trace_pass("loop_strength_reduce", ret);
    return ret;
}

class let_optimizer_t : public ir_mutator_t {
public:
    // Also track alloc_t and for_t to validate all variable usages.
    object_t _mutate(const alloc_t *obj) override {
        return mutate_scope(obj, obj->buf);
    }
    object_t _mutate(const for_t *obj) override {
        level_++;
        auto new_obj = mutate_scope(obj, obj->var);
        level_--;
        return new_obj;
    }
    object_t _mutate(const let_t *obj) override {
        return mutate_scope(obj, obj->var);
    }

    object_t _mutate(const var_t *obj) override {
        ir_assert(refs_.count(obj) == 1)
                << "Variable is not defined: " << expr_t(obj);
        refs_[obj].update(increment_, level_);
        return ir_mutator_t::_mutate(obj);
    }

private:
    struct ref_info_t {
        ref_info_t(int level = 0)
            : refs(0), min_level(level), max_level(level) {}

        void update(int increment, int level) {
            refs += increment;
            max_level = std::max(max_level, level);
        }

        bool is_same_level() const { return min_level == max_level; }

        int refs;
        int min_level;
        int max_level;
    };

    template <typename T>
    object_t mutate_scope(const T *obj, const expr_t &var) {
        auto ret = refs_.insert({var, ref_info_t(level_)});
        ir_assert(ret.second) << stmt_t(obj);
        MAYBE_UNUSED(ret);

        auto new_obj = ir_mutator_t::_mutate(obj);
        auto &ref_info = refs_[var];

        if (std::is_same<T, let_t>())
            new_obj = mutate_let(new_obj.template as<let_t>(), ref_info);

        refs_.erase(var);
        return new_obj;
    }

    object_t mutate_let(const let_t &obj, const ref_info_t &ref_info) {
        ir_assert(ref_info.refs >= 1);
        if (ref_info.refs == 1) {
            // Variable is not used.
            remove_refs(obj);
            return obj.body;
        }
        // Check following conditions to substitute let value:
        // - 2 references: one from producer, one from consumer - means single usage
        // - Consumer and producer are on the same level (same loop)
        // - Variable is not external
        if (ref_info.refs == 2 && ref_info.is_same_level()
                && !obj.value.is_empty()) {
            return substitute(obj.body, obj.var, obj.value);
        }
        return &obj;
    }

    void remove_refs(const let_t &obj) {
        increment_ = -1;
        mutate(obj.value);
        increment_ = 1;
    }

    int increment_ = 1;
    int level_ = 0;
    object_map_t<expr_t, ref_info_t> refs_;
};

stmt_t optimize_let(const stmt_t &s) {
    auto ret = let_optimizer_t().mutate(s);
    trace_pass("optimize_let", ret);
    return ret;
}

class slm_buffering_loop_updater_t : public ir_mutator_t {
public:
    object_t _mutate(const let_t *obj) override {
        if (level_ == 0) {
            // Skip top-level let statements.
            return ir_mutator_t::_mutate(obj);
        }
        lets_.push_back(obj);
        auto new_body = mutate(obj->body);
        if (!lets_.back()) {
            // Let was moved to the innermost loop.
            lets_.pop_back();
            return new_body;
        }
        lets_.pop_back();
        if (new_body.is_same(obj->body)) return obj;
        return let_t::make(obj->var, obj->value, new_body);
    }

    object_t _mutate(const for_t *obj) override {
        level_++;
        found_loop_ = false;
        auto new_obj = ir_mutator_t::_mutate(obj);
        level_--;
        if (!found_loop_) {
            // Innermost loop, inject let statements.
            auto body = get_stmt_body(new_obj);
            for (auto it = lets_.rbegin(); it != lets_.rend(); ++it) {
                body = let_t::make((*it)->var, (*it)->value, body);
                *it = nullptr;
            }
            new_obj = replace_stmt_body(new_obj, body);
        }
        found_loop_ = true;
        return new_obj;
    }

private:
    bool found_loop_ = false;
    int level_ = 0;
    std::vector<const let_t *> lets_;
};

// Eliminates let statements from the outer loops to be able to unroll loop
// nest for SLM buffering. Example:
// Before:
//     for (int i = 0; i < I; i++) {
//         int tmp = TMP;
//         for (int j = 0; j < J; j++) {
//            ...
//         }
//     }
// After:
//     for (int i = 0; i < I; i++) {
//         for (int j = 0; j < J; j++) {
//             int tmp = TMP;
//             ...
//         }
//     }
stmt_t update_loops_for_slm_buffering(const stmt_t &s) {
    auto ret = slm_buffering_loop_updater_t().mutate(s);
    trace_pass("update_loops_for_slm_buffering", ret);
    return ret;
}

// Helper structure for for_t.
struct loop_info_t {
    loop_info_t(const stmt_t &s) {
        ir_assert(s.is<for_t>()) << s;
        auto &loop = s.as<for_t>();
        stmt = s;
        var = loop.var;
        init = to_cpp<int>(loop.init);
        bound = to_cpp<int>(loop.bound);
    }

    stmt_t stmt;
    expr_t var;
    int init;
    int bound;
};

// Iterates through multiple nested loops with fixed bounds. Used to unroll
// such nested loops.
class multi_loop_iterator_t {
public:
    // Ordered from innermost to outermost.
    multi_loop_iterator_t(const std::vector<loop_info_t> &loops)
        : loops_(loops) {
        for (auto &l : loops)
            var_values_.push_back(l.init);
    }

    int var_value(const expr_t &var) const {
        for (size_t i = 0; i < loops_.size(); i++) {
            if (loops_[i].var.is_same(var)) return var_values_[i];
        }
        ir_error_not_expected();
        return 0;
    }

    void advance() {
        if (loops_.empty()) return;
        for (size_t i = 0; i < loops_.size(); i++) {
            auto &l = loops_[i];
            if (++var_values_[i] < l.bound) break;
            var_values_[i] = l.init;
        }
        ir_assert(var_values_.back() < loops_.back().bound);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "multi_loop_iterator_t(";
        for (size_t i = 0; i < loops_.size(); i++) {
            oss << (i != 0 ? ", " : "");
            oss << loops_[i].var << " = " << var_values_[i];
        }
        oss << ")";
        return oss.str();
    }

private:
    std::vector<loop_info_t> loops_;
    std::vector<int> var_values_;
};

// Extracts different parts of the compute iteration and verifies the loop nest
// is properly formed and can be further injected with SLM buffering.
class compute_step_visitor_t : public ir_visitor_t {
public:
    stmt_t find_stmt_group(const stmt_label_t &label) const {
        auto groups = find_stmt_groups(label);
        ir_assert(groups.size() == 1);
        return groups[0];
    }

    std::vector<stmt_t> find_stmt_groups(const stmt_label_t &label) const {
        std::vector<stmt_t> ret;
        for (auto &_g : stmt_groups_) {
            auto &g = _g.as<stmt_group_t>();
            if (g.label == label) ret.push_back(_g);
        }
        ir_assert(!ret.empty());
        return ret;
    }

    const std::vector<stmt_t> &inner_let_stmts() const {
        return inner_let_stmts_;
    }

#define HANDLE_IR_OBJECT(type) \
    void _visit(const type *obj) override { visit_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

    template <typename T>
    void visit_stmt(const T *obj) {
        auto obj_type_id = T::_type_id();
        bool is_for = (obj_type_id == for_t::_type_id());
        bool is_stmt_group = (obj_type_id == stmt_group_t::_type_id());
        bool is_let = (obj_type_id == let_t::_type_id());
        bool is_stmt_seq = (obj_type_id == stmt_seq_t::_type_id());

        // Loop may contain:
        // - Another loop
        // - Container statement (stmt_seq_t or stmt_group_t)
        // - Let statement (in the innermost loop only)
        // - Barrier
        if (loop_level_ > 0) {
            bool ok = false;
            if (is_for || is_let || is_stmt_group || is_stmt_seq) {
                ok = true;
            } else if (obj_type_id == func_call_t::_type_id()) {
                auto &call = obj->template as<func_call_t>();
                ok = call.func.is_equal(funcs::barrier_func());
            }

            if (!ok) {
                ir_error_not_expected()
                        << "Found unexpected statement inside loop.\n"
                        << stmt_t(obj);
            }
        }

        bool is_compute_loop = false;
        if (is_stmt_group) {
            auto label = obj->template as<stmt_group_t>().label;
            stmt_groups_.push_back(obj);
            if (utils::one_of(label, stmt_label_t::g2s_load(),
                        stmt_label_t::g2s_store(), stmt_label_t::g2r_load(),
                        stmt_label_t::s2r_load(), stmt_label_t::mul())) {
                // Leaf labels, do not visit them.
                return;
            }
            if (label == stmt_label_t::compute_loop()) {
                is_compute_loop = true;
                in_compute_loop_ = true;
            }
        }

        if (is_for) loop_level_++;
        found_loop_ = false;
        ir_visitor_t::_visit(obj);
        if (in_compute_loop_ && is_let) {
            if (found_loop_)
                ir_error_not_expected()
                        << "Let is allowed in the innermost loop only.";

            inner_let_stmts_.push_back(replace_stmt_body(obj, stmt_t()));
        }
        if (is_for) {
            loop_level_--;
            found_loop_ = true;
        }

        if (is_compute_loop) in_compute_loop_ = false;
    }

private:
    bool found_loop_ = false;
    bool in_compute_loop_ = false;
    int loop_level_ = 0;

    std::vector<stmt_t> stmt_groups_;
    std::vector<stmt_t> inner_let_stmts_;
};

// Provides access to different parts of the inner compute iteration.
class compute_step_t {
public:
    compute_step_t(const stmt_t &parent) {
        compute_step_visitor_t v;
        v.visit(parent);

        compute_loop_ = v.find_stmt_group(stmt_label_t::compute_loop());
        g2s_load_ = v.find_stmt_group(stmt_label_t::g2s_load());
        g2s_store_ = v.find_stmt_group(stmt_label_t::g2s_store());
        g2r_load_ = v.find_stmt_groups(stmt_label_t::g2r_load());
        s2r_load_ = v.find_stmt_groups(stmt_label_t::s2r_load());
        mul_ = v.find_stmt_groups(stmt_label_t::mul());
        c_zero_out_ = v.find_stmt_group(stmt_label_t::c_zero_out());
        inner_let_stmts_ = v.inner_let_stmts();

        ir_assert(g2r_load_.size() == mul_.size());
        ir_assert(s2r_load_.size() == mul_.size());

        for (auto &_let : inner_let_stmts_) {
            auto &var = _let.as<let_t>().var;
            bool in_g2s_load = count_object(g2s_load_, var);
            bool in_g2s_store = count_object(g2s_store_, var);
            bool in_g2r_load = count_object(g2r_load_, var);
            bool in_s2r_load = count_object(s2r_load_, var);
            bool in_mul = count_object(mul_, var);
            ir_assert(!in_mul && !in_s2r_load && !in_g2s_store)
                    << "Unexpected let usage.";
            if (in_g2s_load) g2s_lets_.insert(_let);
            if (in_g2r_load) mul_lets_.insert(_let);
            MAYBE_UNUSED(in_g2s_store);
            MAYBE_UNUSED(in_s2r_load);
            MAYBE_UNUSED(in_mul);
        }
    }

    // See ir_core.hpp for the description.
    const stmt_t &compute_loop() const { return compute_loop_; }
    const stmt_t &g2s_load() const { return g2s_load_; }
    const stmt_t &g2s_store() const { return g2s_store_; }
    const std::vector<stmt_t> &g2r_load() const { return g2r_load_; }
    const std::vector<stmt_t> &s2r_load() const { return s2r_load_; }
    const std::vector<stmt_t> &mul() const { return mul_; }
    const stmt_t &c_zero_out() const { return c_zero_out_; }
    const std::vector<stmt_t> &inner_let_stmts() const {
        return inner_let_stmts_;
    }

    bool is_g2s_let(const stmt_t &s) const { return g2s_lets_.count(s) > 0; }

    bool is_mul_let(const stmt_t &s) const { return mul_lets_.count(s) > 0; }

private:
    stmt_t compute_loop_;
    stmt_t g2s_load_;
    stmt_t g2s_store_;
    std::vector<stmt_t> g2r_load_;
    std::vector<stmt_t> s2r_load_;
    std::vector<stmt_t> mul_;
    stmt_t c_zero_out_;

    std::vector<stmt_t> inner_let_stmts_;

    // Let statements can be used from two different contexts:
    // - In GMEM to SLM loads (with SLM buffering)
    // - In GMEM to GRF loads (no SLM buffering)
    // Due to loop unrolling such lets depend on different values for loop
    // variables hence we need to differentiate between them.
    object_set_t<stmt_t> g2s_lets_;
    object_set_t<stmt_t> mul_lets_;
};

// Helper class to work with loop nest of the compute loop.
class compute_loop_nest_t {
public:
    compute_loop_nest_t(const stmt_t &root) : root_(root) {
        for (auto &l : find_objects<for_t>(root)) {
            loops_.emplace_back(l);
        }

        if (loops_.empty()) {
            outer_loop_size_ = 1;
            return;
        }

        // Outer loop may not be used for unrolling hence loop iterations must
        // not use its index. If this doesn't hold, assume a dummy outer loop
        // with single iteration.
        auto &outer_info = loops_.back();
        auto &outer_loop = outer_info.stmt.as<for_t>();
        if (count_object(outer_loop.body, outer_loop.var) == 0) {
            outer_loop_size_ = outer_info.bound - outer_info.init;
        } else {
            outer_loop_size_ = 1;
        }
    }

    const std::vector<loop_info_t> &loops() const { return loops_; }

    // Number of iterations of all loops.
    int size() const {
        int ret = 1;
        for (auto &l : loops_)
            ret *= (l.bound - l.init);
        return ret;
    }

    // Number of iterations in the outermost loop (see comments in ctor).
    int outer_loop_size() const { return outer_loop_size_; }

    template <typename F>
    void for_each_loop_var(const F &f) const {
        for (auto &l : loops_)
            f(l.var);
    }

    // Number of iterations of all loops except the outermost.
    int inner_loops_size() const { return size() / outer_loop_size(); }

private:
    stmt_t root_;
    std::vector<loop_info_t> loops_;
    int outer_loop_size_;
};

struct compute_params_t {
    compute_params_t(int slm_bufs, int gmem_bufs, int slm_buf_size,
            int inner_loops_iters)
        : slm_bufs(slm_bufs), gmem_bufs(gmem_bufs), slm_buf_size(slm_buf_size) {
        if (slm_bufs >= 1) {
            unroll = math::lcm(gmem_bufs * slm_bufs, inner_loops_iters);
        } else {
            unroll = inner_loops_iters;
        }
    }

    int unroll;
    int slm_bufs;
    int gmem_bufs;
    int slm_buf_size;
};

// Helper class to implement SLM buffering.
class compute_iterator_t {
public:
    compute_iterator_t(const compute_params_t &params,
            const compute_loop_nest_t &loop_nest)
        : params(params)
        , g2s_loop_it(loop_nest.loops())
        , mul_loop_it(loop_nest.loops()) {
        ir_assert(utils::one_of(gmem_bufs(), 1, 2));
        ir_assert(utils::one_of(slm_bufs(), 0, 1, 2, 3));

        int compute_iters = loop_nest.size();
        iters = compute_iters;
        ir_assert(iters >= 1) << "Empty loop is not expected.";
        ir_assert(gmem_bufs() >= 1);

        iters += std::max(0, slm_bufs() - 1) + (gmem_bufs() - 1);
        ramp_up_iters = std::max(1, slm_bufs() + (gmem_bufs() - 1));
        ramp_down_iters
                = std::min(std::max(0, slm_bufs() - 1) + (gmem_bufs() - 1),
                        iters - ramp_up_iters);
        body_iters = iters - ramp_up_iters - ramp_down_iters;
        body_iters = utils::rnd_dn(body_iters, params.unroll);
        ramp_down_iters = iters - ramp_up_iters - body_iters;

        ir_assert(ramp_up_iters + body_iters + ramp_down_iters == iters);

        iter = 0;
        linear_id = 0;
        riter = iters - 1;
    }

    int unroll() const { return params.unroll; }

    int slm_bufs() const { return params.slm_bufs; }

    int gmem_bufs() const { return params.gmem_bufs; }

    compute_iterator_t &operator++() {
        if (do_g2s_load()) g2s_loop_it.advance();
        if (do_mul()) mul_loop_it.advance();
        ++iter;
        ++linear_id;
        --riter;
        return *this;
    }

    void advance(int n) {
        ir_assert(n % params.unroll == 0);
        ir_assert(iter + n <= iters);

        iter += n;
        riter -= n;
    }

    bool do_mul() const {
        return iter >= std::max(0, slm_bufs() - 1) + (gmem_bufs() - 1);
    }

    bool is_first_mul() const {
        return iter == std::max(0, slm_bufs() - 1) + (gmem_bufs() - 1);
    }
    bool is_last_mul() const { return riter == 0; }

    bool do_g2s_load() const {
        if (slm_bufs() == 0) return false;
        return riter >= (slm_bufs() - 1) + (gmem_bufs() - 1);
    }
    bool do_s2r_load() const {
        if (slm_bufs() == 0) return false;
        return iter >= (gmem_bufs() - 1) && riter >= (slm_bufs() - 1);
    }

    int gmem_write_buf_index() const {
        ir_assert(do_g2s_load());
        return iter % gmem_bufs();
    }

    int gmem_read_buf_index() const {
        ir_assert(do_s2r_load());
        return (iter - (gmem_bufs() - 1)) % gmem_bufs();
    }

    int slm_read_offset_update() const {
        ir_assert(slm_bufs() >= 1);
        ir_assert(do_mul());

        int slm_iter = iter - (gmem_bufs() - 1) - (slm_bufs() - 1);
        int cur_slm_idx = slm_iter % slm_bufs();
        int next_slm_idx = (slm_iter + 1) % slm_bufs();
        int ret = next_slm_idx * params.slm_buf_size
                - cur_slm_idx * params.slm_buf_size;
        return ret;
    }

    int slm_write_offset_update() const {
        ir_assert(slm_bufs() >= 1);
        ir_assert(do_s2r_load());

        int slm_iter = iter - (gmem_bufs() - 1);
        int cur_slm_idx = slm_iter % slm_bufs();
        int next_slm_idx = (slm_iter + 1) % slm_bufs();
        int ret = next_slm_idx * params.slm_buf_size
                - cur_slm_idx * params.slm_buf_size;
        return ret;
    }

    compute_params_t params;
    multi_loop_iterator_t g2s_loop_it;
    multi_loop_iterator_t mul_loop_it;

    // ramp_up_iters + body_iters + ramp_down_iters == iters
    int iters;
    int ramp_up_iters;
    int body_iters;
    int ramp_down_iters;

    // Invariant: iter + riter = iters - 1
    int iter;
    int riter;

    int linear_id;
};

// Basic LRU SBID allocator, tries to use the same SBIDs for the same GRF
// buffers.
class sbid_manager_t {
public:
    sbid_manager_t() : tuple_func_(builtin_t::make("tuple")) {}

    ngen_proxy::SBID get_sbid(const expr_t &buf, int index = 0) {
        auto key = tuple_func_.call({buf, expr_t(index)});

        int free_idx = -1;
        for (int i = 0; i < sbid_count; i++) {
            auto &e = entries_[i];
            if (key.is_equal(e.key)) {
                e.time = cur_time_++;
                return ngen_proxy::SBID(i);
            }
            if (free_idx == -1 && e.key.is_empty()) free_idx = i;
        }

        // Not found but there is a free SBID.
        if (free_idx != -1) {
            entries_[free_idx] = {key, cur_time_++};
            return ngen_proxy::SBID(free_idx);
        }

        // Find the oldest SBID and use it.
        int old_idx = 0;
        int old_time = entries_[0].time;
        for (int i = 1; i < sbid_count; i++) {
            if (entries_[i].time < old_time) {
                old_idx = i;
                old_time = entries_[i].time;
            }
        }

        entries_[old_idx] = entry_t({key, cur_time_++});
        return ngen_proxy::SBID(old_idx);
    }

private:
    struct entry_t {
        stmt_t key;
        int time;
    };

    static const int sbid_count = 16;
    std::array<entry_t, sbid_count> entries_;

    func_t tuple_func_;
    int cur_time_ = 0;
};

class slm_buffering_injector_t {
public:
    slm_buffering_injector_t(
            const stmt_t &root, const conv_config_t &cfg, ir_context_t &ir_ctx)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , root_(root)
        , alloc_mgr_(root_)
        , step_(root)
        , loop_nest_(root) {
        for (auto &b : find_send_buffers(step_.g2s_load(), /*is_mem=*/false)) {
            g2s_reg_bufs_.emplace_back(b, alloc_mgr_.alloc_size(b));
        }
    }

    stmt_t inject() {
        int slm_size = alloc_mgr_.total_size(alloc_kind_t::slm);
        int inner_iters = loop_nest_.inner_loops_size();
        compute_params_t params(
                cfg_.slm_bufs, cfg_.gmem_bufs, slm_size, inner_iters);
        compute_iterator_t it(params, loop_nest_);
        stmt_t body;

        sbid_manager_t sbid_mgr;

        // Ramp-up.
        for (int i = 0; i < it.ramp_up_iters; i++) {
            body = stmt_seq_t::make(body, create_iteration(it, sbid_mgr));
            ++it;
        }

        // Body.
        if (it.body_iters > 0) {
            stmt_t loop_body;
            for (int i = 0; i < it.unroll(); i++) {
                loop_body = loop_body.append(create_iteration(it, sbid_mgr));
                ++it;
            }
            int extent = it.body_iters / it.unroll();
            if (extent == 1) {
                body = body.append(loop_body);
            } else {
                ir_assert(extent > 0);
                auto for_var = ir_ctx_.create_tmp_var(type_t::s32(), "i");
                body = body.append(for_t::make(for_var, 0, extent, loop_body));
            }
            it.advance(it.body_iters - it.unroll());
        }

        // Ramp-down.
        for (int i = 0; i < it.ramp_down_iters; i++) {
            body = body.append(create_iteration(it, sbid_mgr));
            ++it;
        }

        alloc_updater_t alloc_updater;

        // Update buffer sizes.
        for (auto &b : g2s_reg_bufs_) {
            alloc_updater.resize(
                    b.buf, alloc_mgr_.alloc_size(b.buf) * cfg_.gmem_bufs);
        }

        auto slm_buffers = alloc_mgr_.find_buffers(alloc_kind_t::slm);
        if (!slm_buffers.empty()) {
            ir_assert(slm_buffers.size() == 1);

            auto &slm_buf = slm_buffers[0];
            alloc_updater.resize(slm_buf, slm_size * cfg_.slm_bufs);
        }

        auto ret = substitute(root_, step_.compute_loop(), body, 1);
        ret = alloc_updater.update(ret);

        // Remove zero-out statement for C (handled by sub_dpas_src0_with_zero).
        ret = substitute(ret, step_.c_zero_out(), stmt_t(), 1);

        return ret;
    }

private:
    struct buffer_info_t {
        buffer_info_t(const expr_t &buf, int size) : buf(buf), size(size) {}

        expr_t buf;
        int size;
    };

    stmt_t create_iteration(
            const compute_iterator_t &it, sbid_manager_t &sbid_mgr) const {
        auto g2s_load = step_.g2s_load();
        auto g2s_store = step_.g2s_store();
        auto g2r_load = step_.g2r_load();
        auto s2r_load = step_.s2r_load();
        auto mul = step_.mul();
        auto lets = step_.inner_let_stmts();

        loop_nest_.for_each_loop_var([&](const expr_t &v) {
            g2s_load = const_fold(substitute(
                    g2s_load, v, expr_t(it.g2s_loop_it.var_value(v))));
            g2s_store = const_fold(substitute(
                    g2s_store, v, expr_t(it.g2s_loop_it.var_value(v))));
            for (auto &s : g2r_load) {
                s = const_fold(
                        substitute(s, v, expr_t(it.mul_loop_it.var_value(v))));
            }
            for (auto &s : s2r_load) {
                s = const_fold(
                        substitute(s, v, expr_t(it.g2s_loop_it.var_value(v))));
            }
            for (int i = 0; i < int(lets.size()); i++) {
                auto &let = lets[i];
                auto &orig_let = step_.inner_let_stmts()[i];
                expr_t var_value;
                if (step_.is_g2s_let(orig_let)) {
                    var_value = it.g2s_loop_it.var_value(v);
                } else if (step_.is_mul_let(orig_let)) {
                    var_value = it.mul_loop_it.var_value(v);
                } else {
                    ir_error_not_expected() << orig_let;
                }
                let = const_fold(substitute(let, v, var_value));
            }
        });

        g2s_load = sub_gmem_bufs(g2s_load, it, /*is_read=*/false);
        g2s_store = sub_gmem_bufs(g2s_store, it, /*is_read=*/true);

        g2s_store = sub_slm_bufs(g2s_store, it, /*is_read=*/false);
        for (auto &s : s2r_load) {
            s = sub_slm_bufs(s, it, /*is_read=*/true);
        }

        if (it.is_first_mul()) {
            for (auto &m : mul) {
                m = sub_dpas_src0_with_zero(m);
            }
        }

        stmt_t iter_stmt;
        if (it.slm_bufs() == 3 && it.do_mul()) {
            iter_stmt = iter_stmt.append(funcs::barrier_wait());
        }

        if (it.do_g2s_load()) iter_stmt = iter_stmt.append(g2s_load);

        if (it.slm_bufs() == 3 && it.iter == it.gmem_bufs()) {
            iter_stmt = iter_stmt.append(funcs::slm_fence());
            iter_stmt = iter_stmt.append(funcs::signal());
        }

        if (it.do_s2r_load() && it.slm_bufs() == 1) {
            iter_stmt = iter_stmt.append(funcs::barrier());
            iter_stmt = iter_stmt.append(g2s_store);
            iter_stmt = iter_stmt.append(funcs::barrier());
        }

        if (it.do_mul()) {
            for (size_t i = 0; i < mul.size(); i++) {
                iter_stmt = iter_stmt.append(g2r_load[i]);
                iter_stmt = iter_stmt.append(s2r_load[i]);
                iter_stmt = iter_stmt.append(mul[i]);
            }
            if (it.slm_bufs() == 3 && !it.is_last_mul()) {
                iter_stmt = iter_stmt.append(funcs::signal());
            }
        }
        if (it.do_s2r_load() && it.slm_bufs() >= 2) {
            iter_stmt = iter_stmt.append(g2s_store);
            if (it.slm_bufs() == 2) {
                iter_stmt = iter_stmt.append(funcs::barrier());
            }
        }

        if (cfg_.assign_sbids)
            iter_stmt = assign_sbids(iter_stmt, it, sbid_mgr);

        iter_stmt = inject_local_let(iter_stmt, lets, it.linear_id);

        return iter_stmt;
    }

    stmt_t sub_gmem_bufs(const stmt_t &stmt, const compute_iterator_t &it,
            bool is_read) const {
        if (it.slm_bufs() == 0) return stmt;
        if (is_read && !it.do_s2r_load()) return stmt;
        if (!is_read && !it.do_g2s_load()) return stmt;

        int buf_idx = (is_read ? it.gmem_read_buf_index()
                               : it.gmem_write_buf_index());
        if (buf_idx == 0) return stmt;

        auto ret = stmt;
        for (auto &b : g2s_reg_bufs_) {
            ret = substitute(ret, b.buf, b.buf[buf_idx * b.size]);
        }
        return ret;
    }

    stmt_t sub_slm_bufs(const stmt_t &stmt, const compute_iterator_t &it,
            bool is_read) const {
        if (it.slm_bufs() <= 1) return stmt;
        if (is_read && !it.do_mul()) return stmt;
        if (!is_read && !it.do_s2r_load()) return stmt;

        int upd = (is_read ? it.slm_read_offset_update()
                           : it.slm_write_offset_update());

        auto stmt_vec = flatten_statements(stmt);

        stmt_t ret = stmt;
        for (auto &s : stmt_vec) {
            auto *call = s.as_ptr<func_call_t>();
            if (!call) continue;

            auto &send = call->func.as<send_t>();
            auto &args = call->args;
            auto &mem_buf = send_t::arg_mem_buf(args);
            auto &header_buf = send_t::arg_mem_off(args);

            // This is not send to SLM, skip.
            if (send.address_model != ngen_proxy::AddressModel::ModelSLM)
                continue;

            // May have signed offset.
            auto store_obj = send.create_offset_store(
                    header_buf, mem_buf, upd, /*is_signed_offset=*/true);
            auto &store = store_obj.as<store_t>();
            expr_t old_value
                    = load_t::make(send.address_type(), store.buf, store.off);
            auto post_inc_store = store_t::make(
                    store.buf, store.off, old_value + store.value);
            ret = substitute(ret, s, stmt_seq_t::make(s, post_inc_store), 1);
        }

        return ret;
    }

    stmt_t sub_dpas_src0_with_zero(const stmt_t &stmt) const {
        auto stmt_vec = flatten_statements(stmt);

        stmt_t ret = stmt;
        for (auto &s : stmt_vec) {
            if (!is_func_call<dpas_t>(s)) continue;

            auto &call = s.as<func_call_t>();

            auto &dst = dpas_t::arg_dst(s);
            auto src0 = expr_t(0); // Will be translated to null register.
            auto &src1 = dpas_t::arg_src1(s);
            auto &src2 = dpas_t::arg_src2(s);

            auto new_call = func_call_t::make(
                    call.func, {dst, src0, src1, src2}, call.attr);
            ret = substitute(ret, s, new_call, 1);
        }
        return ret;
    }

    // Returns memory buffers if is_mem is true and register buffers otherwise.
    static object_set_t<expr_t> find_send_buffers(
            const stmt_t &s, bool is_mem) {
        object_set_t<expr_t> ret;
        auto calls = find_objects<func_call_t>(s);
        for (auto &_c : calls) {
            auto &c = _c.as<func_call_t>();
            if (!c.func.is<send_t>()) continue;
            auto &buf = (is_mem ? send_t::arg_mem_buf(_c)
                                : send_t::arg_reg_buf(_c));
            ret.insert(buf.as<ptr_t>().base);
        }
        return ret;
    }

    stmt_t assign_sbids(const stmt_t &stmt, const compute_iterator_t &it,
            sbid_manager_t &sbid_mgr) const {
        auto is_slm_send = [](const stmt_t &s) {
            if (!is_func_call<send_t>(s)) return false;
            auto &send = s.as<func_call_t>().func.as<send_t>();
            return send.address_model == ngen_proxy::AddressModel::ModelSLM;
        };

        auto is_read_send = [](const stmt_t &s) {
            if (!is_func_call<send_t>(s)) return false;
            auto &send = s.as<func_call_t>().func.as<send_t>();
            return send.access_type == ngen_proxy::Access::Read;
        };

        auto g2r_loads = find_stmt_groups(stmt, stmt_label_t::g2r_load());
        auto is_g2r_load = [&](const stmt_t &s) {
            for (auto &l : g2r_loads) {
                if (count_object(l, s) > 0) return true;
            }
            return false;
        };

        auto update_call_with_sbid
                = [](const stmt_t &s, const ngen_proxy::SBID &sbid) {
                      return instruction_modifier_attr_t::make(
                              ngen_proxy::InstructionModifier().with_sbid(sbid))
                              .apply_to(s);
                  };
        auto stmt_vec = flatten_statements(stmt);
        int g2s_store_idx = it.do_s2r_load() ? it.gmem_read_buf_index() : 0;
        int g2s_load_idx = it.do_g2s_load() ? it.gmem_write_buf_index() : 0;
        stmt_t ret = stmt;
        for (auto &_s : stmt_vec) {
            if (!_s.is<func_call_t>()) continue;
            auto s = _s;
            if (is_slm_send(s) && is_read_send(s)) {
                auto sbid = sbid_mgr.get_sbid(send_t::arg_reg_buf(s));
                s = update_call_with_sbid(s, sbid);
            } else if (is_slm_send(s) && !is_read_send(s)) {
                auto sbid = sbid_mgr.get_sbid(
                        send_t::arg_reg_buf(s), g2s_store_idx);
                s = update_call_with_sbid(s, sbid);
            } else if (is_read_send(s)) {
                int idx = is_g2r_load(s) ? 0 : g2s_load_idx;
                auto sbid = sbid_mgr.get_sbid(send_t::arg_reg_buf(s), idx);
                s = update_call_with_sbid(s, sbid);
            } else if (is_func_call<dpas_t>(s)) {
                auto &attr = s.as<func_call_t>().attr;
                auto *mod_attr = attr.as_ptr<instruction_modifier_attr_t>();
                if (!mod_attr || !mod_attr->mod.is_atomic) {
                    // Last dpas in Atomic chain.
                    auto sbid = sbid_mgr.get_sbid(dpas_t::arg_src1(s));
                    s = update_call_with_sbid(s, sbid);
                }
            } else if (s.is<func_call_t>()) {
                auto &c = s.as<func_call_t>();
                if (c.func.is_equal(funcs::signal_func())
                        || c.func.is_equal(funcs::slm_fence_func())
                        || c.func.is_equal(funcs::barrier_func())) {
                    // Use 0 as the key for signals and SLM fences.
                    auto sbid = sbid_mgr.get_sbid(expr_t(0));
                    s = update_call_with_sbid(s, sbid);
                }
            } else {
                ir_error_not_expected() << s;
            }
            ret = substitute(ret, _s, s, 1);
        }
        return ret;
    }

    static stmt_t inject_local_let(const stmt_t &_s,
            const std::vector<stmt_t> &enclosed_lets, int id) {
        auto s = _s;

        // Inject let statements from the innermost loop.
        for (auto &_let : enclosed_lets) {
            auto &let = _let.as<let_t>();
            s = let_t::make(let.var, let.value, s);
        }

        // Substitute variables to avoid clashing.
        auto lets = find_objects<let_t>(s);
        for (auto &_let : lets) {
            auto &let = _let.as<let_t>();
            auto &var = let.var.as<var_t>();
            auto local_var = var_t::make(
                    var.type, var.name + "_" + std::to_string(id));
            s = substitute(s, let.var, local_var);
        }
        return s;
    }

    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;

    stmt_t root_;
    alloc_manager_t alloc_mgr_;
    compute_step_t step_;
    compute_loop_nest_t loop_nest_;

    std::vector<buffer_info_t> g2s_reg_bufs_;
};

// Injects SLM buffering based on the config.
stmt_t inject_slm_buffering(
        const stmt_t &s, const conv_config_t &cfg, ir_context_t &ir_ctx) {
    auto ret = slm_buffering_injector_t(s, cfg, ir_ctx).inject();
    trace_pass("inject_slm_buffering", ret);
    return ret;
}

        }
    }
}

// Implements reorder between dense GRF buffers. Conversion between data types
// is supported.
class linear_reorder_builder_t {
public:
    linear_reorder_builder_t(ir_context_t &ir_ctx, int elems,
            const type_t &src_type, const expr_t &src_buf,
            const type_t &dst_type, const expr_t &dst_buf)
        : ir_ctx_(ir_ctx)
        , elems_(elems)
        , src_type_(src_type)
        , src_buf_(src_buf)
        , dst_type_(dst_type)
        , dst_buf_(dst_buf) {
        ir_assert(src_type_.is_scalar()) << "Vector types are unsupported.";
        ir_assert(dst_type_.is_scalar()) << "Vector types are unsupported.";
        build();
    }

    const stmt_t &stmt() const { return stmt_; }

private:
    void build() {
        if (src_type_ == dst_type_) {
            int step = (elems_ < 16 ? 8 : 16);
            ir_assert(elems_ % step == 0) << "Unsupported elems: " << elems_;

            auto vec_type = src_type_.with_elems(step);
            for (int i = 0; i < elems_; i += step) {
                int off = (i / step) * vec_type.size();
                auto load = load_t::make(vec_type, src_buf_, off);
                auto store = store_t::make(dst_buf_, off, load);
                stmt_ = stmt_.append(store);
            }
            return;
        }
        // f32/s32 -> s8/u8:
        // - Use saturation
        // - s8/u8 must be DW-strided: use temporary
        if (src_type_.size() == 4 && dst_type_.size() == 1) {
            int step = (elems_ < 16 ? 8 : 16);
            auto tmp_buf = ir_ctx_.create_tmp_var(type_t::byte_ptr());
            int tmp_size = dst_type_.size() * step;
            for (int i = 0; i < elems_; i += step) {
                int cur_elems = std::min(step, elems_ - i);
                ir_assert(math::is_pow2(cur_elems));
                auto src_vec_type = src_type_.with_elems(cur_elems);
                auto dst_vec_type = dst_type_.with_elems(cur_elems);
                int off = i * src_type_.size();
                auto load_src = load_t::make(src_vec_type, src_buf_, off);
                auto cvt_strided
                        = cast(load_src, dst_vec_type, /*saturate=*/true);
                auto store_strided = store_t::make(tmp_buf, 0, cvt_strided, 4);
                auto load_strided = load_t::make(dst_vec_type, tmp_buf, 0, 4);
                auto store_dense = store_t::make(dst_buf_, off, load_strided);
                auto step_stmt = stmt_seq_t::make(store_strided, store_dense);
                stmt_ = stmt_.append(alloc_t::make(
                        tmp_buf, tmp_size, alloc_kind_t::grf, {}, step_stmt));
            }
            return;
        }
        // f32 -> bf16 or f32 -> f16:
        // - SIMD16 does not support mixed mode move.
        if (src_type_ == type_t::f32()
                && utils::one_of(dst_type_, type_t::bf16(), type_t::f16())) {
            int step = 8;
            for (int i = 0; i < elems_; i += step) {
                int cur_elems = std::min(step, elems_ - i);
                ir_assert(math::is_pow2(cur_elems));
                ir_assert(utils::one_of(i % 16, 0, 8))
                        << "Not supported in HW.";
                auto src_vec_type = src_type_.with_elems(cur_elems);
                auto dst_vec_type = dst_type_.with_elems(cur_elems);
                int src_off = i * src_type_.size();
                int dst_off = i * dst_type_.size();
                auto load = load_t::make(src_vec_type, src_buf_, src_off);
                auto store = store_t::make(
                        dst_buf_, dst_off, cast(load, dst_vec_type));
                stmt_ = stmt_.append(store);
            }
            return;
        }

        ir_error_not_expected();
    }

    ir_context_t &ir_ctx_;

    int elems_;
    type_t src_type_;
    expr_t src_buf_;
    type_t dst_type_;
    expr_t dst_buf_;

    stmt_t stmt_;
};

// Implements reorder between GRF buffers in given layouts. Conversion between
// data types is supported.
class reorder_builder_t {
public:
    reorder_builder_t(ir_context_t &ir_ctx, const view_t &src,
            const view_t &dst, const expr_t &src_buf, const expr_t &dst_buf)
        : ir_ctx_(ir_ctx)
        , src_(src.create_vlayout())
        , dst_(dst.create_vlayout())
        , src_buf_(src_buf)
        , dst_buf_(dst_buf) {
        ir_assert(src_.ndims() == dst_.ndims()) << "Layouts are incompatible.";
        ir_assert(src_.elems() == dst_.elems()) << "Layouts are incompatible.";
        build();
    }

    const stmt_t &stmt() const { return stmt_; }

private:
    void build() {
        // 1. Split layouts to have aligned blocks.
        auto a = src_;
        auto b = dst_;
        layout_t::align_layouts(a, b);

        // 2. Find the biggest innermost dense tensor (tile).
        auto a_blocks = a.blocks();
        auto b_blocks = b.blocks();

        std::vector<dim_t> tile_dims(a.ndims(), 1);
        for (size_t i = 0; i < std::min(a_blocks.size(), b_blocks.size());
                i++) {
            auto &ab = a_blocks[i];
            auto &bb = b_blocks[i];
            if (ab.dim_idx != bb.dim_idx || ab.block != bb.block
                    || ab.stride != bb.stride) {
                break;
            }
            tile_dims[ab.dim_idx] *= ab.block;
        }

        // 3. Generate copy/convert statements for every tile.
        tensor_t tile(tile_dims);
        dim_t tile_elems = tile.elems();
        src_.for_each_tile(tile, [&](const std::vector<dim_t> &start) {
            dim_t src_off = src_(start);
            dim_t dst_off = dst_(start);
            ir_assert(src_off % tile_elems == 0);
            ir_assert(dst_off % tile_elems == 0);
            auto src_sub_buf = src_buf_[src_off * src_.type().size()];
            auto dst_sub_buf = dst_buf_[dst_off * dst_.type().size()];
            linear_reorder_builder_t b(ir_ctx_, tile_elems, src_.type(),
                    src_sub_buf, dst_.type(), dst_sub_buf);
            stmt_ = stmt_.append(b.stmt());
        });
    }

    ir_context_t &ir_ctx_;

    layout_t src_;
    layout_t dst_;

    expr_t src_buf_;
    expr_t dst_buf_;

    stmt_t stmt_;
};

// Generates loads or stores to move data between memory (global or SLM) and
// GRF. Memory layout is a parameter. GRF layout is deduced automatically,
// according to the decomposition into messages.
class access_builder_t {
public:
    access_builder_t() = default;

    access_builder_t(const constraint_set_t &cset, const view_t &mem_view,
            const expr_t &mem_buf, const expr_t &reg_buf, bool is_slm,
            bool is_load)
        : cset_(&cset)
        , mem_view_(mem_view)
        , mem_buf_(mem_buf)
        , reg_buf_(reg_buf)
        , is_slm_(is_slm)
        , is_load_(is_load) {
        build();
    }

    bool is_slm() const { return is_slm_; }

    const view_t &reg_view() const { return reg_view_; }

    int reg_buf_size() const { return reg_buf_size_; }

    const stmt_t &stmt() const { return stmt_; }

    std::string str() const {
        std::ostringstream oss;
        oss << "Memory view:          " << mem_view_ << std::endl;
        oss << "Register view:        " << reg_view_ << std::endl;
        oss << "Register buffer:      " << reg_buf_ << std::endl;
        oss << "Register buffer size: " << reg_buf_size_ << " ("
            << reg_buf_size_ / reg_bytes << " regs)" << std::endl;
        oss << "Statement:            " << std::endl << stmt_;
        return oss.str();
    }

private:
    void build() {
        // List of send functions that can be used for the access.
        auto send_list = send_t::get_all([&](const func_t &_s) {
            auto &s = _s.as<send_t>();
            bool is_read = (s.access_type == ngen_proxy::Access::Read);
            bool ok = true;
            ok &= is_slm_
                    ? (s.address_model == ngen_proxy::AddressModel::ModelSLM)
                    : (s.address_model == ngen_proxy::AddressModel::ModelA64);
            ok &= (is_load_ == is_read);
            // XXX: Generate only block messages for now.
            ok &= (s.type == message_type_t::block);
            return ok;
        });

        // Find the first send candidate matching the layout.
        func_t _send;
        tensor_t send_tensor;
        for (auto &_s : send_list) {
            auto &s = _s.as<send_t>();
            int block_bytes = s.block_size();
            if (block_bytes % mem_view_.type().size() != 0) continue;

            int elems_per_block = block_bytes / mem_view_.type().size();

            // Check if the view can be decomposed for this send.
            auto tensor
                    = mem_view_.split_into_dense_tile(elems_per_block, s.slots);
            if (tensor.is_empty()) continue;

            // Check if this send supports the required mask.
            if (!has_compatible_mask(
                        *cset_, s, mem_view_.create_sub_view(tensor)))
                continue;

            // TODO: Check alignment requirements.

            // Success, send is found, stop iterating.
            _send = _s;
            send_tensor = tensor;
            break;
        }
        ir_assert(!_send.is_empty()) << "Can't decompose view into messages.";

        auto &send = _send.as<send_t>();
        reg_view_ = create_register_view_for_message(
                send, mem_view_, reg_buf_size_);

        int64_t reg_buf_off = 0;
        mem_view_.for_each_tile(
                send_tensor, [&](const std::vector<dim_t> &start) {
                    auto tile = tensor_t(send_tensor.dims(), start);
                    auto sub_view = mem_view_.create_sub_view(tile);
                    auto reg_sub_buf = reg_buf_[reg_buf_off];
                    stmt_ = stmt_seq_t::make(stmt_,
                            create_send_stmt(*cset_, send, mem_buf_,
                                    reg_sub_buf, sub_view));
                    reg_buf_off += send.register_size();
                });
    }

    const constraint_set_t *cset_;

    view_t mem_view_;
    expr_t mem_buf_;
    view_t reg_view_;
    expr_t reg_buf_;
    int reg_buf_size_;
    bool is_slm_;
    bool is_load_;
    stmt_t stmt_;
};

class read_builder_t : public access_builder_t {
public:
    read_builder_t() = default;

    read_builder_t(const constraint_set_t &cset, const view_t &view,
            const expr_t &mem_buf, const expr_t &reg_buf, bool is_slm)
        : access_builder_t(
                cset, view, mem_buf, reg_buf, is_slm, /*is_load=*/true) {}
};

class write_builder_t : public access_builder_t {
public:
    write_builder_t() = default;

    write_builder_t(const constraint_set_t &cset, const view_t &view,
            const expr_t &mem_buf, const expr_t &reg_buf, bool is_slm)
        : access_builder_t(
                cset, view, mem_buf, reg_buf, is_slm, /*is_load=*/false) {}
};

// Performs the following steps after the computation:
// - Conversion
// - GRF reorder to match the memory layout
// - Store to the destination
class epilogue_builder_t {
public:
    epilogue_builder_t(ir_context_t &ir_ctx, const constraint_set_t &cset,
            const view_t &mem_view, const view_t &reg_view,
            const expr_t &mem_buf, const expr_t &reg_buf)
        : ir_ctx_(ir_ctx)
        , cset_(cset)
        , mem_view_(mem_view)
        , reg_view_(reg_view)
        , mem_buf_(mem_buf)
        , reg_buf_(reg_buf)
        , tmp_reg_buf_(make_buffer("c_tmp")) {
        build();
    }

    const expr_t &tmp_reg_buf() const { return tmp_reg_buf_; }

    int tmp_buf_size() const { return tmp_buf_size_; }

    const stmt_t &stmt() const { return stmt_; }

private:
    void build() {
        auto &mem_type = mem_view_.type();
        int tmp_buf_elems = tmp_buf_size_ / mem_view_.type().size();
        auto base_tile = mem_view_.split_into_max_innermost_tile(tmp_buf_elems);
        mem_view_.for_each_tile(
                base_tile, [&](const std::vector<dim_t> &start) {
                    auto tile = tensor_t(base_tile.dims(), start);
                    auto mem_sub_view = mem_view_.create_sub_view(tile);
                    auto reg_sub_view = reg_view_.create_sub_view(tile);

                    write_builder_t r2g(cset_, mem_sub_view, mem_buf_,
                            tmp_reg_buf_,
                            /*is_slm=*/false);
                    auto chunk_stmt = r2g.stmt();

                    if (!reg_sub_view.has_same_vlayout(r2g.reg_view())) {
                        // Generate reorder between layouts.
                        reorder_builder_t reorder(ir_ctx_, reg_sub_view,
                                r2g.reg_view(), reg_buf_, tmp_reg_buf_);
                        chunk_stmt
                                = stmt_seq_t::make(reorder.stmt(), chunk_stmt);
                    } else {
                        // Layouts are the same, no need to reorder. Use the register
                        // buffer directly.
                        auto reg_sub_off = reg_view_(start) * mem_type.size();
                        auto reg_sub_buf = reg_buf_[reg_sub_off];
                        chunk_stmt = substitute(
                                chunk_stmt, tmp_reg_buf_, reg_sub_buf);
                    }
                    stmt_ = stmt_.append(chunk_stmt);
                });
    }

    ir_context_t &ir_ctx_;
    const constraint_set_t &cset_;

    view_t mem_view_;
    view_t reg_view_;

    expr_t mem_buf_;
    expr_t reg_buf_;
    expr_t tmp_reg_buf_;

    // TODO: Add logic to determine blocking bytes, hard-coding for now.
    int tmp_buf_size_ = 128;

    stmt_t stmt_;
};

class multiply_builder_t {
public:
    multiply_builder_t() = default;

    multiply_builder_t(const layout_t &a_layout, const layout_t &b_layout,
            const expr_t &a_buf, const expr_t &b_buf, const expr_t &c_buf)
        : a_layout_(a_layout)
        , b_layout_(b_layout)
        , a_buf_(a_buf)
        , b_buf_(b_buf)
        , c_buf_(c_buf) {

        if (try_build_dpas()) return;

        ir_error_not_expected();
    }

    const stmt_t &stmt() const { return stmt_; }

    const layout_t &a_layout() const { return a_layout_; }
    const layout_t &b_layout() const { return b_layout_; }
    const layout_t &c_layout() const { return c_layout_; }

    ngen_proxy::Bundle a_grf_bundle() {
        if (!do_transpose_) return ngen_proxy::Bundle();
        return ngen_proxy::Bundle(1, ngen_proxy::Bundle::any);
    }

    ngen_proxy::Bundle b_grf_bundle() {
        if (do_transpose_) return ngen_proxy::Bundle();
        return ngen_proxy::Bundle(1, ngen_proxy::Bundle::any);
    }

    ngen_proxy::Bundle c_grf_bundle() {
        return ngen_proxy::Bundle(0, ngen_proxy::Bundle::any);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "A layout:  " << a_layout_ << std::endl;
        oss << "B layout:  " << b_layout_ << std::endl;
        oss << "C layout:  " << c_layout_ << std::endl;
        oss << "Statement: " << std::endl << stmt_;
        return oss.str();
    }

private:
    bool try_build_dpas() {
        multiply_desc_t desc(a_layout_, b_layout_, true);
        if (!dpas_t::matches_types(desc.a_type(), desc.b_type(), desc.c_type()))
            return false;

        auto _dpas = dpas_t::make(/*is_dpasw=*/false, /*sdepth=*/8,
                /*rcount=*/8, desc.c_type(), desc.a_type(), desc.b_type());
        if (_dpas.as<dpas_t>().matches(desc)) {
            build_dpas(_dpas.as<dpas_t>(), desc);
            return true;
        }

        // Try to transpose and flip.
        _dpas = dpas_t::make(/*is_dpasw=*/false, /*sdepth=*/8, /*rcount=*/8,
                desc.c_type(), desc.b_type(), desc.a_type());

        desc = multiply_desc_t(
                b_layout_.transpose(), a_layout_.transpose(), true);
        if (_dpas.as<dpas_t>().matches(desc)) {
            a_layout_ = desc.a_layout();
            b_layout_ = desc.b_layout();
            std::swap(a_buf_, b_buf_);
            do_transpose_ = true;
            build_dpas(_dpas.as<dpas_t>(), desc);
            return true;
        }
        return false;
    }

    void build_dpas(const dpas_t &dpas, const multiply_desc_t &desc) {
        int m_blk = 8;
        int n_blk = dpas.rcount;

        c_layout_ = compute_dpas_c_layout(m_blk, n_blk, dpas.c_layout(), desc);

        for (int i_m = 0; i_m < desc.m(); i_m += m_blk) {
            for (int i_n = 0; i_n < desc.n(); i_n += n_blk) {
                std::vector<int> a_args = {i_m, 0};
                std::vector<int> b_args = {0, i_n};
                std::vector<int> c_args = {i_m, i_n};
                auto a = a_buf_[desc.a_layout()(a_args) * desc.a_type().size()];
                auto b = b_buf_[desc.b_layout()(b_args) * desc.b_type().size()];
                auto c = c_buf_[c_layout_(c_args) * desc.c_type().size()];
                stmt_ = stmt_seq_t::make(stmt_, dpas(c, c, a, b));
            }
        }

        // Transpose C layout back if needed.
        if (do_transpose_) c_layout_ = c_layout_.transpose();
    }

    static layout_t compute_dpas_c_layout(int m_blk, int n_blk,
            const layout_t &blk_layout, const multiply_desc_t &desc) {
        auto c_layout = blk_layout;
        c_layout = c_layout.add_outer_block(1, desc.n() / n_blk);
        c_layout = c_layout.add_outer_block(0, desc.m() / m_blk);
        return c_layout;
    }

    bool do_transpose_ = false;

    layout_t a_layout_;
    layout_t b_layout_;
    layout_t c_layout_;

    expr_t a_buf_;
    expr_t b_buf_;
    expr_t c_buf_;

    stmt_t stmt_;
};

class compute_builder_t {
public:
    compute_builder_t(const conv_config_t &cfg, ir_context_t &ir_ctx,
            const constraint_set_t &cset)
        : cfg_(cfg), ir_ctx_(ir_ctx), cset_(cset) {}

    const std::vector<stmt_t> &allocs() const { return allocs_; }

    const stmt_t &c_zero_out_stmt() const { return c_zero_out_stmt_; }

    stmt_t iter_stmt() const {
        stmt_t stmt;
        stmt = stmt.append(
                stmt_group_t::make(stmt_label_t::g2s_load(), g2s_load_stmt_));
        stmt = stmt.append(funcs::barrier());
        stmt = stmt.append(
                stmt_group_t::make(stmt_label_t::g2s_store(), g2s_store_stmt_));
        stmt = stmt.append(funcs::barrier());
        stmt = stmt.append(load_mul_stmt_);
        return stmt;
    }

    const stmt_t &c_store_stmt() const { return c_store_stmt_; }

    void set_thread_group(const grid_info_t &tg_grid) { tg_grid_ = tg_grid; }

    // Setters for original AP/BP/CP buffers (P - problem notation).
    void set_ap_buf(const expr_t &buf) { ap_buf_ = buf; }
    void set_bp_buf(const expr_t &buf) { bp_buf_ = buf; }
    void set_cp_buf(const expr_t &buf) { cp_buf_ = buf; }

    // Setters for thread group views (problem notation).
    void set_ap_tg_view(const view_t &v) { ap_tg_view_ = v; }
    void set_bp_tg_view(const view_t &v) { bp_tg_view_ = v; }
    void set_cp_tg_view(const view_t &v) { cp_tg_view_ = v; }

    // Setters for thread group blocks (GEMM notation).
    void set_m_tg_blk(int b) { m_tg_blk_ = b; }
    void set_n_tg_blk(int b) { n_tg_blk_ = b; }
    void set_k_tg_blk(int b) { k_tg_blk_ = b; }

    void build() {
        // Initialize SLM buffers.
        expr_t a_slm_buf = make_buffer("a_slm");
        expr_t b_slm_buf = make_buffer("b_slm");

        // Initialize GRF buffers.
        expr_t a_g2s_reg_buf = make_buffer("a_g2s");
        expr_t b_g2s_reg_buf = make_buffer("b_g2s");

        expr_t a_buf = make_buffer("a");
        expr_t b_buf = make_buffer("b");
        expr_t c_buf = make_buffer("c");

        view_t ap_slm_view;
        view_t bp_slm_view;

        prepare_gmem_to_slm("A", cfg_.use_a_slm, ap_tg_view_, ap_buf_,
                a_g2s_reg_buf, a_slm_buf, ap_slm_view);
        prepare_gmem_to_slm("B", cfg_.use_b_slm, bp_tg_view_, bp_buf_,
                b_g2s_reg_buf, b_slm_buf, bp_slm_view);

        // Split A across tg1, B across tg0.
        int m_thr_blk = m_tg_blk_ / tg_grid_.dim(1);
        int n_thr_blk = n_tg_blk_ / tg_grid_.dim(0);

        // Views to multiply by a thread group.
        auto &ap_x_view = (cfg_.use_a_slm ? ap_slm_view : ap_tg_view_);
        auto &bp_x_view = (cfg_.use_b_slm ? bp_slm_view : bp_tg_view_);

        // Views to multiply by a thread.
        view_t ap_thr_view
                = ap_x_view.split(mnk_tensor_t({mnk_kind_t::m, mnk_kind_t::k},
                                          {m_thr_blk, k_tg_blk_}),
                        tg_grid_.sub_grid({1}));
        view_t bp_thr_view
                = bp_x_view.split(mnk_tensor_t({mnk_kind_t::k, mnk_kind_t::n},
                                          {k_tg_blk_, n_thr_blk}),
                        tg_grid_.sub_grid({0}));

        auto a_idx = ir_ctx_.create_tmp_var(type_t::s32(), "a_idx");
        auto b_idx = ir_ctx_.create_tmp_var(type_t::s32(), "b_idx");

        std::vector<block_t> a_i_outer_blocks;
        std::vector<block_t> b_j_outer_blocks;
        auto _a_i_view = ap_thr_view.split(
                mnk_tensor_t({mnk_kind_t::m, mnk_kind_t::k},
                        {m_thr_blk / cfg_.a_sub_tiles, k_tg_blk_}),
                grid_info_t({cfg_.a_sub_tiles}, {a_idx}), &a_i_outer_blocks);
        auto _b_j_view = bp_thr_view.split(
                mnk_tensor_t({mnk_kind_t::k, mnk_kind_t::n},
                        {k_tg_blk_, n_thr_blk / cfg_.b_sub_tiles}),
                grid_info_t({cfg_.b_sub_tiles}, {b_idx}), &b_j_outer_blocks);

        mnk_mapper_t mnk_mapper;
        bool is_first = true;
        int c_buf_off = 0;
        layout_t c_sub_tile_layout;
        alloc_attr_t c_attr;
        ir_assert(cfg_.a_sub_tiles == 1 || cfg_.b_sub_tiles == 1)
                << "At most one tensor can be tiled.";
        for (int i = 0; i < cfg_.a_sub_tiles; i++) {
            // Load A_i.
            auto a_i_view = _a_i_view.substitute(a_idx, i);
            read_builder_t a_read(cset_, a_i_view,
                    cfg_.use_a_slm ? a_slm_buf : ap_buf_, a_buf,
                    /*is_slm=*/cfg_.use_a_slm);
            ir_trace() << "A GMEM/SLM to GRF load #" << i << ":\n"
                       << a_read.str() << std::endl;
            for (int j = 0; j < cfg_.b_sub_tiles; j++) {
                // Load B_i.
                auto b_j_view = _b_j_view.substitute(b_idx, j);
                read_builder_t b_read(cset_, b_j_view,
                        cfg_.use_b_slm ? b_slm_buf : bp_buf_, b_buf,
                        /*is_slm=*/cfg_.use_b_slm);
                ir_trace() << "B GMEM/SLM to GRF load #" << j << ":\n"
                           << b_read.str() << std::endl;

                stmt_t ab_g2r_load;
                stmt_t ab_s2r_load;
                if (j == 0) {
                    if (a_read.is_slm()) {
                        ab_s2r_load = ab_s2r_load.append(a_read.stmt());
                    } else {
                        ab_g2r_load = ab_g2r_load.append(a_read.stmt());
                    }
                }
                if (b_read.is_slm()) {
                    ab_s2r_load = ab_s2r_load.append(b_read.stmt());
                } else {
                    ab_g2r_load = ab_g2r_load.append(b_read.stmt());
                }
                load_mul_stmt_ = load_mul_stmt_.append(stmt_group_t::make(
                        stmt_label_t::g2r_load(i + j), ab_g2r_load));
                load_mul_stmt_ = load_mul_stmt_.append(stmt_group_t::make(
                        stmt_label_t::s2r_load(i + j), ab_s2r_load));

                auto a_layout = mnk_mapper.map_to_mnk(
                        a_read.reg_view(), {mnk_kind_t::m, mnk_kind_t::k});
                auto b_layout = mnk_mapper.map_to_mnk(
                        b_read.reg_view(), {mnk_kind_t::k, mnk_kind_t::n});

                // Multiply C_i_j += A_i x B_j in GEMM notation.
                multiply_builder_t mul_builder(
                        a_layout, b_layout, a_buf, b_buf, c_buf[c_buf_off]);
                ir_trace() << "Multiply (" << i << ", " << j << "):\n"
                           << mul_builder.str() << std::endl;
                load_mul_stmt_ = load_mul_stmt_.append(stmt_group_t::make(
                        stmt_label_t::mul(i + j), mul_builder.stmt()));
                if (is_first) {
                    is_first = false;
                    mnk_mapper.push_view(a_read.reg_view(), cp_tg_view_);
                    mnk_mapper.push_view(b_read.reg_view(), cp_tg_view_);
                    c_sub_tile_layout = mul_builder.c_layout();
                    c_attr = grf_alloc_attr_t::make(mul_builder.c_grf_bundle());
                    auto a_attr = grf_alloc_attr_t::make(
                            mul_builder.a_grf_bundle());
                    auto b_attr = grf_alloc_attr_t::make(
                            mul_builder.b_grf_bundle());
                    register_buffer(a_buf, a_read.reg_buf_size(),
                            alloc_kind_t::grf, a_attr);
                    register_buffer(b_buf, b_read.reg_buf_size(),
                            alloc_kind_t::grf, b_attr);
                } else {
                    ir_assert(mul_builder.c_layout() == c_sub_tile_layout)
                            << "Sub-tile layouts must be equal.";
                }
                c_buf_off += c_sub_tile_layout.size();
            }
        }

        mnk_mapper.push_blocks(a_i_outer_blocks, ap_thr_view, cp_tg_view_);
        mnk_mapper.push_blocks(b_j_outer_blocks, bp_thr_view, cp_tg_view_);

        // C layout in GEMM notation.
        auto c_layout = c_sub_tile_layout;

        // Add outer blocks coming from A/B sub-tiles.
        c_layout = c_layout.add_outer_block(0, cfg_.a_sub_tiles);
        c_layout = c_layout.add_outer_block(1, cfg_.b_sub_tiles);

        view_t cp_thr_mem_view
                = create_cp_thr_mem_view(ap_thr_view, bp_thr_view);

        // C layout in the problem notation.
        auto cp_thr_reg_layout
                = mnk_mapper.map_from_mnk(c_layout, cp_thr_mem_view.nvdims());
        cp_thr_reg_layout = cp_thr_reg_layout.normalize();

        auto cp_thr_reg_view = view_t(cp_thr_mem_view, cp_thr_reg_layout);

        epilogue_builder_t c_m2g(ir_ctx_, cset_, cp_thr_mem_view,
                cp_thr_reg_view, cp_buf_, c_buf);
        ir_trace() << "C GRF to GMEM store:\n" << c_m2g.stmt() << std::endl;

        register_buffer(
                c_m2g.tmp_reg_buf(), c_m2g.tmp_buf_size(), alloc_kind_t::grf);
        register_buffer(c_buf, c_layout.size(), alloc_kind_t::grf, c_attr);

        int step_bytes = 2 * reg_bytes;
        for (int i = 0; i < c_layout.size(); i += step_bytes) {
            c_zero_out_stmt_ = c_zero_out_stmt_.append(store_t::make(c_buf, i,
                    shuffle_t::make_broadcast(
                            expr_t(0.0f), step_bytes / sizeof(float))));
        }
        c_zero_out_stmt_ = stmt_group_t::make(
                stmt_label_t::c_zero_out(), c_zero_out_stmt_);
        c_store_stmt_ = c_m2g.stmt();

        // Replace DPAS by DPASW when applicable.
        if (cfg_.use_dpasw) {
            alloc_updater_t alloc_updater;
            inject_dpasw(load_mul_stmt_, c_store_stmt_, alloc_updater,
                    tg_grid_.idx(0));
            for (auto &a : allocs_) {
                a = alloc_updater.update(a);
            }
        }

        // Assign {Atomic} for DPAS(W) when applicable.
        load_mul_stmt_ = inject_atomic(load_mul_stmt_);
    }

private:
    void register_buffer(const stmt_t &alloc) {
        ir_assert(alloc.is<alloc_t>());
        allocs_.push_back(alloc);
    }

    void register_buffer(const expr_t &buf, int size, alloc_kind_t kind,
            const alloc_attr_t &attr = {}) {
        register_buffer(alloc_t::make(buf, size, kind, attr));
    }

    // Handles GMEM to SLM load for A and B. Done in two steps:
    // 1. Load: GMEM -> GRF (temporary)
    // 2. Store: GRF (temporary) -> SLM
    void prepare_gmem_to_slm(const char *tag, bool use_x_slm,
            const view_t &x_tg_view, const expr_t &xp_buf,
            const expr_t &x_g2s_reg_buf, const expr_t &x_slm_buf,
            view_t &xp_slm_view) {
        if (!use_x_slm) return;

        // Per-thread view to load from GMEM to SLM.
        auto x_g2s_view = x_tg_view.split(tg_grid_);

        // GMEM -> GRF load.
        read_builder_t x_read(
                cset_, x_g2s_view, xp_buf, x_g2s_reg_buf, /*is_slm=*/false);
        ir_trace() << tag << " GMEM to GRF load:\n"
                   << x_read.str() << std::endl;

        register_buffer(
                x_g2s_reg_buf, x_read.reg_buf_size(), alloc_kind_t::grf);
        g2s_load_stmt_ = g2s_load_stmt_.append(x_read.stmt());

        auto xp_slm_view_layout = x_tg_view.create_dense_vlayout();
        if (cfg_.pad_slm)
            xp_slm_view_layout = pad_slm_layout(xp_slm_view_layout);
        xp_slm_view = view_t(x_tg_view, xp_slm_view_layout);
        register_buffer(
                x_slm_buf, xp_slm_view_layout.size(), alloc_kind_t::slm);

        // GRF -> SLM store.
        write_builder_t x_write(cset_,
                xp_slm_view.create_sub_view(
                        x_g2s_view.vtensor(), /*relative_vstart=*/false),
                x_slm_buf, x_g2s_reg_buf, /*is_slm=*/true);
        ir_trace() << tag << " GRF to SLM store:\n"
                   << x_write.str() << std::endl;

        g2s_store_stmt_ = g2s_store_stmt_.append(x_write.stmt());

        ir_assert(x_read.reg_view().has_same_vlayout(x_write.reg_view()))
                << "Requested register layouts for " << tag << " do not match.";
    }

    // SLM has 65 dword-granularity banks (Gen12HP):
    //      banks:   [bank 0] [bank 1] [bank 2] ... [bank 0]
    // byte offsets: | 0      | 4      | 8      ... | 4 * 65
    // SLM reads don't have conflicts. During SLM writes each fused EU writes
    // 64 bytes (in total 128 bytes per clock). If there are repeating banks
    // between 128 bytes the write takes 2 clocks to complete.
    // Assume that every X-axis thread (across tg_dim[0]) writes the
    // corresponding outer block of the layout. The goal is to ensure that the
    // stride between outer blocks allows to avoid duplicated banks.
    layout_t pad_slm_layout(const layout_t &layout) const {
        auto tg_dim0 = tg_grid_.dim(0);
        auto tg_dim1 = tg_grid_.dim(1);
        ir_assert(layout.elems() % tg_dim0 == 0) << layout;

        dim_t inner_block = layout.elems() / tg_grid_.dim(0);
        std::vector<dim_t> multi_blocks = {tg_dim0, inner_block};
        auto l = layout.split_into_multi_blocks(multi_blocks);

        auto padded_blocks = l.blocks();
        dim_t stride = -1;
        dim_t remaining_elems = inner_block;
        bool past_inner_block = false;
        int type_size = layout.type().size();
        for (auto &b : padded_blocks) {
            if (past_inner_block) {
                if (stride == -1) {
                    dim_t per_thr_bytes = inner_block * type_size;
                    ir_assert(per_thr_bytes % tg_dim1 == 0);
                    per_thr_bytes /= tg_dim1;
                    dim_t stride_bytes = find_min_stride_without_conflicts(
                            per_thr_bytes, dim_t(b.stride) * type_size);
                    ir_assert(stride_bytes % type_size == 0);
                    stride = stride_bytes / type_size;
                }
                b.stride = stride;
                stride = b.stride * b.block;
                continue;
            }
            ir_assert(remaining_elems % b.block == 0);
            remaining_elems /= b.block;
            if (remaining_elems == 1) past_inner_block = true;
        }
        return layout_t(
                layout.type(), layout.ndims(), layout.offset(), padded_blocks);
    }

    dim_t find_min_stride_without_conflicts(
            dim_t inner_bytes, dim_t dense_stride_bytes) const {
        int write_step = 64;
        int stride_step = 16;
        dim_t stride_beg = dense_stride_bytes;
        dim_t stride_end = 2 * dense_stride_bytes;
        const int slm_banks = 65;
        for (dim_t s = stride_beg; s < stride_end; s += stride_step) {
            bool ok = true;
            for (dim_t off0 = 0; off0 < inner_bytes; off0 += write_step) {
                // Check banks for a single SLM write.
                bool found[slm_banks] = {false};
                for (dim_t off = off0; off < off0 + write_step;
                        off += sizeof(uint32_t)) {
                    int bank0 = (off / sizeof(uint32_t)) % slm_banks;
                    int bank1 = ((off + s) / sizeof(uint32_t)) % slm_banks;
                    if (found[bank0]) {
                        ok = false;
                        break;
                    }
                    found[bank0] = true;
                    if (found[bank1]) {
                        ok = false;
                        break;
                    }
                    found[bank1] = true;
                }
                if (ok) return s;
            }
        }

        ir_error_not_expected();
        return dense_stride_bytes;
    }

    view_t create_cp_thr_mem_view(
            const view_t &ap_thr_view, const view_t &bp_thr_view) const {
        std::vector<dim_t> thr_dims(cp_tg_view_.nvdims(), 1);
        std::vector<expr_t> thr_start(cp_tg_view_.nvdims(), 0);

        for (int i = 0; i < cp_tg_view_.nvdims(); i++) {
            auto &cvar = cp_tg_view_.vvar(i);

            bool found = false;
            for (int j = 0; j < ap_thr_view.nvdims(); j++) {
                if (ap_thr_view.vvar(j).is_same(cvar)) {
                    found = true;
                    thr_dims[i] = ap_thr_view.vdims()[j];

                    auto off = ap_tg_view_.vstart(j) - cp_tg_view_.vstart(i);
                    thr_start[i] = simplify(ap_thr_view.vstart(j) - off);
                }
            }
            if (found) continue;
            for (int j = 0; j < bp_thr_view.nvdims(); j++) {
                if (bp_thr_view.vvar(j).is_same(cvar)) {
                    found = true;
                    thr_dims[i] = bp_thr_view.vdims()[j];

                    auto off = bp_tg_view_.vstart(j) - cp_tg_view_.vstart(i);
                    thr_start[i] = simplify(bp_thr_view.vstart(j) - off);
                }
            }
            ir_assert(found) << "Unknown dimension: " << cvar;
        }
        return cp_tg_view_.create_sub_view(
                tensor_t(thr_dims, thr_start), /*relative_vstart=*/false);
    }

    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    const constraint_set_t &cset_;

    grid_info_t tg_grid_;

    int m_tg_blk_;
    int n_tg_blk_;
    int k_tg_blk_;

    expr_t ap_buf_;
    expr_t bp_buf_;
    expr_t cp_buf_;

    std::vector<stmt_t> allocs_;

    // Views with dimensions in problem notation.
    view_t ap_tg_view_;
    view_t bp_tg_view_;
    view_t cp_tg_view_;

    stmt_t g2s_load_stmt_;
    stmt_t g2s_store_stmt_;
    stmt_t load_mul_stmt_;
    stmt_t c_zero_out_stmt_;
    stmt_t c_store_stmt_;
};

void kernel_builder_t::build() {
    // Only 2D thread groups are supported for now.
    if (cfg_.tg_grid_dim[2] != 1) ir_error_not_implemented();

    ir_context_t ir_ctx;
    constraint_set_t init_cset;

    int grid_ndims = 3;
    kernel_grid_ = grid_info_t(grid_ndims);
    tg_grid_ = grid_info_t(grid_ndims);
    for (int i = 0; i < grid_ndims; i++) {
        local_id_[i]
                = var_t::make(type_t::u16(), "local_id" + std::to_string(i));
        kernel_grid_.dim(i) = cfg_.kernel_grid_dim[i];
        kernel_grid_.idx(i)
                = var_t::make(type_t::s32(), "grid_idx" + std::to_string(i));
        tg_grid_.dim(i) = cfg_.tg_grid_dim[i];
        tg_grid_.idx(i)
                = var_t::make(type_t::s32(), "tg_idx" + std::to_string(i));

        init_cset.add_constraint(kernel_grid_.idx(i) >= 0);
        init_cset.add_constraint(kernel_grid_.idx(i) < cfg_.kernel_grid_dim[i]);
        init_cset.add_constraint(tg_grid_.idx(i) >= 0);
        init_cset.add_constraint(tg_grid_.idx(i) < cfg_.tg_grid_dim[i]);
    }

    std::vector<stmt_t> init_stmts;
    for (int i = 0; i < grid_ndims; i++) {
        auto value = local_id_[i];
        if (i == 0) value /= cfg_.simd_size;
        init_stmts.push_back(let_t::make(tg_grid_.idx(i), value));
    }

    // Initialize memory buffers.
    auto src_buf = kernel_arg_info_.find_arg("src");
    auto wei_buf = kernel_arg_info_.find_arg("wei");
    auto dst_buf = kernel_arg_info_.find_arg("dst");

    std::vector<stmt_t> reduction_loops;
    view_t ap_tg_view;
    view_t bp_tg_view;
    view_t cp_tg_view;

    init_fwd(init_cset, init_stmts, reduction_loops, ap_tg_view, bp_tg_view,
            cp_tg_view);

    compute_builder_t cb(cfg_, ir_ctx, init_cset);

    cb.set_thread_group(tg_grid_);
    cb.set_ap_buf(src_buf);
    cb.set_bp_buf(wei_buf);
    cb.set_cp_buf(dst_buf);
    cb.set_ap_tg_view(ap_tg_view);
    cb.set_bp_tg_view(bp_tg_view);
    cb.set_cp_tg_view(cp_tg_view);
    cb.set_m_tg_blk(cfg_.m_tg_blk);
    cb.set_n_tg_blk(cfg_.n_tg_blk);
    cb.set_k_tg_blk(cfg_.k_tg_blk);

    cb.build();

    std::vector<stmt_t> allocs;
    for (int i = 0; i < kernel_arg_info_.nargs(); i++) {
        auto &var = kernel_arg_info_.arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }
    allocs.insert(allocs.end(), cb.allocs().begin(), cb.allocs().end());

    // Create IR statements.
    stmt_t loop_stmt = cb.iter_stmt();
    for (auto &l : reduction_loops) {
        auto &_for = l.as<for_t>();
        loop_stmt = for_t::make(_for.var, _for.init, _for.bound, loop_stmt);
    }
    loop_stmt = stmt_group_t::make(stmt_label_t::compute_loop(), loop_stmt);

    auto c_store_stmt
            = stmt_group_t::make(stmt_label_t::c_store(), cb.c_store_stmt());
    stmt_ = stmt_seq_t::make(loop_stmt, c_store_stmt);
    for (auto it = init_stmts.rbegin(); it != init_stmts.rend(); ++it) {
        auto &let = it->as<let_t>();
        stmt_ = let_t::make(let.var, let.value, stmt_);
    }

    stmt_ = stmt_seq_t::make(cb.c_zero_out_stmt(), stmt_);

    for (auto it = allocs.rbegin(); it != allocs.rend(); ++it) {
        auto &alloc = it->as<alloc_t>();
        if (alloc.kind != alloc_kind_t::global) {
            ir_assert(alloc.size > 0) << *it;
        }
        stmt_ = alloc_t::make(
                alloc.buf, alloc.size, alloc.kind, alloc.attr, stmt_);
    }

    stmt_ = inject_external_var_let(stmt_);
    stmt_ = merge_slm_buffers(stmt_);
    stmt_ = lift_buffer_offsets_in_send(stmt_);
    stmt_ = simplify(stmt_, init_cset);
    stmt_ = inject_send(stmt_, ir_ctx, init_cset);
    stmt_ = lift_alloc(stmt_);
    stmt_ = eliminate_common_subexprs(stmt_, ir_ctx);
    stmt_ = hoist_exprs(stmt_, ir_ctx);
    stmt_ = loop_strength_reduce(stmt_);
    stmt_ = optimize_let(stmt_);
    stmt_ = update_loops_for_slm_buffering(stmt_);
    stmt_ = inject_slm_buffering(stmt_, cfg_, ir_ctx);
    stmt_ = simplify(stmt_, init_cset);
    stmt_ = optimize_let(stmt_);
    stmt_ = stmt_group_t::make(stmt_label_t::kernel(), stmt_);

    ir_trace() << "Kernel body:\n" << stmt_ << std::endl;
}

void kernel_builder_t::init_fwd(constraint_set_t &init_cset,
        std::vector<stmt_t> &init_stmts, std::vector<stmt_t> &reduction_loops,
        view_t &src_tg_view, view_t &wei_tg_view, view_t &dst_tg_view) {
    // Reduction variables.
    auto ic_blk_idx = var_t::make(type_t::s32(), "ic_blk_idx");
    auto ic_idx = ic_blk_idx * cfg_.ic_blk;
    auto kd_idx = var_t::make(type_t::s32(), "kd_idx");
    auto kh_idx = var_t::make(type_t::s32(), "kh_idx");
    auto kw_idx = var_t::make(type_t::s32(), "kw_idx");

    // Loops are ordered from innermost to outermost.
    reduction_loops.emplace_back(for_t::make(kw_idx, 0, cfg_.kw));
    reduction_loops.emplace_back(for_t::make(kh_idx, 0, cfg_.kh));
    reduction_loops.emplace_back(for_t::make(kd_idx, 0, cfg_.kd));
    reduction_loops.emplace_back(
            for_t::make(ic_blk_idx, 0, utils::div_up(cfg_.ic, cfg_.ic_blk)));

    // Variables.
    auto mb_tg_blk_idx = var_t::make(type_t::s32(), "mb_tg_blk_idx");
    auto oc_tg_blk_idx = var_t::make(type_t::s32(), "oc_tg_blk_idx");
    auto odhw_tg_blk_idx = var_t::make(type_t::s32(), "odhw_tg_blk_idx");

    auto od_tg_idx = var_t::make(type_t::s32(), "od_tg_idx");
    auto oh_tg_idx = var_t::make(type_t::s32(), "oh_tg_idx");
    auto ow_tg_idx = var_t::make(type_t::s32(), "ow_tg_idx");

    init_stmts.push_back(let_t::make(oc_tg_blk_idx, kernel_grid_.idx(0)));
    init_stmts.push_back(let_t::make(odhw_tg_blk_idx, kernel_grid_.idx(1)));
    init_stmts.push_back(let_t::make(mb_tg_blk_idx, kernel_grid_.idx(2)));

    auto mb_tg_idx = mb_tg_blk_idx * cfg_.mb_tg_blk;
    auto oc_tg_idx = oc_tg_blk_idx * cfg_.oc_tg_blk;
    init_cset.add_constraint(oc_tg_idx % cfg_.oc_tg_blk == 0);
    init_stmts.push_back(let_t::make(
            od_tg_idx, (odhw_tg_blk_idx / cfg_.ow_tg_dim) / cfg_.oh));
    init_stmts.push_back(let_t::make(
            oh_tg_idx, (odhw_tg_blk_idx / cfg_.ow_tg_dim) % cfg_.oh));
    init_stmts.push_back(let_t::make(
            ow_tg_idx, (odhw_tg_blk_idx % cfg_.ow_tg_dim) * cfg_.ow_tg_blk));

    // Reshape layouts to 3D spatial to unify code.
    int old_spatial_ndims = cfg_.ndims - 2;
    auto src_layout = normalize_spatial(
            cfg_.src_layout, old_spatial_ndims, cfg_.reduced_to_1d);
    auto wei_layout = normalize_spatial(
            cfg_.wei_layout, old_spatial_ndims, cfg_.reduced_to_1d);
    auto dst_layout = normalize_spatial(
            cfg_.dst_layout, old_spatial_ndims, cfg_.reduced_to_1d);

    // Initialize thread group views.
    auto mb = var_t::make(type_t::s32(), "mb");
    auto ic = var_t::make(type_t::s32(), "ic");
    auto oc = var_t::make(type_t::s32(), "oc");
    auto od = var_t::make(type_t::s32(), "od");
    auto oh = var_t::make(type_t::s32(), "oh");
    auto ow = var_t::make(type_t::s32(), "ow");
    auto kd = var_t::make(type_t::s32(), "kd");
    auto kh = var_t::make(type_t::s32(), "kh");
    auto kw = var_t::make(type_t::s32(), "kw");

    // Initialize masks.
    expr_t id_mask, ih_mask, iw_mask;
    expr_t od_mask, oh_mask, ow_mask;
    expr_t src_mb_mask, dst_mb_mask;
    expr_t wei_oc_mask, dst_oc_mask;

    auto need_src_check = [&](int o, int i, int k, int p, int s, int d) {
        if (cfg_.is_fwd) {
            int i_min = -p;
            int i_max = (o - 1) * s - p + (k - 1) * (1 + d);
            return (i_min < 0) || (i_max >= i);
        }
        // Backward.
        int os_min = p - (k - 1) * (1 + d);
        int os_max = (o - 1) + p;
        return (os_min < 0) || (os_max >= i * s);
    };

    bool check_ow = (cfg_.ow % cfg_.ow_tg_blk != 0);
    bool check_iw = check_ow
            || need_src_check(
                    cfg_.ow, cfg_.iw, cfg_.kw, cfg_.pw, cfg_.sw, cfg_.dw);
    bool check_ih = need_src_check(
            cfg_.oh, cfg_.ih, cfg_.kh, cfg_.ph, cfg_.sh, cfg_.dh);
    bool check_id = need_src_check(
            cfg_.od, cfg_.id, cfg_.kd, cfg_.pd, cfg_.sd, cfg_.dd);

    int wei_oc = int(wei_layout.dim(cfg_.with_groups ? 1 : 0));
    int dst_oc = int(dst_layout.dim(1));
    int wei_oc_inner_blk
            = wei_oc / int(wei_layout.outer_block(cfg_.with_groups ? 1 : 0));
    int dst_oc_inner_blk = dst_oc / int(dst_layout.outer_block(1));

    bool check_wei_oc = (wei_oc % cfg_.oc_tg_blk != 0);
    bool check_dst_oc = (dst_oc % cfg_.oc_tg_blk != 0);

    int src_mb = int(src_layout.dim(0));
    int dst_mb = int(src_layout.dim(0));

    bool check_src_mb = (src_mb % cfg_.mb_tg_blk != 0);
    bool check_dst_mb = (dst_mb % cfg_.mb_tg_blk != 0);

    auto &x = view_t::placeholder_var();
    if (check_id) id_mask = (x >= 0) & (x < cfg_.id);
    if (check_ih) ih_mask = (x >= 0) & (x < cfg_.ih);
    if (check_iw) iw_mask = (x >= 0) & (x < cfg_.iw);
    if (check_ow) ow_mask = (x >= 0) & (x < cfg_.ow);
    if (check_wei_oc)
        wei_oc_mask = (x / wei_oc_inner_blk < wei_oc / wei_oc_inner_blk);
    if (check_dst_oc)
        dst_oc_mask = (x / dst_oc_inner_blk < dst_oc / dst_oc_inner_blk);
    if (check_src_mb) src_mb_mask = (x < src_mb);
    if (check_dst_mb) dst_mb_mask = (x < dst_mb);

    // Source.
    src_tg_view = view_t({mb, ic, od, oh, ow, kd, kh, kw}, 5);
    src_tg_view.set_vdim(mb, cfg_.mb_tg_blk, mb_tg_idx, mnk_kind_t::m);
    src_tg_view.set_vdim(ic, cfg_.ic_blk, ic_idx, mnk_kind_t::k);
    src_tg_view.set_vdim(od, 1, od_tg_idx, mnk_kind_t::m);
    src_tg_view.set_vdim(oh, 1, oh_tg_idx, mnk_kind_t::m);
    src_tg_view.set_vdim(ow, cfg_.ow_tg_blk, ow_tg_idx, mnk_kind_t::m);
    src_tg_view.set_vdim(kd, 1, kd_idx, mnk_kind_t::k);
    src_tg_view.set_vdim(kh, 1, kh_idx, mnk_kind_t::k);
    src_tg_view.set_vdim(kw, 1, kw_idx, mnk_kind_t::k);
    src_tg_view.set_tdim(0, mb, src_mb_mask); // mb
    src_tg_view.set_tdim(1, ic); // ic
    src_tg_view.set_tdim(2, od * cfg_.sd - cfg_.pd + kd * (1 + cfg_.dd),
            id_mask); // id
    src_tg_view.set_tdim(3, oh * cfg_.sh - cfg_.ph + kh * (1 + cfg_.dh),
            ih_mask); // ih
    src_tg_view.set_tdim(4, ow * cfg_.sw - cfg_.pw + kw * (1 + cfg_.dw),
            iw_mask); // iw
    src_tg_view.set_tlayout(src_layout);

    // Weights.
    wei_tg_view = view_t({oc, ic, kd, kh, kw}, 5);
    wei_tg_view.set_vdim(oc, cfg_.oc_tg_blk, oc_tg_idx, mnk_kind_t::n);
    wei_tg_view.set_vdim(ic, cfg_.ic_blk, ic_idx, mnk_kind_t::k);
    wei_tg_view.set_vdim(kd, 1, kd_idx, mnk_kind_t::k);
    wei_tg_view.set_vdim(kh, 1, kh_idx, mnk_kind_t::k);
    wei_tg_view.set_vdim(kw, 1, kw_idx, mnk_kind_t::k);
    wei_tg_view.set_tdim(0, oc, wei_oc_mask); // oc
    wei_tg_view.set_tdim(1, ic); // ic
    wei_tg_view.set_tdim(2, kd); // kd
    wei_tg_view.set_tdim(3, kh); // kh
    wei_tg_view.set_tdim(4, kw); // kw
    wei_tg_view.set_tlayout(wei_layout);

    // Destination.
    dst_tg_view = view_t({mb, oc, od, oh, ow}, 5);
    dst_tg_view.set_vdim(mb, cfg_.mb_tg_blk, mb_tg_idx, mnk_kind_t::m);
    dst_tg_view.set_vdim(oc, cfg_.oc_tg_blk, oc_tg_idx, mnk_kind_t::n);
    dst_tg_view.set_vdim(od, 1, od_tg_idx, mnk_kind_t::m);
    dst_tg_view.set_vdim(oh, 1, oh_tg_idx, mnk_kind_t::m);
    dst_tg_view.set_vdim(ow, cfg_.ow_tg_blk, ow_tg_idx, mnk_kind_t::m);
    dst_tg_view.set_tdim(0, mb, dst_mb_mask); // mb
    dst_tg_view.set_tdim(1, oc, dst_oc_mask); // oc
    dst_tg_view.set_tdim(2, od, od_mask); // od
    dst_tg_view.set_tdim(3, oh, oh_mask); // oh
    dst_tg_view.set_tdim(4, ow, ow_mask); // ow
    dst_tg_view.set_tlayout(dst_layout);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

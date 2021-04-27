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

#ifndef GPU_JIT_CONV_KERNEL_ARG_INFO_HPP
#define GPU_JIT_CONV_KERNEL_ARG_INFO_HPP

#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive_exec_types.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/jit/conv/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Stores kernel arguments. Kernel arguments can be:
// - Internal arguments: only scalar
//   - Examples: common output scales (contain a single value)
// - Resource arguments: stored to a resource storage during primitive creation
//   - Examples: output scales or zero points
// - User arguments: passed by the user at run time
//   - Examples: source, weights, destination
class kernel_arg_info_t {
public:
    void register_internal_arg(const expr_t &var, const expr_t &value) {
        register_arg(var, arg_kind_t::internal, -1, /*is_input=*/true, value);
    }

    void register_resource_arg(const expr_t &var) {
        register_arg(var, arg_kind_t::resource, /*is_input=*/true, nargs());
    }

    void register_user_arg(const expr_t &var, int dnnl_arg, bool is_input) {
        register_arg(var, arg_kind_t::user, dnnl_arg, is_input);
    }

    const std::string &arg_name(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].var.as<var_t>().name;
    }

    const expr_t &arg_var(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].var;
    }

    const type_t &arg_type(int idx) const { return arg_var(idx).type(); }

    expr_t find_arg(const std::string &name) const {
        for (int i = 0; i < nargs(); i++) {
            if (arg_name(i) == name) return args_[i].var;
        }
        ir_error_not_expected();
        return expr_t();
    }

    int dnnl_arg(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].dnnl_arg;
    }

    int nargs() const { return int(args_.size()); }

    bool is_resource(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].kind == arg_kind_t::resource;
    }

    void set_args(const exec_ctx_t &ctx, const gpu_primitive_t *primitive,
            compute::kernel_arg_list_t &arg_list) const {
        int idx = 0;
        for (int i = 0; i < nargs(); i++) {
            bool is_input = args_[i].is_input;
            switch (args_[i].kind) {
                case arg_kind_t::internal: {
                    auto &value = args_[i].value;
                    auto &type = value.type();
                    if (type == type_t::f32()) {
                        arg_list.set(idx++, to_cpp<float>(value));
                    } else {
                        ir_error_not_expected();
                    }
                    break;
                }
                case arg_kind_t::resource: {
                    auto &arg_storage
                            = res_storage(ctx, primitive, args_[i].dnnl_arg);
                    arg_list.set(idx++, arg_storage);
                    break;
                }
                case arg_kind_t::user: {
                    auto &arg_storage
                            = (is_input ? *ctx.input(args_[i].dnnl_arg)
                                                    ->memory_storage()
                                        : *ctx.output(args_[i].dnnl_arg)
                                                    ->memory_storage());
                    arg_list.set(idx++, arg_storage);
                    break;
                }
                default: ir_error_not_expected();
            }
        }
    }

private:
    enum class arg_kind_t { internal, resource, user };

    struct arg_t {
        arg_t(const expr_t &var, arg_kind_t kind, int dnnl_arg, bool is_input,
                const expr_t &value)
            : var(var)
            , kind(kind)
            , dnnl_arg(dnnl_arg)
            , is_input(is_input)
            , value(value) {}

        expr_t var;
        arg_kind_t kind;
        int dnnl_arg;
        bool is_input;
        expr_t value; // For internal arguments, must be a constant.
    };

    static const memory_storage_t &res_storage(
            const exec_ctx_t &ctx, const gpu_primitive_t *primitive, int arg) {
#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
        return *(primitive->cached_mapper()
                         ->template get<gpu_resource_t>(primitive)
                         ->get_memory_storage(arg));
#else
        return *(ctx.get_resource_mapper()
                         ->get<gpu_resource_t>(primitive)
                         ->get_memory_storage(arg));
#endif
    }

    void register_arg(const expr_t &var, arg_kind_t kind, int dnnl_arg,
            bool is_input, const expr_t &value = expr_t()) {
        ir_assert(is_var(var)) << "Expected var, got: " << var;
        args_.emplace_back(var, kind, dnnl_arg, is_input, value);
    }

    std::vector<arg_t> args_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

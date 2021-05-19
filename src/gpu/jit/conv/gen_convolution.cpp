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

#include "gpu/jit/conv/gen_convolution.hpp"

#include <iostream>

#include "common/reorder.hpp"
#include "common/utils.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/conv_kernel.hpp"
#include "gpu/jit/conv/kernel_arg_info.hpp"
#include "gpu/jit/conv/utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace compute;

class gen_convolution_t {
public:
    template <typename T>
    static status_t init_pd(T *pd, engine_t *engine) {
        auto *compute_engine = utils::downcast<compute_engine_t *>(engine);

        if (!compute_engine->is_xe_hp()
#if DNNL_WITH_XE_HPG
                && !compute_engine->is_xe_hpg()
#endif
#if DNNL_WITH_XE_HPC
                && !compute_engine->is_xe_hpc()
#endif
        )
            return status::unimplemented;
        if (!compute_engine->mayiuse_ngen_kernels())
            return status::unimplemented;
        if (!pd->set_default_alg_kind(alg_kind::convolution_direct))
            return status::unimplemented;
        pd->cfg = std::make_shared<conv_config_t>();
        pd->kernel_arg_info = std::make_shared<kernel_arg_info_t>();
        CHECK(pd->cfg->init(pd, engine));

        CHECK(init_kernel_arg_info(pd, *pd->kernel_arg_info));

        return status::success;
    }

    gen_convolution_t() = default;

    template <typename T>
    status_t init(T *primitive, engine_t *engine) {
        ir_trace() << "Configuration:" << std::endl;
        ir_trace() << get_cfg(primitive);

        auto compute_engine = utils::downcast<compute_engine_t *>(engine);
        auto device_info = compute_engine->device_info();

        std::unique_ptr<jit::jit_generator_base> jit_gen_convolution;
        switch (device_info->gpu_arch()) {
            case gpu_arch_t::xe_hp:
                jit_gen_convolution.reset(new conv_kernel_t<gpu_xe_hp>(
                        get_cfg(primitive), primitive->pd(),
                        *primitive->pd()->kernel_arg_info));
                break;
#if DNNL_WITH_XE_HPG
            case gpu_arch_t::xe_hpg:
                jit_gen_convolution.reset(new conv_kernel_t<gpu_xe_hpg>(
                        get_cfg(primitive), primitive->pd(),
                        *primitive->pd()->kernel_arg_info));
                break;
#endif
#if DNNL_WITH_XE_HPC
            case gpu_arch_t::xe_hpc:
                jit_gen_convolution.reset(new conv_kernel_t<gpu_xe_hpc>(
                        get_cfg(primitive), primitive->pd(),
                        *primitive->pd()->kernel_arg_info));
                break;
#endif
            default: return status::unimplemented;
        }
        CHECK(primitive->create_kernel(engine, &kernel_, *jit_gen_convolution));

        return status::success;
    }

    template <typename T>
    status_t execute(const T *primitive, const exec_ctx_t &ctx) const {
        auto &kernel_arg_info = *primitive->pd()->kernel_arg_info;

        std::vector<memory_storage_wrapper_t> storage_list;
        kernel_arg_info.init_memory_storage_list(storage_list, ctx, primitive);

        kernel_arg_list_t arg_list;
        kernel_arg_info.set_args(arg_list, storage_list);

        auto &cfg = get_cfg(primitive);
        auto *compute_stream
                = utils::downcast<compute_stream_t *>(ctx.stream());

        if (cfg.zero_out_output) {
            for (int i = 0; i < kernel_arg_info.nargs(); i++) {
                if (kernel_arg_info.is_input(i)) continue;

                int key = kernel_arg_info.key(i);
                if (kernel_arg_info.is_scratchpad(i)) {
                    if (!utils::one_of(key,
                                memory_tracking::names::key_conv_wei_reduction,
                                memory_tracking::names::key_conv_bia_reduction))
                        continue;
                }
                if (kernel_arg_info.is_user(i)) {
                    if (!utils::one_of(
                                key, DNNL_ARG_DIFF_WEIGHTS, DNNL_ARG_DIFF_BIAS))
                        continue;
                }

                const auto &storage
                        = kernel_arg_info.arg_storage(i, ctx, primitive);
                size_t size = kernel_arg_info.arg_size(i, primitive);
                CHECK(compute_stream->fill(*storage.get(), 0, size));
            }
        }

        auto nd_range = cfg.nd_range();
        CHECK(primitive->parallel_for(ctx, nd_range, kernel_, arg_list));

        return status::success;
    }

private:
    template <typename T>
    static const conv_config_t &get_cfg(const T *primitive) {
        return *primitive->pd()->cfg;
    }

    template <typename T>
    static status_t init_kernel_arg_info(
            const T *pd, kernel_arg_info_t &kernel_arg_info) {
        auto &cfg = *pd->cfg;
        auto *attr = pd->attr();

        // Initialize main arguments.
        if (cfg.is_fwd) {
            kernel_arg_info.register_user_arg(
                    make_buffer("src"), DNNL_ARG_SRC, /*is_input=*/true);
            kernel_arg_info.register_user_arg(
                    make_buffer("wei"), DNNL_ARG_WEIGHTS, /*is_input=*/true);
            kernel_arg_info.register_user_arg(
                    make_buffer("dst"), DNNL_ARG_DST, /*is_input=*/false);
        } else if (cfg.is_bwd_d) {
            kernel_arg_info.register_user_arg(
                    make_buffer("dst"), DNNL_ARG_DIFF_DST, /*is_input=*/true);
            kernel_arg_info.register_user_arg(
                    make_buffer("wei"), DNNL_ARG_WEIGHTS, /*is_input=*/true);
            kernel_arg_info.register_user_arg(
                    make_buffer("src"), DNNL_ARG_DIFF_SRC, /*is_input=*/false);
        } else if (cfg.is_bwd_w) {
            kernel_arg_info.register_user_arg(
                    make_buffer("src"), DNNL_ARG_SRC, /*is_input=*/true);
            kernel_arg_info.register_user_arg(
                    make_buffer("dst"), DNNL_ARG_DIFF_DST, /*is_input=*/true);
            if (!cfg.do_post_wei_reorder) {
                kernel_arg_info.register_user_arg(make_buffer("wei"),
                        DNNL_ARG_DIFF_WEIGHTS, /*is_input=*/false);
            }
            if (cfg.with_bias && !cfg.do_post_bia_reorder) {
                kernel_arg_info.register_user_arg(make_buffer("bia"),
                        DNNL_ARG_DIFF_BIAS, /*is_input=*/false);
            }
        } else {
            ir_error_not_expected();
        }

        // Initialize post-op arguments.
        if ((cfg.is_fwd || cfg.is_bwd_d) && cfg.with_bias) {
            kernel_arg_info.register_user_arg(
                    make_buffer("bia"), DNNL_ARG_BIAS, /*is_input=*/true);
        }

        bool with_oscales = !attr->output_scales_.has_default_values();
        if (with_oscales) {
            bool is_runtime_oscales = !attr->output_scales_.defined();
            bool is_common_oscales = (attr->output_scales_.mask_ == 0);
            if (is_runtime_oscales) {
                kernel_arg_info.register_user_arg(make_buffer("oscales"),
                        DNNL_ARG_ATTR_OUTPUT_SCALES, /*is_input=*/true);
            } else if (is_common_oscales) {
                auto oscales_buf = var_t::make(type_t::f32(), "oscales");
                auto value = float_imm_t::make(attr->output_scales_.scales_[0]);
                kernel_arg_info.register_internal_arg(oscales_buf, value);
            } else {
                kernel_arg_info.register_resource_arg(make_buffer("oscales"));
            }
        }

        for (int i = 0; i < attr->post_ops_.len(); i++) {
            auto &po = attr->post_ops_.entry_[i];
            if (po.is_binary()) {
                auto buf = make_buffer("binary_rhs_" + std::to_string(i));
                kernel_arg_info.register_user_arg(buf,
                        DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1,
                        /*is_input=*/true);
            }
        }

        return status::success;
    }

    kernel_t kernel_;
};

status_t gen_convolution_fwd_t::pd_t::init(engine_t *engine) {
    if (!is_fwd()) return status::unimplemented;
    CHECK(gen_convolution_t::init_pd(this, engine));
    return status::success;
}

status_t gen_convolution_fwd_t::init(engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_convolution_fwd_t::init_res_storage(
        engine_t *engine, gpu_resource_t *r) const {
    auto &kernel_arg_info = *pd()->kernel_arg_info;
    for (int i = 0; i < kernel_arg_info.nargs(); i++) {
        if (!kernel_arg_info.is_resource(i)) continue;

        auto &arg_name = kernel_arg_info.arg_name(i);
        int key = kernel_arg_info.key(i);
        if (arg_name == "oscales") {
            CHECK(init_output_scales_res_storage(engine, r, key));
        } else {
            ir_error_not_expected();
        }
    }
    return status::success;
}

status_t gen_convolution_bwd_data_t::pd_t::init(engine_t *engine) {
    if (!is_bwd_d()) return status::unimplemented;
    CHECK(gen_convolution_t::init_pd(this, engine));
    return status::success;
}

status_t gen_convolution_bwd_weights_t::pd_t::init(engine_t *engine) {
    if (!is_bwd_w()) return status::unimplemented;
    CHECK(gen_convolution_t::init_pd(this, engine));

    if (cfg->do_post_wei_reorder) {
        tmp_wei_md = *diff_weights_md();
        tmp_wei_md.data_type = data_type::f32;
        CHECK(reorder_primitive_desc_create(wei_reorder_pd, engine, &tmp_wei_md,
                diff_weights_md(), nullptr));
    }

    if (cfg->do_post_bia_reorder) {
        tmp_bia_md = *diff_weights_md(1);
        tmp_bia_md.data_type = data_type::f32;
        CHECK(reorder_primitive_desc_create(bia_reorder_pd, engine, &tmp_bia_md,
                diff_weights_md(1), nullptr));
    }

    CHECK(init_scratchpad(*kernel_arg_info));
    return status::success;
}

status_t gen_convolution_bwd_weights_t::pd_t::init_scratchpad(
        kernel_arg_info_t &kernel_arg_info) {
    auto scratchpad = scratchpad_registry().registrar();
    if (wei_reorder_pd) {
        size_t tmp_wei_size
                = memory_desc_wrapper(tmp_wei_md).nelems(/*with_padding=*/true)
                * types::data_type_size(data_type::f32);
        scratchpad.book(memory_tracking::names::key_conv_wei_reduction,
                tmp_wei_size, 1, ocl::OCL_BUFFER_ALIGNMENT);
        kernel_arg_info.register_scratchpad_arg(make_buffer("wei"),
                memory_tracking::names::key_conv_wei_reduction,
                /*is_input=*/false, tmp_wei_size);

        scratchpad.book(memory_tracking::names::key_nested,
                wei_reorder_pd->scratchpad_registry().size(), 1,
                ocl::OCL_BUFFER_ALIGNMENT);
    }
    if (bia_reorder_pd) {
        size_t tmp_bia_size
                = memory_desc_wrapper(tmp_bia_md).nelems(/*with_padding=*/true)
                * types::data_type_size(data_type::f32);
        scratchpad.book(memory_tracking::names::key_conv_bia_reduction,
                tmp_bia_size, 1, ocl::OCL_BUFFER_ALIGNMENT);
        kernel_arg_info.register_scratchpad_arg(make_buffer("bia"),
                memory_tracking::names::key_conv_bia_reduction,
                /*is_input=*/false, tmp_bia_size);

        scratchpad.book(memory_tracking::names::key_nested + 1,
                bia_reorder_pd->scratchpad_registry().size(), 1,
                ocl::OCL_BUFFER_ALIGNMENT);
    }
    return status::success;
}

status_t gen_convolution_bwd_data_t::init(engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_convolution_bwd_weights_t::init(engine_t *engine) {
    if (pd()->wei_reorder_pd) {
        CHECK(pd()->wei_reorder_pd->create_primitive(wei_reorder_, engine));
    }
    if (pd()->bia_reorder_pd) {
        CHECK(pd()->bia_reorder_pd->create_primitive(bia_reorder_, engine));
    }
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    CHECK(impl_->execute(this, ctx));
    if (wei_reorder_) {
        auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_conv_wei_reduction);
        std::unique_ptr<memory_t> tmp_wei_mem;
        CHECK(safe_ptr_assign(tmp_wei_mem,
                new memory_t(ctx.stream()->engine(), &pd()->tmp_wei_md,
                        std::move(scratchpad))));

        exec_args_t args;
        args[DNNL_ARG_SRC] = memory_arg_t {tmp_wei_mem.get(), true};
        args[DNNL_ARG_DST]
                = memory_arg_t {ctx.output(DNNL_ARG_DIFF_WEIGHTS), false};

        nested_scratchpad_t ns(
                ctx, memory_tracking::names::key_nested, wei_reorder_);
        exec_ctx_t reorder_ctx(ctx, std::move(args));
        reorder_ctx.set_scratchpad_grantor(ns.grantor());

        CHECK(wei_reorder_->execute(reorder_ctx));
    }
    if (bia_reorder_) {
        auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_conv_bia_reduction);
        std::unique_ptr<memory_t> tmp_bia_mem;
        CHECK(safe_ptr_assign(tmp_bia_mem,
                new memory_t(ctx.stream()->engine(), &pd()->tmp_bia_md,
                        std::move(scratchpad))));

        exec_args_t args;
        args[DNNL_ARG_SRC] = memory_arg_t {tmp_bia_mem.get(), true};
        args[DNNL_ARG_DST]
                = memory_arg_t {ctx.output(DNNL_ARG_DIFF_BIAS), false};

        nested_scratchpad_t ns(
                ctx, memory_tracking::names::key_nested + 1, bia_reorder_);
        exec_ctx_t reorder_ctx(ctx, std::move(args));
        reorder_ctx.set_scratchpad_grantor(ns.grantor());

        CHECK(bia_reorder_->execute(reorder_ctx));
    }
    return status::success;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

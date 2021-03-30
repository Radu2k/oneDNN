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

#include "common/utils.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/conv_kernel.hpp"
#include "gpu/jit/conv/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class gen_convolution_t {
public:
    template <typename T>
    static status_t init_pd(T *pd, engine_t *engine, format_tag_t &src_tag,
            format_tag_t &wei_tag, format_tag_t &dst_tag) {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);

        if (!compute_engine->is_gen12hp() && !compute_engine->is_gen12p7())
            return status::unimplemented;
        if (!compute_engine->mayiuse_ngen_kernels())
            return status::unimplemented;
        if (!pd->set_default_alg_kind(alg_kind::convolution_direct))
            return status::unimplemented;
        pd->cfg.reset(new conv_config_t());
        CHECK(pd->cfg->init(pd, engine));

        src_tag = pd->cfg->src_tag();
        wei_tag = pd->cfg->wei_tag();
        dst_tag = pd->cfg->dst_tag();

        return status::success;
    }

    gen_convolution_t() = default;

    template <typename T>
    status_t init(T *primitive, engine_t *engine) {
        ir_trace() << "Configuration:" << std::endl;
        ir_trace() << cfg(primitive);

        using namespace compute;

        auto compute_engine = utils::downcast<compute_engine_t *>(engine);
        auto device_info = compute_engine->device_info();

        std::unique_ptr<jit::jit_generator_base> jit_gen_convolution;
        switch (device_info->gpu_arch()) {
            case gpu_arch_t::gen12hp:
                jit_gen_convolution.reset(
                        new conv_kernel_t<gpu_gen12hp>(cfg(primitive)));
                break;
            case gpu_arch_t::gen12p7:
                jit_gen_convolution.reset(
                        new conv_kernel_t<gpu_gen12p7>(cfg(primitive)));
                break;
            default: return status::unimplemented;
        }
        CHECK(primitive->create_kernel(engine, &kernel_, *jit_gen_convolution));

        return status::success;
    }

    template <typename T>
    status_t execute(const T *primitive, const exec_ctx_t &ctx) const {
        auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
        auto &wei = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
        auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, src);
        arg_list.set(1, wei);
        arg_list.set(2, dst);

        auto nd_range = cfg(primitive).nd_range();
        CHECK(primitive->parallel_for(ctx, nd_range, kernel_, arg_list));

        return status::success;
    }

private:
    template <typename T>
    const conv_config_t &cfg(const T *primitive) const {
        return *primitive->pd()->cfg;
    }

    compute::kernel_t kernel_;
};

status_t gen_convolution_fwd_t::pd_t::init(engine_t *engine) {
    format_tag_t src_tag, dst_tag, wei_tag;
    CHECK(gen_convolution_t::init_pd(this, engine, src_tag, wei_tag, dst_tag));
    bool ok = set_default_formats_common(src_tag, wei_tag, dst_tag);
    return ok ? status::success : status::unimplemented;
}

status_t gen_convolution_fwd_t::init(engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

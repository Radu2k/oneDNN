/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef NGEN_GEN9_SIMPLE_SUM_HPP
#define NGEN_GEN9_SIMPLE_SUM_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "compute/compute.hpp"
#include "ocl/ngen_gen9_simple_sum_kernel_f32.hpp"
#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_sum_pd.hpp"
#include "ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct ngen_gen9_simple_sum_t : public primitive_impl_t {
    struct pd_t : public ocl_sum_pd_t {
        using ocl_sum_pd_t::ocl_sum_pd_t;

        DECLARE_SUM_PD_T("ngen:simple:any", ngen_gen9_simple_sum_t);

        status_t init() {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());
            if (!compute_engine->mayiuse_ngen_kernels())
                return status::unimplemented;

            const int n = n_inputs();

            constexpr auto data_type = data_type::f32;

            bool ok = true && ocl_sum_pd_t::init() == status::success;
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper o_d(dst_md());
            ok = ok && o_d.data_type() == data_type && o_d.is_dense();
            if (!ok) return status::unimplemented;

            for (int i = 0; i < n; ++i) {
                const memory_desc_wrapper i_d(src_md(i));
                if (i_d != o_d) return status::unimplemented;
            }

            return ngen_gen9_simple_sum_kernel_f32_t::init_conf(
                    jss_, src_md(0));
        }
        jit_simple_sum_conf_t jss_;
    };

    ngen_gen9_simple_sum_t(const pd_t *apd) : primitive_impl_t(apd) {
        generator_.reset(new ngen_gen9_simple_sum_kernel_f32_t());
    }

    virtual status_t init() override {
        compute::kernel_ctx_t kernel_ctx;

        auto *gpu_engine = utils::downcast<ocl_gpu_engine_t *>(engine());
        if (!gpu_engine) return status::runtime_error;
        if (!generator_) return status::runtime_error;

        kernel_ = compute::kernel_t(new ocl_gpu_kernel_t(generator_->getKernel(
                gpu_engine->context(), gpu_engine->device())));
        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        compute::compute_stream_t *compute_stream
                = utils::downcast<compute::compute_stream_t *>(ctx.stream());
        auto &output = CTX_OUT_STORAGE(DNNL_ARG_DST);

        const int num_arrs = pd()->n_inputs();
        const memory_desc_wrapper o_d(pd()->dst_md());
        const size_t nelems = o_d.nelems();

        for (int a = 0; a < num_arrs; ++a) {
            auto &input = CTX_IN_STORAGE(DNNL_ARG_MULTIPLE_SRC + a);
            const float scale = pd()->scales()[a];

            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, input);
            arg_list.set(1, output);
            arg_list.set(2, scale);
            arg_list.set(3, a);

            size_t gws[3] = {nelems, 1, 1};
            size_t lws[3] = {1, 1, 1};
            auto nd_range = compute::nd_range_t(gws, lws);
            status_t status
                    = compute_stream->parallel_for(nd_range, kernel_, arg_list);
            if (status != status::success) return status;
        }
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    std::unique_ptr<ngen_gen9_simple_sum_kernel_f32_t> generator_;
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif // NGEN_GEN9_SIMPLE_SUM_HPP

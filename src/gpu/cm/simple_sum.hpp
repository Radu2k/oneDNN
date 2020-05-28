/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_CM_SIMPLE_SUM_HPP
#define GPU_CM_SIMPLE_SUM_HPP

#include "common/c_types_map.hpp"
#include "gpu/cm/kernel_utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_sum_pd.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cm {

struct simple_sum_t : public gpu_primitive_t {
    struct pd_t : public gpu_sum_pd_t {
        using gpu_sum_pd_t::gpu_sum_pd_t;

        DECLARE_SUM_PD_T("cm:ref:any", simple_sum_t);

        status_t init(engine_t *engine) {
            if (!is_cm_enabled()) return status::unimplemented;

            const int n = n_inputs();

            bool ok = gpu_sum_pd_t::init(engine) == status::success
                    && n <= max_num_arrs;
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper o_d(dst_md());
            ok = ok && o_d.is_dense();
            if (!ok) return status::unimplemented;

            for (int i = 0; i < n; ++i) {
                const memory_desc_wrapper i_d(src_md(i));
                if (i_d != o_d) return status::unimplemented;
            }

            return init_conf();
        }

        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        sum_conf_t conf;
    };

    simple_sum_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    virtual status_t init(engine_t *engine) {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);

        compute::kernel_ctx_t kernel_ctx;
        pd()->init_kernel_ctx(kernel_ctx);

        kernel_ = cm::create_kernel(compute_engine, "simple_sum", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const;

    static const int max_num_arrs = 16;

private:
    const pd_t *pd() const { return (const pd_t *)gpu_primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace cm
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

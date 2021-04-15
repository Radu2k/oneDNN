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

#ifndef GPU_JIT_CONV_KERNEL_BUILDER_HPP
#define GPU_JIT_CONV_KERNEL_BUILDER_HPP

#include "common/convolution_pd.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/post_op_support.hpp"
#include "gpu/jit/conv/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class kernel_builder_t {
public:
    kernel_builder_t(const conv_config_t &cfg, const convolution_pd_t *pd,
            kernel_arg_info_t &kernel_arg_info)
        : cfg_(cfg), pd_(pd), kernel_arg_info_(kernel_arg_info) {
        build();
    }

    const stmt_t &stmt() const { return stmt_; }

    const expr_t &kernel_grid_idx(int dim_idx) const {
        return kernel_grid_.idx(dim_idx);
    }

    const expr_t &local_id(int dim_idx) const { return local_id_[dim_idx]; }

private:
    void build();
    void init_fwd(constraint_set_t &init_cset, std::vector<stmt_t> &init_stmts,
            std::vector<stmt_t> &reduction_loops, view_t &src_tg_view,
            view_t &wei_tg_view, view_t &dst_tg_view, view_t &dst_view);
    void init_bwd_data(constraint_set_t &init_cset,
            std::vector<stmt_t> &init_stmts,
            std::vector<stmt_t> &reduction_loops, view_t &src_tg_view,
            view_t &wei_tg_view, view_t &dst_tg_view,
            view_t &dst_view);

    const conv_config_t &cfg_;
    const convolution_pd_t *pd_;
    kernel_arg_info_t &kernel_arg_info_;

    expr_t local_id_[3]; // Local IDs (OpenCL) for the 0-th lane.
    grid_info_t kernel_grid_; // Kernel grid (consisting of thread groups).
    grid_info_t tg_grid_; // Thread group grid (consisting of threads).

    tensor_t a_load_thr_block;
    tensor_t b_load_thr_block;

    stmt_t stmt_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

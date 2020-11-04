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

#include <math.h>

#include "common/primitive_exec_types.hpp"

#include "gpu/ocl/gen12hp_reduction.hpp"
#include "gpu/ocl/ocl_utils.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/scratchpad.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t gen12hp_reduction_t::pd_t::init_conf(engine_t *engine) {
    const reduction_pd_t *pd = this;

    const memory_desc_wrapper src_mdw(pd->src_md());
    const memory_desc_wrapper dst_mdw(pd->dst_md());

    const int ndims = src_mdw.ndims();
    const auto src_dims = src_mdw.md_->dims;
    const auto dst_dims = dst_mdw.md_->dims;
    const auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);

    conf.alg = pd->desc()->alg_kind;
    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
    conf.dst_type = dst_mdw.data_type();
    conf.src_type = src_mdw.data_type();
    conf.ndims = ndims;
    conf.power = pd->desc()->p;
    conf.eps = pd->desc()->eps;
    conf.dispatch = compute_engine->create_dispatch(src_mdw.md_);
    conf.finilize_dispatch = compute_engine->create_dispatch();

    auto is_c_blocked_by
            = [](const memory_desc_wrapper &mdw, const int blockSize) {
                  auto &blk = mdw.blocking_desc();
                  if (blk.inner_nblks == 0) return false;
                  return (blk.inner_idxs[blk.inner_nblks - 1] == 1)
                          && (blk.inner_blks[blk.inner_nblks - 1] == blockSize);
              };

    // TODO: currently it supports only nCx16c, generalize that
    if (!is_c_blocked_by(src_mdw, 16) || !is_c_blocked_by(dst_mdw, 16)
            || src_mdw.blocking_desc().inner_nblks != 1)
        return status::unimplemented;

    conf.div = 1;

    // TODO: remove that assumption, generalize kernel
    if (conf.ndims < 3) { return status_t::dnnl_unimplemented; }

    for (int d = 0; d < ndims; d++) {
        conf.reduce_dims[d] = conf.dst_dims[d] = dim_t {1};
        const bool is_reduction_dim = src_dims[d] != dst_dims[d];
        conf.is_reduction_dim[d] = is_reduction_dim;

        // TODO: generalize that
        if ((d >= 2 && !is_reduction_dim) || (d < 2 && is_reduction_dim)) {
            return status_t::dnnl_unimplemented;
        }

        if (is_reduction_dim) {
            conf.reduce_dims[d] = src_dims[d];
            conf.div *= conf.reduce_dims[d];
        } else {
            conf.dst_dims[d] = src_dims[d];
        }
    }

    conf.initial_hwd_dim = conf.div;

    // TODO: add heuristic to calculate hwd block and vector size
    // based on dimension size and threads count
    conf.hwd_block = 256;
    conf.sub_group_size = 16;
    conf.vector_size = 8;

    // TODO: add support for unaligned cases
    if (conf.initial_hwd_dim % conf.hwd_block != 0) {
        return status_t::dnnl_unimplemented;
    }
    if (conf.hwd_block % (conf.sub_group_size * conf.vector_size) != 0) {
        return status_t::dnnl_unimplemented;
    }
    if (conf.dst_dims[1] % conf.sub_group_size != 0) {
        return status_t::dnnl_unimplemented;
    }

    conf.final_hwd_dim = conf.initial_hwd_dim / conf.hwd_block;

    conf.dispatch.define_dim("INITIAL_IN", 0, conf.dst_dims[0]);
    conf.dispatch.define_dim("INITIAL_IC", 0, conf.dst_dims[1]);
    conf.dispatch.define_dim(
            "INITIAL_HWD_DIM", 0, conf.initial_hwd_dim, conf.hwd_block);
    conf.dispatch.vectorize_dim("INITIAL_IC", conf.sub_group_size);
    conf.dispatch.set_kernel_attr_suffix("INITIAL");
    conf.dispatch.generate();

    conf.finilize_dispatch.define_dim("FINAL_IN", 0, conf.dst_dims[0]);
    conf.finilize_dispatch.define_dim("FINAL_IC", 0, conf.dst_dims[1]);
    conf.finilize_dispatch.set_kernel_attr_suffix("FINAL");
    conf.finilize_dispatch.generate();

    return status::success;
}

static status_t init_kernel_ctx_common(
        compute::kernel_ctx_t &kernel_ctx, const reduction_conf_t &conf) {
    using namespace alg_kind;

    kernel_ctx.set_data_type(conf.src_type);

    kernel_ctx.define_int("IN", conf.dst_dims[0]);
    kernel_ctx.define_int("IC", conf.dst_dims[1]);
    kernel_ctx.define_int("INITIAL_HWD_DIM", conf.initial_hwd_dim);
    kernel_ctx.define_int("FINAL_HWD_DIM", conf.final_hwd_dim);
    kernel_ctx.define_int("HWD_BLOCK", conf.hwd_block);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("VECT_DT_N", conf.vector_size);
    kernel_ctx.define_int("REDUCTION_SIZE", conf.div);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("POWER", conf.power);
    kernel_ctx.define_float("EPS", conf.eps);

    switch (conf.alg) {
        case reduction_max: kernel_ctx.define_int("IS_MAX", 1); break;
        case reduction_min: kernel_ctx.define_int("IS_MIN", 1); break;
        case reduction_mean: kernel_ctx.define_int("IS_MEAN", 1); break;
        case reduction_sum: kernel_ctx.define_int("IS_SUM", 1); break;
        case reduction_mul: kernel_ctx.define_int("IS_MUL", 1); break;
        case reduction_norm_lp_max:
            kernel_ctx.define_int("IS_LP_MAX", 1);
            break;
        case reduction_norm_lp_sum:
            kernel_ctx.define_int("IS_LP_SUM", 1);
            break;
        case reduction_norm_lp_power_p_max:
            kernel_ctx.define_int("IS_P_MAX", 1);
            break;
        case reduction_norm_lp_power_p_sum:
            kernel_ctx.define_int("IS_P_SUM", 1);
            break;
        default: return status::invalid_arguments;
    }

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    def_dispatch(kernel_ctx, conf.dispatch);
    def_dispatch(kernel_ctx, conf.finilize_dispatch);

    return status::success;
}

status_t gen12hp_reduction_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

void gen12hp_reduction_t::pd_t::init_scratchpad() {
    size_t size = conf.dst_dims[0] * conf.dst_dims[1] * conf.final_hwd_dim;

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_reduction, size,
            types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
}

status_t gen12hp_reduction_t::execute_gen12hp(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    std::unique_ptr<memory_storage_t> temp_reduce
            = ctx.get_scratchpad_grantor().get_memory_storage(
                    memory_tracking::names::key_reduction);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t reduction_arg_list;
    reduction_arg_list.set(0, src);
    reduction_arg_list.set(1, *temp_reduce);
    auto initial_nd_range = conf.dispatch.nd_range();
    status_t status = parallel_for(
            ctx, initial_nd_range, initial_kernel, reduction_arg_list);
    if (status != status::success) return status;

    compute::kernel_arg_list_t final_reduction_arg_list;
    final_reduction_arg_list.set(0, *temp_reduce);
    final_reduction_arg_list.set(1, dst);
    auto final_nd_range = conf.finilize_dispatch.nd_range();
    return parallel_for(
            ctx, final_nd_range, final_kernel, final_reduction_arg_list);
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

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

#include "gpu/cm/simple_sum.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cm {

status_t simple_sum_t::pd_t::init_conf() {
    const memory_desc_wrapper d(src_md(0));
    const size_t nelems = d.nelems();
    conf.data_type = d.data_type();
    if (conf.data_type == dnnl_bf16) return status::unimplemented;
    size_t data_type_size = types::data_type_size(conf.data_type);

    int elems_per_oword = 16 / (int)data_type_size;
    if (nelems % elems_per_oword) return status::unimplemented;

    // Handle at most 8 OWords per HW thread.
    conf.block_size = elems_per_oword
            * (int)utils::max_div(nelems / elems_per_oword, 8);

    conf.gws_d[0] = nelems / conf.block_size;
    conf.gws_d[1] = 1;
    conf.gws_d[2] = 1;
    return status::success;
}

status_t simple_sum_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.add_option("-cmc");
    kernel_ctx.define_int("BLOCK_SIZE", conf.block_size);
    kernel_ctx.set_data_type(conf.data_type);
    return status::success;
}

status_t simple_sum_t::execute(const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &output = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const int num_arrs = pd()->n_inputs();
    const memory_desc_wrapper o_d(pd()->dst_md());

    for (int a = 0; a < num_arrs; ++a) {

        auto &input = CTX_IN_STORAGE(DNNL_ARG_MULTIPLE_SRC + a);
        const float scale = pd()->scales()[a];

        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, input);
        arg_list.set(1, output);
        arg_list.set(2, scale);
        arg_list.set(3, a);

        const auto &conf = pd()->conf;
        auto nd_range = compute::nd_range_t(conf.gws_d);
        status_t status
                = compute_stream->parallel_for(nd_range, kernel_, arg_list);
        if (status != status::success) return status;
    }
    return status::success;
}

} // namespace cm
} // namespace gpu
} // namespace impl
} // namespace dnnl

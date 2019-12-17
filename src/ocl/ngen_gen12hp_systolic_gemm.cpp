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

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/float16.hpp"
#include "common/type_helpers.hpp"

#include "ocl/ngen_gen12hp_systolic_gemm.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

std::tuple<int64_t, int64_t, int64_t>
ngen_gen12hp_systolic_gemm_t::get_blocking() const {
    int64_t m = pd()->desc()->m;
    int64_t n = pd()->desc()->n;
    int64_t k = pd()->desc()->k;
    auto dt = pd()->desc()->a_type;

    int64_t unroll_m = ngen_gen12hp_systolic_gemm_kernel_t::unroll_m;
    int64_t unroll_n = ngen_gen12hp_systolic_gemm_kernel_t::unroll_n;
    int64_t unroll_k = ngen_gen12hp_systolic_gemm_kernel_t::unroll_k(dt);

    int64_t align_m
            = unroll_m * ngen_gen12hp_systolic_gemm_kernel_t::thread_group_m;
    int64_t align_n
            = unroll_n * ngen_gen12hp_systolic_gemm_kernel_t::thread_group_n;

    m = utils::rnd_up(m, align_m);
    n = utils::rnd_up(n, align_n);

    // Decide on m/n blocking. Assume 512 EU config for now.
    int64_t block_m = 2048; // Default blocking: 32 * 64
    int64_t block_n = 1536; // Default blocking: 48 * 32
    int64_t max_block_m = nstl::min(m, block_m * 4);
    int64_t max_block_n = nstl::min(n, block_n * 4);

    if (n < block_n)
        block_m = (block_m * block_n) / n;
    else if (m < block_m)
        block_n = (block_m * block_n) / m;
    else if (n < 2 * block_n) {
        block_n = utils::rnd_up(n / 2, align_n);
        block_m = (2 * block_m * block_n) / n;
    } else if (m < 2 * block_m) {
        block_m = utils::rnd_up(m / 2, align_m);
        block_n = (2 * block_m * block_n) / m;
    }

    block_m = utils::rnd_dn(nstl::min(block_m, max_block_m), align_m);
    block_n = utils::rnd_dn(nstl::min(block_n, max_block_n), align_n);

    // Decide on k blocking.
    int64_t block_k = 4608 / types::data_type_size(dt);
    int64_t nblock_k = utils::div_up(k, block_k);
    block_k = utils::div_up(k, nblock_k);
    block_k = utils::rnd_up(block_k, unroll_k);

    return std::make_tuple(block_m, block_n, block_k);
}

status_t ngen_gen12hp_systolic_gemm_t::launch_copy(
        compute::compute_stream_t *compute_stream, int64_t r, int64_t c,
        const memory_storage_t &src, int64_t offset_src, int64_t ld_src,
        const memory_storage_t &dst, int32_t offset_dst, int32_t ld_dst,
        bool copyb) const {

    int64_t unroll_m = ngen_gen12hp_systolic_gemm_kernel_t::unroll_m;
    int64_t unroll_n = ngen_gen12hp_systolic_gemm_kernel_t::unroll_n;
    int64_t unroll_k = ngen_gen12hp_systolic_gemm_kernel_t::unroll_k(
            pd()->desc()->a_type);

    int64_t align_r = 0, align_c = 0;

    if (!copyb) {
        align_r = unroll_m
                * ngen_gen12hp_systolic_gemm_kernel_t::thread_group_m;
        align_c = unroll_k;
    } else {
        align_r = unroll_k;
        align_c = unroll_n
                * ngen_gen12hp_systolic_gemm_kernel_t::thread_group_n;
    }

    bool transa = (pd()->desc()->transa == dnnl_trans);
    bool transb = (pd()->desc()->transb == dnnl_trans);
    bool trans = !copyb ? transa : transb;

    auto &kernel = copy_kernel_[copyb];

    assert(kernel);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, r);
    arg_list.set(1, c);
    arg_list.set(2, src);
    arg_list.set(3, offset_src);
    arg_list.set(4, ld_src);
    arg_list.set(5, dst);
    arg_list.set(6, offset_dst);
    arg_list.set(7, ld_dst);

    size_t r_threads = utils::div_up(utils::rnd_up(r, align_r),
            jit_gen12hp_systolic_gemm_copy_kernel::unroll_r(copyb, trans));
    size_t c_threads = utils::div_up(utils::rnd_up(c, align_c),
            jit_gen12hp_systolic_gemm_copy_kernel::unroll_c(copyb, trans));
    size_t sg = jit_gen12hp_systolic_gemm_copy_kernel::subgroup_size(
            copyb, trans);

    size_t gws[3] = {r_threads * sg, c_threads, 1};
    size_t lws[3] = {sg, 1, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return compute_stream->parallel_for(nd_range, kernel, arg_list);
}

status_t ngen_gen12hp_systolic_gemm_t::launch_compute(
        compute::compute_stream_t *compute_stream, int32_t m, int32_t n,
        int32_t k, const memory_storage_t &ap, int64_t offset_a, int32_t lda,
        const memory_storage_t &bp, int64_t offset_b, int32_t ldb,
        const memory_storage_t &c, int64_t offset_c, int32_t ldc, float alpha,
        float beta) const {

    using kernel_t = ngen_gen12hp_systolic_gemm_kernel_t;
    auto unroll_m = kernel_t::unroll_m;
    auto unroll_n = kernel_t::unroll_n;
    auto unroll_k = kernel_t::unroll_k(pd()->desc()->a_type);
    auto tg_m = kernel_t::thread_group_m;
    auto tg_n = kernel_t::thread_group_n;
    auto sg = kernel_t::nominal_subgroup_size;

    auto &kernel = (beta == 1.0f) ? kernel_b1_ : kernel_;

    //   kernel void gemm_kernel(global char *Ap, global uchar *Bp, global int *C,
    //                           int k, int ldc,
    //                           long offsetA, long offsetB, long offsetC,
    //                           int m, int n,
    //                           float alpha, float beta,
    //                           int lda, int ldb)

    assert(kernel);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, ap);
    arg_list.set(1, bp);
    arg_list.set(2, c);
    arg_list.set(3, utils::div_up(k, unroll_k));
    arg_list.set(4, ldc);
    arg_list.set(5, offset_a);
    arg_list.set(6, offset_b);
    arg_list.set(7, offset_c);
    arg_list.set(8, m);
    arg_list.set(9, n);
    arg_list.set(10, alpha);
    arg_list.set(11, beta);
    arg_list.set(12, lda);
    arg_list.set(13, ldb);

    auto thread_m = utils::div_up(m, unroll_m * tg_m) * tg_m;
    auto thread_n = utils::div_up(n, unroll_n * tg_n) * tg_n;

    size_t gws[3] = {size_t(sg * thread_m), size_t(thread_n), 1};
    size_t lws[3] = {size_t(sg * tg_m), size_t(tg_n), 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return compute_stream->parallel_for(nd_range, kernel, arg_list);
}

status_t ngen_gen12hp_systolic_gemm_t::execute(const exec_ctx_t &ctx) const {

    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto a_type = pd()->desc()->a_type;
    auto b_type = pd()->desc()->b_type;
    auto c_type = pd()->desc()->c_type;

    auto m = pd()->desc()->m;
    auto n = pd()->desc()->n;
    auto k = pd()->desc()->k;

    bool transa = (pd()->desc()->transa == dnnl_trans);
    bool transb = (pd()->desc()->transb == dnnl_trans);

    auto lda = pd()->desc()->lda;
    auto ldb = pd()->desc()->ldb;
    auto ldc = pd()->desc()->ldc;

    auto alpha = pd()->desc()->alpha;
    auto beta = pd()->desc()->beta;

    auto &a = CTX_IN_STORAGE(DNNL_ARG_SRC_0);
    auto &b = CTX_IN_STORAGE(DNNL_ARG_SRC_1);
    auto &c = CTX_OUT_STORAGE(DNNL_ARG_DST);

    size_t off_a0 = a.get_offset() / types::data_type_size(a_type)
            + pd()->dyn_offset_a;
    size_t off_b0 = b.get_offset() / types::data_type_size(b_type)
            + pd()->dyn_offset_b;
    size_t off_c0 = c.get_offset() / types::data_type_size(c_type)
            + pd()->dyn_offset_c;

    int64_t block_m = 0, block_n = 0, block_k = 0;
    std::tie(block_m, block_n, block_k) = get_blocking();

    auto unroll_k = ngen_gen12hp_systolic_gemm_kernel_t::unroll_k(a_type);
    auto lda_packed = utils::rnd_up(block_k, unroll_k);
    auto ldb_packed = lda_packed;

    status_t status;

    for (int64_t Bk = 0; Bk < k; Bk += block_k) {
        int64_t size_k = k - Bk;
        bool last_k_block = (size_k <= block_k);
        if (!last_k_block) size_k = block_k;

        for (int64_t Bm = 0; Bm < m; Bm += block_m) {
            int64_t size_m = m - Bm;
            if (size_m > block_m) size_m = block_m;

            auto off_a = off_a0 + (!transa ? (Bm + Bk * lda) : (Bk + Bm * lda));
            auto off_a_packed = 0;

            status = launch_copy(compute_stream, size_m, size_k, a, off_a, lda,
                    *a_packed_, off_a_packed, lda_packed, false);
            if (status) return status;

            for (int64_t Bn = 0; Bn < n; Bn += block_n) {
                int64_t size_n = n - Bn;
                if (size_n > block_n) size_n = block_n;

                auto off_b = off_b0
                        + (!transb ? (Bk + Bn * ldb) : (Bn + Bk * ldb));
                auto off_b_packed = 0;

                if ((Bm == 0) || (n > block_n)) {
                    status = launch_copy(compute_stream, size_k, size_n, b,
                            off_b, ldb, *b_packed_, off_b_packed, ldb_packed,
                            true);
                    if (status) return status;
                }

                auto off_c = off_c0 + Bm + Bn * ldc;

                float this_beta = (Bk == 0) ? beta : 1.0f;
                status = launch_compute(compute_stream, size_m, size_n, size_k,
                        *a_packed_, off_a_packed, lda_packed, *b_packed_,
                        off_b_packed, ldb_packed, c, off_c, ldc, alpha,
                        this_beta);
                if (status) return status;
            }
        }
    }

    return status::success;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

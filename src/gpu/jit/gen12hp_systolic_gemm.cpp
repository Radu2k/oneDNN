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

#include "gpu/jit/gen12hp_systolic_gemm.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/float16.hpp"
#include "common/type_helpers.hpp"
#include "gpu/jit/gen12hp_systolic_gemm_kernel.hpp"
#include "gpu/jit/ngen_type_bridge.hpp"
#include "gpu/ocl/gemm/gen12hp_systolic_gemm_copy_kernel.hpp"
#include "gpu/ocl/ocl_gpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

status_t gen12hp_systolic_gemm_t::pd_t::init() {
    using namespace prop_kind;
    using namespace data_type;
    using namespace primitive_kind;
    using smask_t = primitive_attr_t::skip_mask_t;

    assert(this->engine()->kind() == engine_kind::gpu);
    auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine());

    const auto d = desc();

    // LIMITATIONS:
    // - batch is not supported
    // - runtime dims are not supported
    // - bias is not supported for f16/bf16
    bool limits_ok = d->batch == 1
            && !utils::one_of(DNNL_RUNTIME_DIM_VAL, d->m, d->n, d->k, d->lda,
                    d->ldb, d->ldc)
            && d->bias_type == data_type::undef;

    bool dt_float_ok
            = (d->a_type == d->b_type && utils::one_of(d->a_type, bf16, f16)
                    && utils::one_of(d->c_type, f32, d->a_type));

    bool dt_int_ok = (utils::one_of(d->a_type, u8, s8)
            && utils::one_of(d->b_type, u8, s8) && (d->c_type == s32));

    const auto attr_skip_mask = dt_int_ok
            ? smask_t::oscale | smask_t::post_ops | smask_t::zero_points_runtime
            : smask_t::none;

    bool ok = true && limits_ok && (dt_float_ok || dt_int_ok)
            && compute_engine->mayiuse(compute::device_ext_t::
                            intel_subgroup_split_matrix_multiply_accumulate)
            && attr()->has_default_values(attr_skip_mask);

    if (dt_int_ok) {
        ok &= attr()->zero_points_.defined(DNNL_ARG_SRC)
                && attr()->zero_points_.defined(DNNL_ARG_WEIGHTS)
                && (attr()->zero_points_.has_default_values(DNNL_ARG_DST)
                        || !attr()->zero_points_.defined(DNNL_ARG_DST))
                && attr()->output_scales_.mask_ == 0
                && attr()->post_ops_.len_ <= 2
                && IMPLICATION(attr()->post_ops_.len_ == 1,
                        attr()->post_ops_.find(eltwise) != -1
                                || attr()->post_ops_.find(sum) != -1)
                && IMPLICATION(attr()->post_ops_.len_ == 2,
                        attr()->post_ops_.find(sum) == 0
                                && attr()->post_ops_.find(eltwise) == 1);

        int cmask = 0;
        attr()->zero_points_.get(DNNL_ARG_DST, nullptr, &cmask, nullptr);
        ok &= utils::one_of(cmask, 0, 1 << 0, 1 << 1);
    }

    if (!ok) return status::unimplemented;

    return status::success;
}

dim_t gen12hp_systolic_gemm_t::pd_t::m_aligned() const {
    using kernel_t = gen12hp_systolic_gemm_kernel_t;
    return utils::rnd_up(
            desc()->m, kernel_t::unroll_m * kernel_t::thread_group_m);
}

dim_t gen12hp_systolic_gemm_t::pd_t::n_aligned() const {
    using kernel_t = gen12hp_systolic_gemm_kernel_t;
    return utils::rnd_up(
            desc()->n, kernel_t::unroll_n * kernel_t::thread_group_n);
}

dim_t gen12hp_systolic_gemm_t::pd_t::k_aligned() const {
    return utils::rnd_up(desc()->k,
            gen12hp_systolic_gemm_kernel_t::unroll_k(desc()->a_type));
}

status_t gen12hp_systolic_gemm_t::init() {
    using namespace data_type;
    using kernel_t = gen12hp_systolic_gemm_kernel_t;

    auto *gpu_engine = utils::downcast<ocl::ocl_gpu_engine_t *>(engine());
    if (!gpu_engine) return status::out_of_memory;

    auto a_type = pd()->desc()->a_type;
    auto b_type = pd()->desc()->b_type;
    auto c_type = pd()->desc()->c_type;
    auto acc_type = pd()->desc()->acc_type;

    if (utils::one_of(acc_type, f16, bf16)) acc_type = f32;

    ab_zero_points_ = (c_type == s32);

    int64_t block_m = 0, block_n = 0, block_k = 0;
    std::tie(block_m, block_n, block_k) = get_blocking();

    auto max_ldab_packed
            = kernel_t::max_ld_packed(block_k, a_type, ab_zero_points_);

    memory_storage_t *a_packed_ptr, *b_packed_ptr;
    this->engine()->create_memory_storage(&a_packed_ptr,
            block_m * max_ldab_packed * types::data_type_size(a_type));
    this->engine()->create_memory_storage(&b_packed_ptr,
            block_n * max_ldab_packed * types::data_type_size(b_type));
    if (!a_packed_ptr || !b_packed_ptr) return status::runtime_error;
    a_packed_.reset(a_packed_ptr);
    b_packed_.reset(b_packed_ptr);

    // Initialize compute kernels (assembly)
    kernel_t::config_t cfg;

    cfg.a_type = convert_dnnl_type_to_ngen(a_type);
    cfg.b_type = convert_dnnl_type_to_ngen(b_type);
    cfg.c_type = convert_dnnl_type_to_ngen(c_type);
    cfg.acc_type = convert_dnnl_type_to_ngen(acc_type);
    cfg.alpha1 = (pd()->alpha() == 1.0f);
    cfg.beta0 = (pd()->beta() == 0.0f);
    cfg.beta1 = (pd()->beta() == 1.0f);
    cfg.a_bias = cfg.b_bias = ab_zero_points_;

    co_type_ = 'N';
    if (c_type == s32) {
        int cmask = 0;
        pd()->attr()->zero_points_.get(DNNL_ARG_DST, nullptr, &cmask, nullptr);

        switch (cmask) {
            case 0:
                cfg.c_bias = kernel_t::bias_t::fixed;
                co_type_ = 'F';
                break;
            case (1 << 1):
                cfg.c_bias = kernel_t::bias_t::row;
                co_type_ = 'R';
                break;
            case (1 << 0):
                cfg.c_bias = kernel_t::bias_t::column;
                co_type_ = 'C';
                break;
            default:
                cfg.c_bias = kernel_t::bias_t::none;
                co_type_ = 'N';
                break;
        }
    }

    bool may_k_block = (pd()->desc()->k > default_block_k(a_type));

    for (bool first_k_block : {false, true}) {
        for (bool last_k_block : {false, true}) {
            if ((!first_k_block || !last_k_block) && !may_k_block) continue;
            if (may_k_block && last_k_block
                    && (cfg.c_bias == kernel_t::bias_t::none))
                kernel_[first_k_block][last_k_block]
                        = kernel_[first_k_block][false];
            else if (may_k_block && first_k_block && cfg.beta1)
                kernel_[first_k_block][last_k_block]
                        = kernel_[false][last_k_block];
            else {
                auto cfg_copy = cfg;
                if (!first_k_block) {
                    cfg_copy.beta0 = false;
                    cfg_copy.beta1 = true;
                }
                if (!last_k_block) cfg_copy.c_bias = kernel_t::bias_t::none;

                auto kernel = kernel_t(cfg_copy);
                kernel_[first_k_block][last_k_block] = compute::kernel_t(
                        new ocl::ocl_gpu_kernel_t(kernel.getKernel(
                                gpu_engine->context(), gpu_engine->device())));
            }
        }
    }

    // Initialize copy kernels (OpenCL)
    for (bool copy_b : {false, true}) {
        for (bool clear_sum : {false, true}) {
            if (clear_sum && !ab_zero_points_) continue;

            compute::kernel_ctx_t kernel_ctx;

            auto trans = !copy_b ? pd()->desc()->transa : pd()->desc()->transb;
            auto status
                    = ocl::gen12hp_systolic_gemm_copy_kernel_t::init_kernel_ctx(
                            kernel_ctx, !copy_b ? a_type : b_type, copy_b,
                            trans, ab_zero_points_, clear_sum);
            if (status != status::success) return status;

            gpu_engine->create_kernel(&copy_kernel_[copy_b][clear_sum],
                    "gen12hp_systolic_gemm_copy", kernel_ctx);
            if (!copy_kernel_[copy_b][clear_sum]) return status::runtime_error;
        }
    }

    return status::success;
}

int64_t gen12hp_systolic_gemm_t::default_block_k(data_type_t dt) const {
    return 4608 / types::data_type_size(dt);
}

std::tuple<int64_t, int64_t, int64_t>
gen12hp_systolic_gemm_t::get_blocking() const {
    int64_t m = pd()->desc()->m;
    int64_t n = pd()->desc()->n;
    int64_t k = pd()->desc()->k;
    auto dt = pd()->desc()->a_type;

    int64_t unroll_m = gen12hp_systolic_gemm_kernel_t::unroll_m;
    int64_t unroll_n = gen12hp_systolic_gemm_kernel_t::unroll_n;
    int64_t unroll_k = gen12hp_systolic_gemm_kernel_t::unroll_k(dt);

    int64_t align_m = unroll_m * gen12hp_systolic_gemm_kernel_t::thread_group_m;
    int64_t align_n = unroll_n * gen12hp_systolic_gemm_kernel_t::thread_group_n;

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
    int64_t block_k = default_block_k(dt);
    int64_t nblock_k = utils::div_up(k, block_k);
    block_k = utils::div_up(k, nblock_k);
    block_k = utils::rnd_up(block_k, unroll_k);

    return std::make_tuple(block_m, block_n, block_k);
}

status_t gen12hp_systolic_gemm_t::launch_copy(
        compute::compute_stream_t *compute_stream, int64_t r, int64_t c,
        const memory_storage_t &src, int64_t offset_src, int64_t ld_src,
        const memory_storage_t &dst, int32_t offset_dst, int32_t ld_dst,
        bool copyb) const {

    if (ab_zero_points_) {
        auto status = launch_clear_sum(
                compute_stream, r, c, dst, offset_dst, ld_dst, copyb);
        if (status) return status;
    }

    int64_t unroll_m = gen12hp_systolic_gemm_kernel_t::unroll_m;
    int64_t unroll_n = gen12hp_systolic_gemm_kernel_t::unroll_n;
    int64_t unroll_k
            = gen12hp_systolic_gemm_kernel_t::unroll_k(pd()->desc()->a_type);

    int64_t align_r = 0, align_c = 0;

    if (!copyb) {
        align_r = unroll_m * gen12hp_systolic_gemm_kernel_t::thread_group_m;
        align_c = unroll_k;
    } else {
        align_r = unroll_k;
        align_c = unroll_n * gen12hp_systolic_gemm_kernel_t::thread_group_n;
    }

    bool transa = (pd()->desc()->transa == dnnl_trans);
    bool transb = (pd()->desc()->transb == dnnl_trans);
    bool trans = !copyb ? transa : transb;

    auto &kernel = copy_kernel_[copyb][false];

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

    auto elt_size = types::data_type_size(pd()->desc()->a_type);
    size_t r_threads = utils::div_up(utils::rnd_up(r, align_r),
            ocl::gen12hp_systolic_gemm_copy_kernel_t::unroll_r(
                    elt_size, copyb, trans));
    size_t c_threads = utils::div_up(utils::rnd_up(c, align_c),
            ocl::gen12hp_systolic_gemm_copy_kernel_t::unroll_c(
                    elt_size, copyb, trans));
    size_t sg = ocl::gen12hp_systolic_gemm_copy_kernel_t::subgroup_size(
            elt_size, copyb, trans);

    size_t gws[3] = {r_threads * sg, c_threads, 1};
    size_t lws[3] = {sg, 1, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return compute_stream->parallel_for(nd_range, kernel, arg_list);
}

status_t gen12hp_systolic_gemm_t::launch_clear_sum(
        compute::compute_stream_t *compute_stream, int64_t r, int64_t c,
        const memory_storage_t &dst, int32_t offset_dst, int32_t ld_dst,
        bool copyb) const {

    auto &kernel = copy_kernel_[copyb][true];

    assert(kernel);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, r);
    arg_list.set(1, c);
    arg_list.set(2, dst);
    arg_list.set(3, offset_dst);
    arg_list.set(4, ld_dst);

    auto elt_size = types::data_type_size(pd()->desc()->a_type);
    size_t threads = !copyb
            ? utils::div_up(r, gen12hp_systolic_gemm_kernel_t::unroll_m)
            : utils::div_up(c, gen12hp_systolic_gemm_kernel_t::unroll_n);
    size_t sg
            = ocl::gen12hp_systolic_gemm_copy_kernel_t::subgroup_size_clear_sum(
                    elt_size, copyb);

    size_t gws[3] = {threads * sg, 1, 1};
    size_t lws[3] = {sg, 1, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return compute_stream->parallel_for(nd_range, kernel, arg_list);
}

status_t gen12hp_systolic_gemm_t::launch_compute(
        compute::compute_stream_t *compute_stream, int32_t m, int32_t n,
        int32_t k, const memory_storage_t &ap, int64_t offset_a, int32_t lda,
        const memory_storage_t &bp, int64_t offset_b, int32_t ldb,
        const memory_storage_t &c, int64_t offset_c, int32_t ldc, float alpha,
        float beta, int16_t ao, int16_t bo, const memory_storage_t &co,
        int32_t offset_co, bool first_k_block, bool last_k_block) const {

    using kernel_t = gen12hp_systolic_gemm_kernel_t;
    auto unroll_m = kernel_t::unroll_m;
    auto unroll_n = kernel_t::unroll_n;
    auto tg_m = kernel_t::thread_group_m;
    auto tg_n = kernel_t::thread_group_n;
    auto sg = kernel_t::nominal_subgroup_size;

    auto &kernel = kernel_[first_k_block][last_k_block];

    //   kernel void gemm_kernel(global char *Ap, global uchar *Bp, global int *C,
    //                           int k, int ldc,
    //                           long offsetA, long offsetB, long offsetC,
    //                           int m, int n,
    //                           float alpha, float beta,
    //                           int lda, int ldb)

    assert(kernel);

    compute::kernel_arg_list_t arg_list;
    int argn = 0;
    arg_list.set(argn++, ap);
    arg_list.set(argn++, bp);
    arg_list.set(argn++, c);
    arg_list.set(argn++, k);
    arg_list.set(argn++, ldc);
    arg_list.set(argn++, offset_a);
    arg_list.set(argn++, offset_b);
    arg_list.set(argn++, offset_c);
    arg_list.set(argn++, m);
    arg_list.set(argn++, n);
    arg_list.set(argn++, alpha);
    arg_list.set(argn++, beta);
    arg_list.set(argn++, lda);
    arg_list.set(argn++, ldb);
    if (ab_zero_points_) {
        uint32_t abo = (uint16_t(ao) | (uint16_t(bo) << 16));
        arg_list.set(argn++, abo);
    }
    if (last_k_block && (pd()->desc()->c_type == data_type::s32)) {
        arg_list.set(argn++, co);
        arg_list.set(argn++, offset_co);
    }

    auto thread_m = utils::div_up(m, unroll_m * tg_m) * tg_m;
    auto thread_n = utils::div_up(n, unroll_n * tg_n) * tg_n;

    size_t gws[3] = {size_t(sg * thread_m), size_t(thread_n), 1};
    size_t lws[3] = {size_t(sg * tg_m), size_t(tg_n), 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return compute_stream->parallel_for(nd_range, kernel, arg_list);
}

status_t gen12hp_systolic_gemm_t::execute(const gemm_exec_ctx_t &ctx) const {

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

    auto alpha = pd()->alpha();
    auto beta = pd()->beta();

    auto &a = GEMM_CTX_ARG_STORAGE(a);
    auto &b = GEMM_CTX_ARG_STORAGE(b);
    auto &c = GEMM_CTX_ARG_STORAGE(c);
    auto &co = GEMM_CTX_ARG_STORAGE(c_zero_point);

    int32_t ao = 0, bo = 0;

    size_t off_a0
            = a.offset() / types::data_type_size(a_type) + pd()->dyn_offset_a;
    size_t off_b0
            = b.offset() / types::data_type_size(b_type) + pd()->dyn_offset_b;
    size_t off_c0
            = c.offset() / types::data_type_size(c_type) + pd()->dyn_offset_c;

    if (c_type == data_type::s32) {
        const int *ao_i32 = nullptr;
        const int *bo_i32 = nullptr;
        pd()->attr()->zero_points_.get(DNNL_ARG_SRC, nullptr, nullptr, &ao_i32);
        pd()->attr()->zero_points_.get(
                DNNL_ARG_WEIGHTS, nullptr, nullptr, &bo_i32);
        ao = -*ao_i32;
        bo = -*bo_i32;
    }

    int64_t block_m = 0, block_n = 0, block_k = 0;
    std::tie(block_m, block_n, block_k) = get_blocking();

    auto lda_packed = gen12hp_systolic_gemm_kernel_t::get_ld_packed(
            block_k, a_type, ab_zero_points_);
    auto ldb_packed = lda_packed;

    status_t status;

    for (int64_t Bk = 0; Bk < k; Bk += block_k) {
        int64_t size_k = k - Bk;
        bool first_k_block = (Bk == 0);
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
                auto off_co = 0;
                switch (co_type_) {
                    case 'R': off_co = Bm; break;
                    case 'C': off_co = Bn; break;
                    default: off_co = 0; break;
                }

                float this_beta = first_k_block ? beta : 1.0f;
                status = launch_compute(compute_stream, size_m, size_n, size_k,
                        *a_packed_, off_a_packed, lda_packed, *b_packed_,
                        off_b_packed, ldb_packed, c, off_c, ldc, alpha,
                        this_beta, ao, bo, co, off_co, first_k_block,
                        last_k_block);
                if (status) return status;
            }
        }
    }

    return status::success;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

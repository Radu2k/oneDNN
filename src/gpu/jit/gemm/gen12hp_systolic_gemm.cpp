/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "gpu/jit/gemm/gen12hp_systolic_gemm.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/float16.hpp"
#include "common/type_helpers.hpp"
#include "gpu/jit/gemm/gen12hp_systolic_gemm_kernel.hpp"
#include "gpu/jit/ngen_type_bridge.hpp"
#include "gpu/ocl/gemm/gen12hp_systolic_gemm_copy_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

status_t gen12hp_systolic_gemm_t::pd_t::init(engine_t *engine) {
    using namespace prop_kind;
    using namespace data_type;
    using namespace primitive_kind;
    using smask_t = primitive_attr_t::skip_mask_t;
    using arch_t = compute::gpu_arch_t;

    assert(engine->kind() == engine_kind::gpu);
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    if (!compute_engine->mayiuse_ngen_kernels()) return status::unimplemented;

    auto arch = compute_engine->device_info()->gpu_arch();

    const auto d = desc();

    // Use FMA implementation for small cases.
    if (d->m() < 32 && d->n() < 32) return status::unimplemented;
    if (d->m() < 32 && d->k() < 32) return status::unimplemented;
    if (d->n() < 32 && d->k() < 32) return status::unimplemented;

    bool ok = set_default_formats(d->a_type());
    if (!ok) return status::unimplemented;

    // LIMITATIONS:
    // - batch is not supported for unpacked inputs.
    // - runtime dims are not supported
    bool limits_ok
            = !utils::one_of(DNNL_RUNTIME_DIM_VAL, d->m(), d->n(), d->k());
    if (!packed_a())
        limits_ok = limits_ok && (d->lda() != DNNL_RUNTIME_DIM_VAL)
                && (d->batch() == 1);
    if (!packed_b())
        limits_ok = limits_ok && (d->ldb() != DNNL_RUNTIME_DIM_VAL)
                && (d->batch() == 1);
    if (!packed_c())
        limits_ok = limits_ok && (d->ldc() != DNNL_RUNTIME_DIM_VAL);

    bool dt_float_ok = (d->a_type() == d->b_type()
            && utils::one_of(d->a_type(), bf16, f16)
            && utils::one_of(d->c_type(), f32, d->a_type()));

    bool dt_int_ok = (utils::one_of(d->a_type(), u8, s8)
            && utils::one_of(d->b_type(), u8, s8) && (d->c_type() == s32));

    auto attr_skip_mask = smask_t::oscale | smask_t::post_ops;

    if (dt_int_ok) attr_skip_mask |= smask_t::zero_points_runtime;

    ok = true && limits_ok && (dt_float_ok || dt_int_ok)
            && utils::one_of(arch, arch_t::gen12hp, arch_t::gen12p7)
            && compute_engine->mayiuse(compute::device_ext_t::
                            intel_subgroup_split_matrix_multiply_accumulate)
            && attr()->has_default_values(attr_skip_mask)
            && attr()->output_scales_.mask_ == 0 && attr()->post_ops_.len() <= 2
            && IMPLICATION(attr()->post_ops_.len() == 1,
                    attr()->post_ops_.find(eltwise) != -1
                            || attr()->post_ops_.find(sum) != -1)
            && IMPLICATION(attr()->post_ops_.len() == 2,
                    attr()->post_ops_.find(sum) == 0
                            && attr()->post_ops_.find(eltwise) == 1)
            && IMPLICATION(with_bias(),
                    dt_float_ok
                            && utils::one_of(d->bias_type(), d->a_type(), f32)
                            && utils::one_of(bias_cmask(), 0, 1 << 0, 1 << 1));

    if (dt_int_ok) {
        ok &= attr()->zero_points_.defined(DNNL_ARG_SRC)
                && attr()->zero_points_.defined(DNNL_ARG_WEIGHTS)
                && (attr()->zero_points_.has_default_values(DNNL_ARG_DST)
                        || !attr()->zero_points_.defined(DNNL_ARG_DST));

        int cmask = 0;
        attr()->zero_points_.get(DNNL_ARG_DST, nullptr, &cmask, nullptr);
        ok &= utils::one_of(cmask, 0, 1 << 0, 1 << 1);
        ok &= !packed_a() && !packed_b();
    }

    attr_info_ = attr_info_t::create(attr());

    if (attr_info()->with_eltwise)
        ok &= jit_eltwise_injector_f32<gpu_gen12hp>::is_supported(
                attr_info()->eltwise_alg);

    if (!ok) return status::unimplemented;

    return status::success;
}

status_t gen12hp_systolic_gemm_t::init(engine_t *engine) {
    using namespace data_type;
    using arch_t = compute::gpu_arch_t;

    // Read device information
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    assert(compute_engine);

    arch_ = compute_engine->device_info()->gpu_arch();
    eu_count_ = compute_engine->device_info()->eu_count();

    auto a_type = pd()->desc()->a_type();
    auto b_type = pd()->desc()->b_type();
    auto c_type = pd()->desc()->c_type();
    auto acc_type = pd()->desc()->acc_type;

    if (utils::one_of(acc_type, f16, bf16)) acc_type = f32;

    ab_zero_points_ = (c_type == s32);

    // Initialize compute kernels (assembly)
    using kernel_t = gen12hp_systolic_gemm_kernel_t<gpu_gen12hp>;
    kernel_t::config_t cfg;
    auto attr_info = pd()->attr_info();

    cfg.a_type = convert_dnnl_type_to_ngen(a_type);
    cfg.b_type = convert_dnnl_type_to_ngen(b_type);
    cfg.c_type = convert_dnnl_type_to_ngen(c_type);
    cfg.acc_type = convert_dnnl_type_to_ngen(acc_type);
    cfg.alpha1 = (pd()->alpha() == 1.0f);
    cfg.beta0 = (pd()->beta() == 0.0f);
    cfg.beta1 = (pd()->beta() == 1.0f);
    if (attr_info->with_eltwise) {
        cfg.post_op = attr_info->eltwise_alg;
        cfg.eltwise_alpha = attr_info->eltwise_alpha;
        cfg.eltwise_beta = attr_info->eltwise_beta;
        cfg.eltwise_scale = attr_info->eltwise_scale;
    }
    cfg.a_bias = cfg.b_bias = ab_zero_points_;
    cfg.c_packed = pd()->packed_c();
    cfg.batch = pd()->with_batch();
    walk_n_first_ = cfg.walk_n_first
            = (pd()->desc()->m() >= 2 * pd()->desc()->n());
    unroll_m_ = cfg.tile_m = 32;
    unroll_n_ = cfg.tile_n = 48;

    int cmask = -1;

    if (c_type == s32) {
        cfg.co_type = cfg.c_type;
        pd()->attr()->zero_points_.get(DNNL_ARG_DST, nullptr, &cmask, nullptr);
    } else if (pd()->with_bias()) {
        cfg.early_c_bias = true;
        cfg.co_type = convert_dnnl_type_to_ngen(pd()->desc()->bias_type());
        cmask = pd()->bias_cmask();
    }

    switch (cmask) {
        case 0:
            cfg.c_bias = kernel_t::bias_t::fixed;
            co_kind_ = 'F';
            break;
        case (1 << 1):
            cfg.c_bias = kernel_t::bias_t::row;
            co_kind_ = 'R';
            break;
        case (1 << 0):
            cfg.c_bias = kernel_t::bias_t::column;
            co_kind_ = 'C';
            break;
        case -1:
        default:
            cfg.c_bias = kernel_t::bias_t::none;
            co_kind_ = 'N';
            break;
    }

    bool may_k_block = (pd()->desc()->k() > default_block_k(a_type));

    for (bool first_k_block : {false, true}) {
        for (bool last_k_block : {false, true}) {
            if ((!first_k_block || !last_k_block) && !may_k_block) continue;
            if (may_k_block && last_k_block
                    && (cfg.c_bias == kernel_t::bias_t::none)
                    && !cfg.have_post_op())
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
                if (!last_k_block) {
                    cfg_copy.c_bias = kernel_t::bias_t::none;
                    cfg_copy.post_op = alg_kind::undef;
                }

                switch (arch_) {
                    case arch_t::gen12hp: {
                        auto kernel = kernel_t(cfg_copy);

                        create_kernel(engine,
                                &kernel_[first_k_block][last_k_block], kernel);
                        break;
                    }
                    case arch_t::gen12p7: {
                        using kernel_12p7_t
                                = gen12hp_systolic_gemm_kernel_t<gpu_gen12p7>;
                        cfg_copy.emulate64 = true;
                        auto kernel = kernel_12p7_t(
                                reinterpret_cast<kernel_12p7_t::config_t &>(
                                        cfg_copy));

                        create_kernel(engine,
                                &kernel_[first_k_block][last_k_block], kernel);
                        break;
                    }
                    default:
                        assert(!"Unsupported GPU architecture.");
                        return status::unimplemented;
                        break;
                }
            }
        }
    }

    // Initialize copy kernels (OpenCL)
    for (bool copy_b : {false, true}) {
        for (bool clear_sum : {false, true}) {
            if (clear_sum && !ab_zero_points_) continue;
            if (!copy_b ? pd()->packed_a() : pd()->packed_b()) continue;

            compute::kernel_ctx_t kernel_ctx;

            auto trans
                    = !copy_b ? pd()->desc()->transa() : pd()->desc()->transb();
            auto status
                    = ocl::gen12hp_systolic_gemm_copy_kernel_t::init_kernel_ctx(
                            kernel_ctx, !copy_b ? a_type : b_type, copy_b,
                            trans, ab_zero_points_, clear_sum);
            if (status != status::success) return status;

            create_kernel(engine, &copy_kernel_[copy_b][clear_sum],
                    "gen12hp_systolic_gemm_copy", kernel_ctx);
            if (!copy_kernel_[copy_b][clear_sum]) return status::runtime_error;
        }
    }

    return status::success;
}

status_t gen12hp_systolic_gemm_t::init_res_storage(
        engine_t *engine, gpu_resource_t *r) const {
    using kernel_t = gen12hp_systolic_gemm_kernel_t<gpu_gen12hp>;

    auto a_type = pd()->desc()->a_type();
    auto b_type = pd()->desc()->b_type();

    auto m = pd()->desc()->m();
    auto n = pd()->desc()->n();
    auto k = pd()->desc()->k();

    int64_t align_m = unroll_m_ * kernel_t::thread_group_m;
    int64_t align_n = unroll_n_ * kernel_t::thread_group_n;

    auto m_aligned = utils::rnd_up(m, align_m);
    auto n_aligned = utils::rnd_up(n, align_n);

    auto max_ldab_packed = kernel_t::max_ld_packed(k, a_type, ab_zero_points_);

    if (!pd()->packed_a()) {
        memory_storage_t *a_packed_ptr;
        engine->create_memory_storage(&a_packed_ptr,
                m_aligned * max_ldab_packed * types::data_type_size(a_type));
        if (!a_packed_ptr) return status::runtime_error;

        std::unique_ptr<memory_storage_t> a_packed(a_packed_ptr);
        r->add_memory_storage(A_PACKED_, std::move(a_packed));
    }

    if (!pd()->packed_b()) {
        memory_storage_t *b_packed_ptr;
        engine->create_memory_storage(&b_packed_ptr,
                n_aligned * max_ldab_packed * types::data_type_size(b_type));
        if (!b_packed_ptr) return status::runtime_error;

        std::unique_ptr<memory_storage_t> b_packed(b_packed_ptr);
        r->add_memory_storage(B_PACKED_, std::move(b_packed));
    }

    return status::success;
}

bool gen12hp_systolic_gemm_t::enable_mn_blocking() const {
    return (pd()->desc()->m() >= 8192) && (pd()->desc()->n() >= 8192);
}

int64_t gen12hp_systolic_gemm_t::default_block_m() const {
    return 1024; // 8 thread groups in m dimension
}

int64_t gen12hp_systolic_gemm_t::default_block_n() const {
    return eu_count_
            * 6; // Up to 16 thread groups in n dimension, enough to fill GPU.
}

int64_t gen12hp_systolic_gemm_t::default_block_k(data_type_t dt) const {
    return 8192 / types::data_type_size(dt);
}

std::tuple<int64_t, int64_t, int64_t>
gen12hp_systolic_gemm_t::get_blocking() const {
    using kernel_t = gen12hp_systolic_gemm_kernel_t<gpu_gen12hp>;

    int64_t m = pd()->desc()->m();
    int64_t n = pd()->desc()->n();
    int64_t k = pd()->desc()->k();
    auto dt = pd()->desc()->a_type();

    int64_t unroll_k = kernel_t::unroll_k(dt);

    int64_t align_m = unroll_m_ * kernel_t::thread_group_m;
    int64_t align_n = unroll_n_ * kernel_t::thread_group_n;

    m = utils::rnd_up(m, align_m);
    n = utils::rnd_up(n, align_n);

    // Decide on m/n blocking.
    int64_t block_m = default_block_m();
    int64_t block_n = default_block_n();
    int64_t max_block_m = utils::rnd_up(m, align_m);
    int64_t max_block_n = utils::rnd_up(n, align_n);

    if (enable_mn_blocking()) {
        if (n <= block_n)
            block_m = (block_m * block_n) / n;
        else if (m <= block_m)
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
    } else {
        block_m = m;
        block_n = n;
    }

    // Decide on k blocking.
    int64_t block_k = default_block_k(dt);
    int64_t nblock_k = utils::div_up(k, block_k);
    block_k = utils::div_up(k, nblock_k);
    block_k = utils::rnd_up(
            (pd()->desc()->acc_type != pd()->desc()->c_type()) ? k : block_k,
            unroll_k);

    return std::make_tuple(block_m, block_n, block_k);
}

status_t gen12hp_systolic_gemm_t::launch_copy(const gemm_exec_ctx_t &ctx,
        int64_t r, int64_t c, const memory_storage_t &src, int64_t offset_src,
        int64_t ld_src, const memory_storage_t &dst, int32_t offset_dst,
        int32_t ld_dst, bool copyb) const {

    using compute_kernel_t = gen12hp_systolic_gemm_kernel_t<gpu_gen12hp>;
    using copy_kernel_t = ocl::gen12hp_systolic_gemm_copy_kernel_t;

    if (ab_zero_points_) {
        auto status
                = launch_clear_sum(ctx, r, c, dst, offset_dst, ld_dst, copyb);
        if (status) return status;
    }

    int64_t unroll_k = compute_kernel_t::unroll_k(pd()->desc()->a_type());

    int64_t align_r = 0, align_c = 0;

    if (!copyb) {
        align_r = unroll_m_ * compute_kernel_t::thread_group_m;
        align_c = unroll_k;
    } else {
        align_r = unroll_k;
        align_c = unroll_n_ * compute_kernel_t::thread_group_n;
    }

    bool transa = (pd()->desc()->transa() == dnnl_trans);
    bool transb = (pd()->desc()->transb() == dnnl_trans);
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

    auto elt_size = types::data_type_size(pd()->desc()->a_type());
    size_t r_threads = utils::div_up(utils::rnd_up(r, align_r),
            copy_kernel_t::unroll_r(elt_size, copyb, trans));
    size_t c_threads = utils::div_up(utils::rnd_up(c, align_c),
            copy_kernel_t::unroll_c(elt_size, copyb, trans));
    size_t sg = copy_kernel_t::subgroup_size(elt_size, copyb, trans);

    size_t gws[3] = {r_threads * sg, c_threads, 1};
    size_t lws[3] = {sg, 1, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, kernel, arg_list);
}

status_t gen12hp_systolic_gemm_t::launch_clear_sum(const gemm_exec_ctx_t &ctx,
        int64_t r, int64_t c, const memory_storage_t &dst, int32_t offset_dst,
        int32_t ld_dst, bool copyb) const {

    auto &kernel = copy_kernel_[copyb][true];

    assert(kernel);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, r);
    arg_list.set(1, c);
    arg_list.set(2, dst);
    arg_list.set(3, offset_dst);
    arg_list.set(4, ld_dst);

    auto elt_size = types::data_type_size(pd()->desc()->a_type());
    size_t threads = !copyb ? utils::div_up(r, unroll_m_)
                            : utils::div_up(c, unroll_n_);
    size_t sg
            = ocl::gen12hp_systolic_gemm_copy_kernel_t::subgroup_size_clear_sum(
                    elt_size, copyb);

    size_t gws[3] = {threads * sg, 1, 1};
    size_t lws[3] = {sg, 1, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, kernel, arg_list);
}

status_t gen12hp_systolic_gemm_t::launch_compute(const gemm_exec_ctx_t &ctx,
        int32_t m, int32_t n, int32_t k, const memory_storage_t &ap,
        int64_t offset_a, int32_t lda, const memory_storage_t &bp,
        int64_t offset_b, int32_t ldb, const memory_storage_t &c,
        int64_t offset_c, int32_t ldc, float alpha, float beta, int16_t ao,
        int16_t bo, const memory_storage_t &co, int32_t offset_co,
        bool first_k_block, bool last_k_block, int32_t batch, int32_t stride_a,
        int32_t stride_b, int32_t stride_c) const {

    using kernel_t = gen12hp_systolic_gemm_kernel_t<gpu_gen12hp>;
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
    if (last_k_block
            && (pd()->with_bias()
                    || pd()->desc()->c_type() == data_type::s32)) {
        arg_list.set(argn++, co);
        arg_list.set(argn++, offset_co);
    }
    if (pd()->with_batch()) {
        arg_list.set(argn++, stride_a);
        arg_list.set(argn++, stride_b);
        arg_list.set(argn++, stride_c);
    }

    auto thread_m = utils::div_up(m, unroll_m_ * tg_m) * tg_m;
    auto thread_n = utils::div_up(n, unroll_n_ * tg_n) * tg_n;

    if (walk_n_first_) std::swap(thread_m, thread_n);

    size_t gws[3] = {size_t(sg * thread_m), size_t(thread_n), 1};
    size_t lws[3] = {size_t(sg * tg_m), size_t(tg_n), 1};
    if (pd()->with_batch()) gws[2] = batch;

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, kernel, arg_list);
}

status_t gen12hp_systolic_gemm_t::execute(const gemm_exec_ctx_t &ctx) const {

    using compute_kernel_t = gen12hp_systolic_gemm_kernel_t<gpu_gen12hp>;

    auto a_type = pd()->desc()->a_type();
    auto b_type = pd()->desc()->b_type();
    auto c_type = pd()->desc()->c_type();
    auto bias_type = pd()->desc()->bias_type();
    auto co_type = c_type;

    auto m = pd()->desc()->m();
    auto n = pd()->desc()->n();
    auto k = pd()->desc()->k();
    auto batch = pd()->desc()->batch();

    bool packed_a = pd()->packed_a();
    bool packed_b = pd()->packed_b();
    bool packed_c = pd()->packed_c();

    auto lda = packed_a ? 0 : pd()->desc()->lda();
    auto ldb = packed_b ? 0 : pd()->desc()->ldb();
    auto ldc = packed_c ? pd()->ldc_packed() : pd()->desc()->ldc();

    auto stride_a = pd()->desc()->stride_a();
    auto stride_b = pd()->desc()->stride_b();
    auto stride_c = pd()->desc()->stride_c();

    auto alpha = pd()->alpha();
    auto beta = pd()->beta();

    auto &a = GEMM_CTX_ARG_STORAGE(b);
    auto &b = GEMM_CTX_ARG_STORAGE(a);
    auto &c = GEMM_CTX_ARG_STORAGE(c);
    auto &c_zp = GEMM_CTX_ARG_STORAGE(c_zero_point);
    auto &bias = GEMM_CTX_ARG_STORAGE(bias);
    auto *co = &c_zp;

    auto &a_packed = packed_a ? a : CTX_GPU_RES_STORAGE(A_PACKED_);
    auto &b_packed = packed_b ? b : CTX_GPU_RES_STORAGE(B_PACKED_);

    int32_t ao = 0, bo = 0;

    size_t off_a0
            = a.offset() / types::data_type_size(a_type) + pd()->dyn_offset_a;
    size_t off_b0
            = b.offset() / types::data_type_size(b_type) + pd()->dyn_offset_b;
    size_t off_c0
            = c.offset() / types::data_type_size(c_type) + pd()->dyn_offset_c;
    size_t off_co0 = 0;

    if (c_type == data_type::s32) {
        const int *ao_i32 = nullptr;
        const int *bo_i32 = nullptr;
        pd()->attr()->zero_points_.get(DNNL_ARG_SRC, nullptr, nullptr, &ao_i32);
        pd()->attr()->zero_points_.get(
                DNNL_ARG_WEIGHTS, nullptr, nullptr, &bo_i32);
        ao = -*ao_i32;
        bo = -*bo_i32;
    } else if (pd()->with_bias()) {
        off_co0 = bias.offset() / types::data_type_size(bias_type);
        co = &bias;
        co_type = bias_type;
    }

    int64_t block_m = 0, block_n = 0, block_k = 0;
    std::tie(block_m, block_n, block_k) = get_blocking();

    auto ld_packed
            = compute_kernel_t::get_ld_packed(k, a_type, ab_zero_points_);
    auto lda_packed = packed_a ? pd()->lda_packed() : ld_packed;
    auto ldb_packed = packed_b ? pd()->ldb_packed() : ld_packed;

    status_t status;

    if (!packed_a) {
        assert(batch == 1);
        status = launch_copy(
                ctx, m, k, a, off_a0, lda, a_packed, 0, lda_packed, false);
        if (status) return status;
    }

    if (!packed_b) {
        assert(batch == 1);
        status = launch_copy(
                ctx, k, n, b, off_b0, ldb, b_packed, 0, ldb_packed, true);
        if (status) return status;
    }

    for (int64_t Bk = 0; Bk < k; Bk += block_k) {
        int64_t size_k = k - Bk;
        bool first_k_block = (Bk == 0);
        bool last_k_block = (size_k <= block_k);
        if (!last_k_block) size_k = block_k;

        for (int64_t Bm = 0; Bm < m; Bm += block_m) {
            int64_t size_m = m - Bm;
            if (size_m > block_m) size_m = block_m;

            auto off_a_packed = Bm * lda_packed + Bk * unroll_m_;
            if (packed_a) off_a_packed += off_a0;

            for (int64_t Bn = 0; Bn < n; Bn += block_n) {
                int64_t size_n = n - Bn;
                if (size_n > block_n) size_n = block_n;

                auto off_b_packed = Bn * ldb_packed + Bk * unroll_n_;
                if (packed_b) off_b_packed += off_b0;

                auto off_c = off_c0 + Bm + Bn * ldc;
                auto off_co = int32_t(off_co0);
                switch (co_kind_) {
                    case 'R': off_co += Bm; break;
                    case 'C': off_co += Bn; break;
                    default: break;
                }

                float this_beta = first_k_block ? beta : 1.0f;
                status = launch_compute(ctx, size_m, size_n, size_k, a_packed,
                        off_a_packed, lda_packed, b_packed, off_b_packed,
                        ldb_packed, c, off_c, ldc, alpha, this_beta, ao, bo,
                        *co, off_co, first_k_block, last_k_block, batch,
                        stride_a, stride_b, stride_c);
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

/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/primitive_desc_iface.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/reorder.hpp"
#include "common/stream.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_ncsp_conv.hpp"

#define VCHECK_CONV(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, convolution, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VINFO_CONV(msg, ...) \
    VINFO(primitive, create, check, convolution, "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;

namespace {
format_tag_t get_ncsp_tag(int ndims) {
    using namespace format_tag;
    switch (ndims) {
        case 3: return ncw;
        case 4: return nchw;
        case 5: return ncdhw;
        default: assert("invalid ndims"); return undef;
    }
}
format_tag_t get_nspc_tag(int ndims) {
    using namespace format_tag;
    switch (ndims) {
        case 3: return nwc;
        case 4: return nhwc;
        case 5: return ndhwc;
        default: assert("invalid ndims"); return undef;
    }
}
} // namespace

status_t ncsp_convolution_fwd_t::pd_t::init_convolution(engine_t *engine) {
    // create a convolution descriptor with activations in nspc format
    convolution_desc_t nspc_conv_d = convolution_desc_t();
    format_tag_t nspc_tag = get_nspc_tag(ndims());
    nspc_src_md_ = *src_md();
    nspc_dst_md_ = *dst_md();
    CHECK(memory_desc_init_by_tag(nspc_src_md_, nspc_tag));
    CHECK(memory_desc_init_by_tag(nspc_dst_md_, nspc_tag));

    const convolution_desc_t *ncsp_conv_d = desc();
    CHECK(conv_desc_init(&nspc_conv_d, ncsp_conv_d->prop_kind,
            ncsp_conv_d->alg_kind, &nspc_src_md_, &ncsp_conv_d->weights_desc,
            &ncsp_conv_d->bias_desc, &nspc_dst_md_, ncsp_conv_d->strides,
            ncsp_conv_d->dilates, ncsp_conv_d->padding[0],
            ncsp_conv_d->padding[1]));

    primitive_desc_iterator_t it(engine,
            reinterpret_cast<const op_desc_t *>(&nspc_conv_d), attr(), nullptr);
    if (!it.is_initialized()) return status::out_of_memory;

    if (++it == it.end()) return status::unimplemented;
    nspc_conv_pd_ = *it;

    if (weights_md_.format_kind == format_kind::any)
        weights_md_ = *nspc_conv_pd_->weights_md(0);
    if (bias_md_.format_kind == format_kind::any)
        bias_md_ = *nspc_conv_pd_->weights_md(1);

    CHECK(reorder_primitive_desc_create(
            src_reorder_pd_, engine, src_md(), &nspc_src_md_));
    if (with_sum_)
        CHECK(reorder_primitive_desc_create(
                dst_pre_reorder_pd_, engine, dst_md(), &nspc_dst_md_));
    CHECK(reorder_primitive_desc_create(
            dst_post_reorder_pd_, engine, &nspc_dst_md_, dst_md()));
    // VINFO_CONV("embedded primitive implementation is %s", nspc_conv_pd_->name());
    return status::success;
}

status_t ncsp_convolution_fwd_t::pd_t::init_matmul(engine_t *engine) {
    const bool to_matmul = true;
    CHECK(reduce.reshape_activations(
            &matmul_dst_md_, dst_md(0), to_matmul, true));

    // initialize convolution bias as 1d plain tensor
    if (bias_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_strides(bias_md_, nullptr));

    // For call to matmul:
    // - conv src becomes matmul weights (ie matrix B)
    // - conv weights becomes matmul src (ie matrix A)
    // This allows to keep conv src and conv dst in ncsp layout.
    CHECK(reduce.reshape_activations(
            &matmul_wei_md_, src_md(0), to_matmul, false));
    CHECK(reduce.reshape_weights(&matmul_src_md_, weights_md(0), to_matmul));
    if (with_bias()) CHECK(reduce.reshape_bias(&matmul_bia_md_, weights_md(1)));
    primitive_desc_iface_t *matmul_pdi;
    primitive_attr_t _attr;
    post_ops_t _po;
    if (bias_po_ && with_bias()) {
        CHECK(_po.append_binary(alg_kind::binary_add, &matmul_bia_md_));
        CHECK(_attr.set_post_ops(_po));
    }
    CHECK(dnnl_matmul_primitive_desc_create(&matmul_pdi, engine,
            &matmul_src_md_, &matmul_wei_md_,
            !bias_po_ && with_bias() ? &matmul_bia_md_ : nullptr,
            &matmul_dst_md_, &_attr));
    matmul_pd_ = matmul_pdi->impl();

    if (weights_md_.format_kind == format_kind::any)
        CHECK(reduce.reshape_weights(
                &weights_md_, matmul_pd_->src_md(), false /*to_matmul*/));

    return status::success;
}

status_t ncsp_convolution_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;
    using namespace utils;

    // TODO: enable attributes (could be tricky for binary-like postops)
    VCHECK_CONV(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    VCHECK_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

    VCHECK_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);

    VCHECK_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);

    VCHECK_CONV(memory_desc_matches_tag(*src_md(), get_ncsp_tag(ndims())),
            VERBOSE_UNSUPPORTED_TAG);
    VCHECK_CONV(memory_desc_matches_tag(*dst_md(), get_ncsp_tag(ndims())),
            VERBOSE_UNSUPPORTED_TAG);

    // TODO: support bias and attributes in matmul-based convolution
    // (bias can be supported via binary postop, attr might need translation)
    is_matmul_ = reduce.is_gemm() && attr()->has_default_values();

    if (is_matmul_)
        CHECK(init_matmul(engine));
    else
        CHECK(init_convolution(engine));

    init_name();
    init_scratchpad();
    return status::success;
}

void ncsp_convolution_fwd_t::pd_t::init_scratchpad() {
    using namespace memory_tracking::names;
    auto scratchpad = scratchpad_registry().registrar();
    if (is_matmul_) {
        if (matmul_pd_)
            scratchpad.book(key_nested, matmul_pd_->scratchpad_registry());
    } else {
        const memory_desc_wrapper dst_mdw(dst_md());
        const memory_desc_wrapper src_mdw(src_md());
        scratchpad.book(key_conv_ncsp_dst, dst_mdw.nelems(),
                sizeof(dst_mdw.data_type()));
        scratchpad.book(key_conv_ncsp_src, src_mdw.nelems(),
                sizeof(src_mdw.data_type()));
        if (nspc_conv_pd_)
            scratchpad.book(key_nested, nspc_conv_pd_->scratchpad_registry());
        if (src_reorder_pd_)
            scratchpad.book(key_nested, src_reorder_pd_->scratchpad_registry());
        if (dst_pre_reorder_pd_)
            scratchpad.book(
                    key_nested, dst_pre_reorder_pd_->scratchpad_registry());
        if (dst_post_reorder_pd_)
            scratchpad.book(
                    key_nested, dst_post_reorder_pd_->scratchpad_registry());
    }
}

status_t ncsp_convolution_fwd_t::init(engine_t *engine) {
    if (pd()->matmul_pd_)
        CHECK(pd()->matmul_pd_->create_primitive(matmul_p_, engine));
    if (pd()->nspc_conv_pd_)
        CHECK(pd()->nspc_conv_pd_->create_primitive(nspc_conv_p_, engine));
    if (pd()->src_reorder_pd_)
        CHECK(pd()->src_reorder_pd_->create_primitive(src_reorder_p_, engine));
    if (pd()->dst_pre_reorder_pd_)
        CHECK(pd()->dst_pre_reorder_pd_->create_primitive(
                dst_pre_reorder_p_, engine));
    if (pd()->dst_post_reorder_pd_)
        CHECK(pd()->dst_post_reorder_pd_->create_primitive(
                dst_post_reorder_p_, engine));
    return status::success;
}

status_t ncsp_convolution_fwd_t::reorder_activations(const exec_ctx_t &ctx,
        const std::shared_ptr<primitive_t> prim, engine_t *engine,
        const memory_arg_t &in, const memory_arg_t &out) const {
    using namespace memory_tracking::names;
    exec_args_t r_args;
    r_args[DNNL_ARG_SRC] = in;
    r_args[DNNL_ARG_DST] = out;
    exec_ctx_t r_ctx(ctx, std::move(r_args));

    nested_scratchpad_t ns(ctx, key_nested, prim);
    r_ctx.set_scratchpad_grantor(ns.grantor());
    CHECK(prim->execute(r_ctx));

    return status::success;
}

status_t ncsp_convolution_fwd_t::execute_convolution(
        const exec_ctx_t &ctx) const {

    using namespace memory_tracking::names;
    engine_t *engine = ctx.stream()->engine();
    auto scratchpad = ctx.get_scratchpad_grantor();

    // initialize nspc src memory
    auto nspc_src_mem = scratchpad.get_memory_storage(key_conv_ncsp_src);
    memory_t nspc_src(engine, &(pd()->nspc_src_md_), std::move(nspc_src_mem));

    // initialize nspc dst memory
    auto nspc_dst_mem = scratchpad.get_memory_storage(key_conv_ncsp_dst);
    memory_t nspc_dst(engine, &(pd()->nspc_dst_md_), std::move(nspc_dst_mem));

    // reorder src from ncsp to nspc
    CHECK(reorder_activations(ctx, src_reorder_p_, engine,
            ctx.args().at(DNNL_ARG_SRC), {&nspc_src, false}));

    // maybe reorder dst from ncsp to nspc
    if (pd()->dst_pre_reorder_pd_)
        CHECK(reorder_activations(ctx, dst_pre_reorder_p_, engine,
                ctx.args().at(DNNL_ARG_DST), {&nspc_dst, false}));

    // execute nspc convolution
    const auto &args = ctx.args();
    exec_args_t conv_args;
    conv_args[DNNL_ARG_DST] = {&nspc_dst, false};
    conv_args[DNNL_ARG_SRC] = {&nspc_src, true};
    conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
    if (pd()->with_bias()) conv_args[DNNL_ARG_BIAS] = args.at(DNNL_ARG_BIAS);

    exec_ctx_t nspc_ctx(ctx, std::move(conv_args));

    nested_scratchpad_t ns(
            ctx, memory_tracking::names::key_nested, nspc_conv_p_);
    nspc_ctx.set_scratchpad_grantor(ns.grantor());
    CHECK(nspc_conv_p_->execute(nspc_ctx));

    // reorder dst from nspc to ncsp
    CHECK(reorder_activations(ctx, dst_post_reorder_p_, engine,
            {&nspc_dst, false}, ctx.args().at(DNNL_ARG_DST)));

    return status::success;
}

status_t ncsp_convolution_fwd_t::execute_matmul(const exec_ctx_t &ctx) const {
    engine_t *engine = ctx.stream()->engine();

    // must cast away const-ness to use as handles for new memory objects
    void *conv_src = const_cast<void *>(CTX_IN_MEM(const void *, DNNL_ARG_SRC));
    void *conv_wei
            = const_cast<void *>(CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS));
    void *conv_bia
            = const_cast<void *>(CTX_IN_MEM(const void *, DNNL_ARG_BIAS));
    void *conv_dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    // init matmul src, weights, dst mems from conv weights, src, dst handles
    memory_t matmul_src(engine, &(pd()->matmul_src_md_),
            memory_flags_t::use_runtime_ptr, conv_wei);
    memory_t matmul_wei(engine, &(pd()->matmul_wei_md_),
            memory_flags_t::use_runtime_ptr, conv_src);
    memory_t matmul_bia(engine, &(pd()->matmul_bia_md_),
            memory_flags_t::use_runtime_ptr, conv_bia);
    memory_t matmul_dst(engine, &(pd()->matmul_dst_md_),
            memory_flags_t::use_runtime_ptr, conv_dst);

    // execute matmul
    const auto &args = ctx.args();
    exec_args_t matmul_args;
    matmul_args[DNNL_ARG_SRC] = {&matmul_src, true};
    matmul_args[DNNL_ARG_WEIGHTS] = {&matmul_wei, true};
    matmul_args[DNNL_ARG_DST] = {&matmul_dst, false};
    if (pd()->with_bias()) {
        if (pd()->bias_po_)
            matmul_args[DNNL_ARG_SRC_1 | DNNL_ARG_ATTR_MULTIPLE_POST_OP(0)]
                    = {&matmul_bia, true};
        else
            matmul_args[DNNL_ARG_BIAS] = args.at(DNNL_ARG_BIAS);
    }

    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));

    nested_scratchpad_t ns(ctx, memory_tracking::names::key_nested, matmul_p_);
    matmul_ctx.set_scratchpad_grantor(ns.grantor());
    CHECK(matmul_p_->execute(matmul_ctx));

    return status::success;
}

status_t ncsp_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    if (matmul_p_) return execute_matmul(ctx);
    if (nspc_conv_p_) return execute_convolution(ctx);
    return status::runtime_error;
}

status_t ncsp_convolution_bwd_weights_t::pd_t::init(engine_t *engine) {
    VCHECK_CONV(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VCHECK_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VCHECK_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VCHECK_CONV(is_bwd_w(), VERBOSE_BAD_PROPKIND);
    VCHECK_CONV(memory_desc_matches_tag(*src_md(), get_ncsp_tag(ndims())),
            VERBOSE_UNSUPPORTED_TAG);
    VCHECK_CONV(memory_desc_matches_tag(*diff_dst_md(), get_ncsp_tag(ndims())),
            VERBOSE_UNSUPPORTED_TAG);

    if (one_of(data_type::bf16, diff_dst_md_.data_type, src_md_.data_type)
            && !mayiuse(avx512_core_bf16))
        return status::unimplemented;

    CHECK(init_convolution(engine));
    init_name();
    init_scratchpad();

    return status::success;
}

status_t ncsp_convolution_bwd_weights_t::pd_t::init_convolution(
        engine_t *engine) {
    format_tag_t nspc_tag = get_nspc_tag(ndims());
    nspc_src_md_ = *src_md();
    nspc_diff_dst_md_ = *diff_dst_md();
    CHECK(memory_desc_init_by_tag(nspc_src_md_, nspc_tag));
    CHECK(memory_desc_init_by_tag(nspc_diff_dst_md_, nspc_tag));
    const convolution_desc_t *ncsp_conv_d = desc();
    primitive_desc_iface_t *conv_pdi;
    CHECK(dnnl_convolution_backward_weights_primitive_desc_create(&conv_pdi,
            engine, ncsp_conv_d->alg_kind, &nspc_src_md_, diff_weights_md(0),
            diff_weights_md(1), &nspc_diff_dst_md_, ncsp_conv_d->strides,
            ncsp_conv_d->dilates, ncsp_conv_d->padding[0],
            ncsp_conv_d->padding[1], nullptr, attr()));
    nspc_conv_pd_ = conv_pdi->impl();
    diff_weights_md_ = *nspc_conv_pd_->diff_weights_md(0);
    diff_bias_md_ = *nspc_conv_pd_->diff_weights_md(1);
    CHECK(reorder_primitive_desc_create(
            src_reorder_pd_, engine, src_md(), &nspc_src_md_));
    CHECK(reorder_primitive_desc_create(
            dst_reorder_pd_, engine, diff_dst_md(), &nspc_diff_dst_md_));
    return status::success;
}

void ncsp_convolution_bwd_weights_t::pd_t::init_scratchpad() {
    using namespace memory_tracking::names;
    auto scratchpad = scratchpad_registry().registrar();
    const memory_desc_wrapper diff_dst_mdw(diff_dst_md());
    const memory_desc_wrapper src_mdw(src_md());
    scratchpad.book(key_conv_ncsp_diff_dst, diff_dst_mdw.nelems(),
            diff_dst_mdw.data_type_size());
    scratchpad.book(
            key_conv_ncsp_src, src_mdw.nelems(), sizeof(src_mdw.data_type()));
    if (nspc_conv_pd_)
        scratchpad.book(key_nested, nspc_conv_pd_->scratchpad_registry());
    if (src_reorder_pd_)
        scratchpad.book(key_nested, src_reorder_pd_->scratchpad_registry());
    if (dst_reorder_pd_)
        scratchpad.book(key_nested, dst_reorder_pd_->scratchpad_registry());
}

status_t ncsp_convolution_bwd_weights_t::init(engine_t *engine) {
    if (pd()->nspc_conv_pd_)
        CHECK(pd()->nspc_conv_pd_->create_primitive(nspc_conv_p_, engine));
    if (pd()->src_reorder_pd_)
        CHECK(pd()->src_reorder_pd_->create_primitive(src_reorder_p_, engine));
    if (pd()->dst_reorder_pd_)
        CHECK(pd()->dst_reorder_pd_->create_primitive(dst_reorder_p_, engine));

    return status::success;
}

status_t ncsp_convolution_bwd_weights_t::reorder_activations(
        const exec_ctx_t &ctx, const std::shared_ptr<primitive_t> prim,
        engine_t *engine, const memory_arg_t &in,
        const memory_arg_t &out) const {
    using namespace memory_tracking::names;
    exec_args_t r_args;
    r_args[DNNL_ARG_SRC] = in;
    r_args[DNNL_ARG_DST] = out;
    exec_ctx_t r_ctx(ctx, std::move(r_args));

    nested_scratchpad_t ns(ctx, key_nested, prim);
    r_ctx.set_scratchpad_grantor(ns.grantor());
    CHECK(prim->execute(r_ctx));

    return status::success;
}

status_t ncsp_convolution_bwd_weights_t::execute_convolution(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    engine_t *engine = ctx.stream()->engine();
    auto scratchpad = ctx.get_scratchpad_grantor();

    // initialize nspc src memory
    auto nspc_src_mem = scratchpad.get_memory_storage(key_conv_ncsp_src);
    memory_t nspc_src(engine, &(pd()->nspc_src_md_), std::move(nspc_src_mem));

    // initialize nspc dst memory
    auto nspc_diff_dst_mem
            = scratchpad.get_memory_storage(key_conv_ncsp_diff_dst);
    memory_t nspc_diff_dst_m_(
            engine, &(pd()->nspc_diff_dst_md_), std::move(nspc_diff_dst_mem));

    CHECK(reorder_activations(ctx, dst_reorder_p_, engine,
            ctx.args().at(DNNL_ARG_DIFF_DST), {&nspc_diff_dst_m_, false}));
    CHECK(reorder_activations(ctx, src_reorder_p_, engine,
            ctx.args().at(DNNL_ARG_SRC), {&nspc_src, false}));

    const auto &args = ctx.args();
    exec_args_t conv_args;
    conv_args[DNNL_ARG_DIFF_DST] = {&nspc_diff_dst_m_, true};
    conv_args[DNNL_ARG_SRC] = {&nspc_src, true};
    conv_args[DNNL_ARG_DIFF_WEIGHTS] = args.at(DNNL_ARG_DIFF_WEIGHTS);
    if (pd()->with_bias())
        conv_args[DNNL_ARG_DIFF_BIAS] = args.at(DNNL_ARG_DIFF_BIAS);

    exec_ctx_t nspc_ctx(ctx, std::move(conv_args));

    nested_scratchpad_t ns(
            ctx, memory_tracking::names::key_nested, nspc_conv_p_);

    nspc_ctx.set_scratchpad_grantor(ns.grantor());
    CHECK(nspc_conv_p_->execute(nspc_ctx));

    return status::success;
}

status_t ncsp_convolution_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    return execute_convolution(ctx);
}

status_t ncsp_convolution_bwd_data_t::pd_t::init(engine_t *engine) {
    VCHECK_CONV(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VCHECK_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VCHECK_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VCHECK_CONV(is_bwd_d(), VERBOSE_BAD_PROPKIND);
    VCHECK_CONV(memory_desc_matches_tag(*diff_src_md(), get_ncsp_tag(ndims())),
            VERBOSE_UNSUPPORTED_TAG);
    VCHECK_CONV(memory_desc_matches_tag(*diff_dst_md(), get_ncsp_tag(ndims())),
            VERBOSE_UNSUPPORTED_TAG);

    if (one_of(data_type::bf16, diff_dst_md_.data_type, weights_md_.data_type)
            && !mayiuse(avx512_core_bf16))
        return status::unimplemented;

    is_matmul_ = reduce.is_gemm() && attr()->has_default_values();

    if (is_matmul_)
        CHECK(init_matmul(engine));
    else
        CHECK(init_convolution(engine));
    init_scratchpad();
    init_name();

    return status::success;
}

status_t ncsp_convolution_bwd_data_t::pd_t::init_convolution(engine_t *engine) {
    format_tag_t nspc_tag = get_nspc_tag(ndims());
    nspc_diff_src_md_ = *diff_src_md();
    nspc_diff_dst_md_ = *diff_dst_md();
    CHECK(memory_desc_init_by_tag(nspc_diff_src_md_, nspc_tag));
    CHECK(memory_desc_init_by_tag(nspc_diff_dst_md_, nspc_tag));
    const convolution_desc_t *ncsp_conv_d = desc();
    primitive_desc_iface_t *conv_pdi;

    CHECK(dnnl_convolution_backward_data_primitive_desc_create(&conv_pdi,
            engine, ncsp_conv_d->alg_kind, &nspc_diff_src_md_, weights_md(0),
            &nspc_diff_dst_md_, ncsp_conv_d->strides, ncsp_conv_d->dilates,
            ncsp_conv_d->padding[0], ncsp_conv_d->padding[1], nullptr, attr()));
    nspc_conv_pd_ = conv_pdi->impl();
    CHECK(reorder_primitive_desc_create(
            src_reorder_pd_, engine, &nspc_diff_src_md_, diff_src_md()));
    CHECK(reorder_primitive_desc_create(
            dst_reorder_pd_, engine, diff_dst_md(), &nspc_diff_dst_md_));
    weights_md_ = *nspc_conv_pd_->weights_md(0);
    return status::success;
}

status_t ncsp_convolution_bwd_data_t::pd_t::init_matmul(engine_t *engine) {
    CHECK(reduce.reshape_activations(
            &matmul_wei_md_, diff_dst_md(0), true, true));
    // initialize diff weights to plain format.
    CHECK(memory_desc_init_by_strides(weights_md_, weights_md_.ndims,
            weights_md_.dims, weights_md_.data_type, nullptr));
    // reshape weights to matmul format
    memory_desc_t weights_reshaped_md_;
    CHECK(reduce.reshape_weights(&weights_reshaped_md_, &weights_md_, true));
    CHECK(reduce.transpose(matmul_src_md_, weights_reshaped_md_));
    CHECK(reduce.reshape_activations(
            &matmul_dst_md_, diff_src_md(), true, false));
    primitive_attr_t _attr;
    primitive_desc_iface_t *matmul_pdi;
    CHECK(dnnl_matmul_primitive_desc_create(&matmul_pdi, engine,
            &matmul_src_md_, &matmul_wei_md_, nullptr, &matmul_dst_md_,
            &_attr));
    matmul_diff_src_pd_ = matmul_pdi->impl();
    return status::success;
}

void ncsp_convolution_bwd_data_t::pd_t::init_scratchpad() {
    using namespace memory_tracking::names;
    auto scratchpad = scratchpad_registry().registrar();
    if (is_matmul_) {
        if (matmul_diff_src_pd_)
            scratchpad.book(
                    key_nested, matmul_diff_src_pd_->scratchpad_registry());
    } else {
        const memory_desc_wrapper diff_dst_mdw(diff_dst_md());
        const memory_desc_wrapper diff_src_mdw(diff_src_md());
        scratchpad.book(key_conv_ncsp_diff_dst, diff_dst_mdw.nelems(),
                sizeof(diff_dst_mdw.data_type()));
        scratchpad.book(key_conv_ncsp_diff_src, diff_src_mdw.nelems(),
                sizeof(diff_src_mdw.data_type()));
        if (nspc_conv_pd_)
            scratchpad.book(key_nested, nspc_conv_pd_->scratchpad_registry());
        if (src_reorder_pd_)
            scratchpad.book(key_nested, src_reorder_pd_->scratchpad_registry());
        if (dst_reorder_pd_)
            scratchpad.book(key_nested, dst_reorder_pd_->scratchpad_registry());
    }
}

status_t ncsp_convolution_bwd_data_t::init(engine_t *engine) {
    if (pd()->nspc_conv_pd_)
        CHECK(pd()->nspc_conv_pd_->create_primitive(nspc_conv_p_, engine));
    if (pd()->src_reorder_pd_)
        CHECK(pd()->src_reorder_pd_->create_primitive(src_reorder_p_, engine));
    if (pd()->dst_reorder_pd_)
        CHECK(pd()->dst_reorder_pd_->create_primitive(dst_reorder_p_, engine));
    if (pd()->matmul_diff_src_pd_)
        CHECK(pd()->matmul_diff_src_pd_->create_primitive(
                matmul_diff_src_p_, engine));
    return status::success;
}

status_t ncsp_convolution_bwd_data_t::reorder_activations(const exec_ctx_t &ctx,
        const std::shared_ptr<primitive_t> prim, engine_t *engine,
        const memory_arg_t &in, const memory_arg_t &out) const {
    using namespace memory_tracking::names;
    exec_args_t r_args;
    r_args[DNNL_ARG_SRC] = in;
    r_args[DNNL_ARG_DST] = out;
    exec_ctx_t r_ctx(ctx, std::move(r_args));

    nested_scratchpad_t ns(ctx, key_nested, prim);
    r_ctx.set_scratchpad_grantor(ns.grantor());
    CHECK(prim->execute(r_ctx));

    return status::success;
}

status_t ncsp_convolution_bwd_data_t::execute_convolution(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    engine_t *engine = ctx.stream()->engine();
    auto scratchpad = ctx.get_scratchpad_grantor();

    // initialize nspc src memory
    auto nspc_diff_src_mem
            = scratchpad.get_memory_storage(key_conv_ncsp_diff_src);
    memory_t nspc_diff_src_m_(
            engine, &(pd()->nspc_diff_src_md_), std::move(nspc_diff_src_mem));

    // initialize nspc dst memory
    auto nspc_diff_dst_mem
            = scratchpad.get_memory_storage(key_conv_ncsp_diff_dst);
    memory_t nspc_diff_dst_m_(
            engine, &(pd()->nspc_diff_dst_md_), std::move(nspc_diff_dst_mem));

    CHECK(reorder_activations(ctx, dst_reorder_p_, engine,
            ctx.args().at(DNNL_ARG_DIFF_DST), {&nspc_diff_dst_m_, false}));

    const auto &args = ctx.args();
    exec_args_t conv_args;
    conv_args[DNNL_ARG_DIFF_DST] = {&nspc_diff_dst_m_, true};
    conv_args[DNNL_ARG_DIFF_SRC] = {&nspc_diff_src_m_, false};
    conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);

    exec_ctx_t nspc_ctx(ctx, std::move(conv_args));

    nested_scratchpad_t ns(
            ctx, memory_tracking::names::key_nested, nspc_conv_p_);

    nspc_ctx.set_scratchpad_grantor(ns.grantor());
    CHECK(nspc_conv_p_->execute(nspc_ctx));

    CHECK(reorder_activations(ctx, src_reorder_p_, engine,
            {&nspc_diff_src_m_, false}, ctx.args().at(DNNL_ARG_DIFF_SRC)));

    return status::success;
}

status_t ncsp_convolution_bwd_data_t::execute_matmul(
        const exec_ctx_t &ctx) const {
    engine_t *engine = ctx.stream()->engine();
    using namespace memory_tracking::names;
    void *conv_diff_src
            = const_cast<void *>(CTX_IN_MEM(const void *, DNNL_ARG_DIFF_SRC));
    void *conv_wei
            = const_cast<void *>(CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS));
    void *conv_diff_dst = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_DST);

    memory_t matmul_src_m_(engine, &(pd()->matmul_src_md_),
            memory_flags_t::use_runtime_ptr, conv_wei);
    memory_t matmul_wei_m_(engine, &(pd()->matmul_wei_md_),
            memory_flags_t::use_runtime_ptr, conv_diff_dst);
    memory_t matmul_dst_m_(engine, &(pd()->matmul_dst_md_),
            memory_flags_t::use_runtime_ptr, conv_diff_src);

    exec_args_t matmul_src_diff_args;
    matmul_src_diff_args[DNNL_ARG_SRC] = {&matmul_src_m_, true};
    matmul_src_diff_args[DNNL_ARG_WEIGHTS] = {&matmul_wei_m_, true};
    matmul_src_diff_args[DNNL_ARG_DST] = {&matmul_dst_m_, false};

    exec_ctx_t matmul_src_diff_ctx(ctx, std::move(matmul_src_diff_args));

    nested_scratchpad_t matmul_src_diff_ns(
            ctx, memory_tracking::names::key_nested, matmul_diff_src_p_);
    matmul_src_diff_ctx.set_scratchpad_grantor(matmul_src_diff_ns.grantor());
    CHECK(matmul_diff_src_p_->execute(matmul_src_diff_ctx));

    return status::success;
}

status_t ncsp_convolution_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    if (matmul_diff_src_p_)
        return execute_matmul(ctx);
    else
        return execute_convolution(ctx);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

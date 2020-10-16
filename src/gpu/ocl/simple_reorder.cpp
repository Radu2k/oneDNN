/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "gpu/ocl/simple_reorder.hpp"

#include "common/utils.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::memory_tracking::names;

int innermost_block(dnnl_blocking_desc_t blk) {
    int last = blk.inner_nblks - 1;
    return blk.inner_blks[last];
}

int get_stride(const memory_desc_wrapper &md, int dim) {
    return md.md_->format_desc.blocking.strides[dim];
}

int find_stride_1(const memory_desc_wrapper &md) {
    for (int i = 0; i < md.ndims(); i++) {
        if (get_stride(md, i) == 1) { return i; }
    }
    return -1;
}

int innermost_dim_idx(const memory_desc_wrapper &md) {
    int nblks = md.md_->format_desc.blocking.inner_nblks;
    if (nblks != 0) {
        return md.md_->format_desc.blocking.inner_idxs[nblks - 1];
    } else {
        return find_stride_1(md);
    }
}

int innermost_dim_size(const memory_desc_wrapper &md) {
    int nblks = md.md_->format_desc.blocking.inner_nblks;
    if (nblks != 0) {
        return md.md_->format_desc.blocking.inner_blks[nblks - 1];
    } else {
        return md.padded_dims()[md.md_->ndims - 1];
    }
}

bool try_16x16(const memory_desc_wrapper &one, const memory_desc_wrapper &two) {
    using namespace format_tag;
    // TODO: don't rely on tags and make it more generic
    // The real limitations are:
    // dst's last dimension or block == 16
    // dst's penultimate dimension or block % 16 == 0
    // src's last dimension is dst's penultimate dimension
    // the dimension that's last in dst: in src it must be not blocked
    // or blocked with block size % 16 == 0
    if (innermost_dim_size(one) % 16 != 0) { return false; }
    if (innermost_dim_size(two) != 16) { return false; }
    return one.matches_one_of_tag(abcd) && two.matches_one_of_tag(aBcd16b);
}

// Checks if the transpose_16x16 kernel can be used with given tensors.
// Since it has stricter requirements for one tensor and relaxed for the other,
// two attempts to match are performed.
// Returns 0 if no match
// Returns 1 if src is plain and dst is blocked
// Returns 2 if src is blocked and dst is plain
int matches_16x16_layout(
        const memory_desc_wrapper &src, const memory_desc_wrapper &dst) {
    if (try_16x16(src, dst)) {
        return 1;
    } else if (try_16x16(dst, src)) {
        return 2;
    } else {
        return 0;
    }
}

bool dim_is_div_by_16_or_less_than_16(
        const memory_desc_wrapper &src, int dim_index) {
    const auto &padded_dims = src.padded_dims();
    assert(dim_index < src.ndims());
    return (padded_dims[dim_index] % 16 == 0 || padded_dims[dim_index] < 16);
}

status_t simple_reorder_t::pd_t::init_conf(engine_t *engine) {
    using namespace format_tag;

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);

    status_t status = status::success;

    const auto &padded_dims = dst_mdw.padded_dims();
    conf.with_sum_ab = (alpha() != 1.f || beta() != 0.f);
    conf.scale_quant = attr()->output_scales_.mask_ != 0;
    conf.scale_mask = conf.scale_quant ? attr()->output_scales_.mask_ : 0;
    conf.scales_num = conf.scale_quant ? attr()->output_scales_.count_ : 0;
    conf.with_sum_a = conf.with_sum_ab && beta() == 0.f;
    conf.do_reorder
            = conf.scale_quant || conf.with_sum_ab ? true : src_mdw != dst_mdw;
    conf.has_padding = !src_mdw.is_dense() || !dst_mdw.is_dense();
    conf.ndims = src_mdw.ndims();
    conf.nelems = utils::array_product(padded_dims, conf.ndims);

    conf.use_ref_impl = true;
    conf.with_group = false;
    conf.sub_group_size = 1;

    if (conf.nelems == 0) return status::success;

    int last = conf.ndims - 1;
    size_t last_dim = padded_dims[last];

    if (src_mdw.matches_one_of_tag(gOIw8o16i2o, gOIhw8o16i2o, gOIw8i16o2i,
                gOIhw8i16o2i, gOIdhw8i16o2i, gOIw4o8i8o4i, gOIhw4o8i8o4i,
                gOIhw2o8i8o2i, gOIdhw4o8i8o4i, gIOw4i8o8i4o, gIOhw4i8o8i4o,
                gIOdhw4i8o8i4o)
            || dst_mdw.matches_one_of_tag(gOIw8o16i2o, gOIhw8o16i2o,
                    gOIw8i16o2i, gOIhw8i16o2i, gOIdhw8i16o2i, gOIw4o8i8o4i,
                    gOIhw4o8i8o4i, gOIhw2o8i8o2i, gOIdhw4o8i8o4i, gIOw4i8o8i4o,
                    gIOhw4i8o8i4o, gIOdhw4i8o8i4o))
        conf.with_group = true;

    const bool has_padding_or_scale_quant
            = conf.has_padding || conf.scale_quant;

    const bool type_s8_u8 = utils::one_of(src_mdw.data_type(), dnnl_s8, dnnl_u8)
            || utils::one_of(dst_mdw.data_type(), dnnl_s8, dnnl_u8);

    const bool tr16x16
            = !has_padding_or_scale_quant && padded_dims[last] % 16 == 0;
    conf.transpose16x16 = (int)tr16x16 * matches_16x16_layout(src_mdw, dst_mdw);

    conf.nchw = !conf.transpose16x16 && padded_dims[conf.ndims - 1] % 16 == 0
            && dim_is_div_by_16_or_less_than_16(dst_mdw, 1)
            && src_mdw.matches_one_of_tag(nhwc)
            && dst_mdw.matches_one_of_tag(nchw);

    const bool allow_unroll = !has_padding_or_scale_quant && !type_s8_u8
            && !conf.transpose16x16 && !conf.nchw;

    const bool use_unroll_16a16b = allow_unroll
            && (src_mdw.matches_one_of_tag(ABc16a16b, ABc16b16a, ABcd16a16b,
                        ABcd16b16a, ABcde16a16b, ABcde16b16a, BAc16a16b,
                        BAc16b16a, BAcd16a16b, BAcd16b16a, BAcde16b16a)
                    || dst_mdw.matches_one_of_tag(ABc16a16b, ABc16b16a,
                            ABcd16a16b, ABcd16b16a, ABcde16a16b, ABcde16b16a,
                            BAc16a16b, BAc16b16a, BAcd16a16b, BAcd16b16a,
                            BAcde16b16a));

    const bool use_unroll_16b = allow_unroll
            && (src_mdw.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)
                    || dst_mdw.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b));

    const bool use_unroll_16b16c = allow_unroll
            && (src_mdw.matches_one_of_tag(aBCd16b16c, aBCd16c16b, aBCde16b16c,
                        aBCde16c16b, aBCdef16b16c, aBCdef16c16b, aCBd16b16c,
                        aCBd16c16b, aCBde16b16c, aCBde16c16b, aCBdef16c16b)
                    || dst_mdw.matches_one_of_tag(aBCd16b16c, aBCd16c16b,
                            aBCde16b16c, aBCde16c16b, aBCdef16b16c,
                            aBCdef16c16b, aCBd16b16c, aCBd16c16b, aCBde16b16c,
                            aCBde16c16b, aCBdef16c16b));

    conf.plain_xFxE_to_abcdef = src_mdw.matches_one_of_tag(abdfce)
            && dst_mdw.matches_one_of_tag(abcdef)
            && ((padded_dims[conf.ndims - 2] % 16) == 0)
            && dim_is_div_by_16_or_less_than_16(dst_mdw, last);

    conf.plain_to_ABcd4axb = !conf.scale_quant
            && (src_mdw.matches_one_of_tag(abcd)
                    || src_mdw.matches_one_of_tag(acdb))
            && dst_mdw.matches_one_of_tag(ABcd4a2b, ABcd4a4b)
            && src_mdw.is_dense() && dst_mdw.is_dense(true)
            && padded_dims[3] % 16 == 0;

    bool use_unroll = use_unroll_16b || use_unroll_16b16c || use_unroll_16a16b;

    conf.use_dense_vect = !conf.transpose16x16 && !conf.scale_quant
            && !conf.nchw && (conf.nelems % 256 == 0)
            && src_mdw.similar_to(dst_mdw, true, false, 0)
            && !has_padding_or_scale_quant && !use_unroll;

    // This kernel will be used where last dimension is not reordered.
    // It will vectorize that dimension.
    conf.vectorize_last_dim = !conf.transpose16x16 && !conf.use_dense_vect
            && !conf.nchw && !has_padding_or_scale_quant && src_mdw.is_dense()
            && dst_mdw.is_dense() && last_dim % 8 == 0
            && dst_mdw.md_->format_desc.blocking.strides[last] == 1
            && src_mdw.md_->format_desc.blocking.strides[last] == 1
            && conf.ndims <= 6;

    dim_t blocks[6] = {1, 1, 1, 1, 1, 1};
    if (use_unroll_16a16b) {
        blocks[0] = 16;
    } else if (use_unroll_16b) {
        // No blocking.
    } else if (use_unroll_16b16c) {
        conf.with_group = true;
        blocks[2] = 16;
    } else if (conf.plain_xFxE_to_abcdef) {
        blocks[5] = nstl::min(padded_dims[conf.ndims - 1], dnnl_dim_t(16));
    }

    if (conf.use_dense_vect || use_unroll_16a16b || use_unroll_16b
            || use_unroll_16b16c || conf.plain_xFxE_to_abcdef) {
        conf.use_ref_impl = false;
        conf.sub_group_size = 16;
    }

    if (conf.plain_to_ABcd4axb) {
        conf.use_ref_impl = false;

        auto &blk = dst_mdw.blocking_desc();
        int b_block = blk.inner_blks[blk.inner_nblks - 1];
        conf.sub_group_size = (b_block == 2 ? 8 : 16);
        blocks[0] = 4;
        blocks[1] = b_block;
    }

    if (conf.vectorize_last_dim) {
        conf.use_ref_impl = false;
        for (int dim = conf.ndims - 2; dim >= 0; dim--) {
            if (padded_dims[dim] % 4 == 0) { blocks[dim] = 4; }
            if (padded_dims[dim] % 8 == 0) { blocks[dim] = 8; }
            if (padded_dims[dim] % 16 == 0) { blocks[dim] = 16; }
            if (blocks[dim] != 1) { break; }
        }
    }

    if (conf.transpose16x16) {
        conf.use_ref_impl = false;
        conf.sub_group_size = 16;
        auto dm = innermost_dim_idx(
                (conf.transpose16x16 == 1) ? dst_mdw : src_mdw);
        blocks[dm] = 16;
    }

    if (conf.nchw) {
        conf.use_ref_impl = false;
        conf.sub_group_size = 16;
        blocks[1] = nstl::min(padded_dims[1], dnnl_dim_t(16));
    }

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(dst_mdw.md_);
    for (int i = 0; i < 6; ++i) {
        auto dim_str = utils::format("D%d", i);
        if (i < dst_mdw.ndims() && !conf.use_dense_vect) {
            dim_t block = conf.use_ref_impl ? ((i < 2) ? 1 : 0) : blocks[i];
            conf.dispatch.define_dim(dim_str, i, padded_dims[i], block);
        } else if (i == 0) {
            // 1D indexing for dense_vect cases
            conf.dispatch.define_dim(dim_str, 0, conf.nelems, 16);
            conf.dispatch.vectorize_dim("D0", 16);
        } else {
            conf.dispatch.define_dim(dim_str, 1);
        }
    }

    if (use_unroll_16a16b || use_unroll_16b || use_unroll_16b16c) {
        conf.dispatch.vectorize_dim("D1", 16);
    } else if (conf.plain_xFxE_to_abcdef) {
        conf.dispatch.vectorize_dim("D4", conf.sub_group_size);
    } else if (conf.plain_to_ABcd4axb) {
        conf.dispatch.vectorize_dim("D3", conf.sub_group_size);
    } else if (conf.vectorize_last_dim) {
        int vectorization_range = (last_dim % 16 == 0) ? 16 : 8;
        std::string vector_dim = "D" + std::to_string(conf.ndims - 1);
        conf.dispatch.vectorize_dim(vector_dim, vectorization_range);
    } else if (conf.transpose16x16) {
        auto dm = innermost_dim_idx(
                (conf.transpose16x16 == 1) ? src_mdw : dst_mdw);
        auto dim_str = utils::format("D%d", dm);
        conf.dispatch.vectorize_dim(dim_str, 16);
    } else if (conf.nchw) {
        conf.dispatch.vectorize_dim("D3", 16);
    }

    conf.dispatch.generate();

    return status;
}

status_t simple_reorder_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    using namespace format_tag;

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    if (conf.nelems == 0) return status::success;

    kernel_ctx.define_int("NDIMS", conf.ndims);

    if (conf.with_sum_a)
        kernel_ctx.define_int("WITH_SUM_A", 1);
    else if (conf.with_sum_ab)
        kernel_ctx.define_int("WITH_SUM_AB", 1);

    if (conf.scale_quant) {
        kernel_ctx.define_int("SCALE_QUANT", 1);
        kernel_ctx.define_int("SCALE_MASK", conf.scale_mask);
    }
    kernel_ctx.define_int("WITH_GROUP", conf.with_group);

    def_dispatch(kernel_ctx, conf.dispatch);

    kernel_ctx.define_int("REF_REORDER", conf.use_ref_impl);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);

    kernel_ctx.define_int("PAD_FILL_ZERO", conf.has_padding);
    if (conf.use_dense_vect) {
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");
        kernel_ctx.define_int("USE_DENSE_VECT", 1);
    }

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    if (!conf.use_ref_impl) {
        if (src_mdw.matches_one_of_tag(ABc16a16b, ABcd16a16b, ABcde16a16b,
                    BAc16a16b, BAcd16a16b)) {
            kernel_ctx.define_int("SRC_16A16B", 1);
        } else if (src_mdw.matches_one_of_tag(ABc16b16a, ABcd16b16a,
                           ABcde16b16a, BAc16b16a, BAcd16b16a, BAcde16b16a)) {
            kernel_ctx.define_int("SRC_16B16A", 1);
        } else if (src_mdw.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)) {
            kernel_ctx.define_int("SRC_16B", 1);
        } else if (src_mdw.matches_one_of_tag(aBCd16b16c, aBCde16b16c,
                           aBCdef16b16c, aCBd16b16c, aCBde16b16c)) {
            kernel_ctx.define_int("SRC_16B16C", 1);
        } else if (src_mdw.matches_one_of_tag(aBCd16c16b, aBCde16c16b,
                           aBCdef16c16b, aCBd16c16b, aCBde16c16b,
                           aCBdef16c16b)) {
            kernel_ctx.define_int("SRC_16C16B", 1);
        }
    }

    if (src_mdw.matches_one_of_tag(OIw8o16i2o, OIhw8o16i2o, OIdhw8o16i2o,
                gOIw8o16i2o, gOIhw8o16i2o, gOIdhw8o16i2o)) {
        kernel_ctx.define_int("SRC_OIHW8O16I2O", 1);
    } else if (src_mdw.matches_one_of_tag(OIw8i16o2i, OIhw8i16o2i, OIdhw8i16o2i,
                       gOIw8i16o2i, gOIhw8i16o2i, gOIdhw8i16o2i)) {
        kernel_ctx.define_int("SRC_OIHW8I16O2I", 1);
    } else if (src_mdw.matches_one_of_tag(OIw4o8i8o4i, OIhw4o8i8o4i,
                       OIdhw4o8i8o4i, gOIw4o8i8o4i, gOIhw4o8i8o4i,
                       gOIdhw4o8i8o4i)) {
        kernel_ctx.define_int("SRC_OIHW4O8I8O4I", 1);
    } else if (src_mdw.matches_one_of_tag(IOw4i8o8i4o, IOhw4i8o8i4o,
                       IOdhw4i8o8i4o, gIOw4i8o8i4o, gIOhw4i8o8i4o,
                       gIOdhw4i8o8i4o)) {
        kernel_ctx.define_int("SRC_IOHW4I8O8I4O", 1);
    } else if (src_mdw.matches_one_of_tag(OIhw2o8i8o2i, gOIhw2o8i8o2i)) {
        kernel_ctx.define_int("SRC_OIHW2O8I8O2I", 1);
    }

    if (!conf.use_ref_impl) {
        if (dst_mdw.matches_one_of_tag(ABc16a16b, ABcd16a16b, ABcde16a16b,
                    BAc16a16b, BAcd16a16b)) {
            kernel_ctx.define_int("DST_16A16B", 1);
        } else if (dst_mdw.matches_one_of_tag(ABc16b16a, ABcd16b16a,
                           ABcde16b16a, BAc16b16a, BAcd16b16a, BAcde16b16a)) {
            kernel_ctx.define_int("DST_16B16A", 1);
        } else if (dst_mdw.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)) {
            kernel_ctx.define_int("DST_16B", 1);
        } else if (dst_mdw.matches_one_of_tag(aBCd16b16c, aBCde16b16c,
                           aBCdef16b16c, aCBd16b16c, aCBde16b16c)) {
            kernel_ctx.define_int("DST_16B16C", 1);
        } else if (dst_mdw.matches_one_of_tag(aBCd16c16b, aBCde16c16b,
                           aBCdef16c16b, aCBd16c16b, aCBde16c16b,
                           aCBdef16c16b)) {
            kernel_ctx.define_int("DST_16C16B", 1);
        }
    }

    if (dst_mdw.matches_one_of_tag(OIw8o16i2o, OIhw8o16i2o, OIdhw8o16i2o,
                gOIw8o16i2o, gOIhw8o16i2o, gOIdhw8o16i2o)) {
        kernel_ctx.define_int("DST_OIHW8O16I2O", 1);
    } else if (dst_mdw.matches_one_of_tag(OIw8i16o2i, OIhw8i16o2i, OIdhw8i16o2i,
                       gOIw8i16o2i, gOIhw8i16o2i, gOIdhw8i16o2i)) {
        kernel_ctx.define_int("DST_OIHW8I16O2I", 1);
    } else if (dst_mdw.matches_one_of_tag(OIw4o8i8o4i, OIhw4o8i8o4i,
                       OIdhw4o8i8o4i, gOIw4o8i8o4i, gOIhw4o8i8o4i,
                       gOIdhw4o8i8o4i)) {
        kernel_ctx.define_int("DST_OIHW4O8I8O4I", 1);
    } else if (dst_mdw.matches_one_of_tag(IOw4i8o8i4o, IOhw4i8o8i4o,
                       IOdhw4i8o8i4o, gIOw4i8o8i4o, gIOhw4i8o8i4o,
                       gIOdhw4i8o8i4o)) {
        kernel_ctx.define_int("DST_IOHW4I8O8I4O", 1);
    } else if (dst_mdw.matches_one_of_tag(OIhw2o8i8o2i, gOIhw2o8i8o2i)) {
        kernel_ctx.define_int("DST_OIHW2O8I8O2I", 1);
    }

    if (conf.plain_xFxE_to_abcdef)
        kernel_ctx.define_int("PLAIN_xFxE_TO_ABCDEF", 1);

    if (conf.plain_to_ABcd4axb) kernel_ctx.define_int("PLAIN_TO_ABCD4AXB", 1);

    if (conf.vectorize_last_dim) {
        kernel_ctx.define_int("VECTORIZE_LAST_DIM", 1);
    }

    if (conf.transpose16x16) {
        kernel_ctx.define_int("TRANSPOSE_16X16", 1);
        if (conf.transpose16x16 == 1) {
            kernel_ctx.define_int("PLAIN_TO_BLOCK", 1);
        }
    }

    if (conf.nchw) { kernel_ctx.define_int("REORDER_NCHW", 1); }

    kernel_ctx.print_options();
    return status::success;
}

void simple_reorder_t::pd_t::init_scratchpad() {
    if (conf.scales_num > 0) {
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_reorder_scales,
                conf.scales_num, sizeof(float), OCL_BUFFER_ALIGNMENT);
    }
}

status_t simple_reorder_t::execute(const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_FROM);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_TO);

    const auto &conf = pd()->conf;
    if (conf.nelems == 0) return status::success;

    float alpha = pd()->alpha();
    float beta = pd()->beta();

    status_t status = status::success;

    std::unique_ptr<memory_storage_t> scales;
    if (conf.scale_quant) {
        scales = ctx.get_scratchpad_grantor().get_memory_storage(
                key_reorder_scales);

        void *tmp_ptr = nullptr;
        status = scales->map_data(&tmp_ptr, ctx.stream(),
                sizeof(float) * pd()->attr()->output_scales_.count_);
        if (status != status::success) return status;
        utils::array_copy((float *)tmp_ptr,
                pd()->attr()->output_scales_.scales_,
                pd()->attr()->output_scales_.count_);
        status = scales->unmap_data(tmp_ptr, ctx.stream());
        if (status != status::success) return status;
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);
    arg_list.set(2, alpha);
    arg_list.set(3, beta);
    arg_list.set(4, scales ? *scales : memory_storage_t::empty_storage());

    auto nd_range = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

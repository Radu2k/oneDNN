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

#ifndef REF_CONVOLUTION_HPP
#define REF_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "ocl/jit_primitive_conf.hpp"
#include "ocl/ocl_convolution_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"
#include <assert.h>

extern const char *ref_conv_fwd_data_u8s8s32x_kernel;
extern const char *ref_conv_bwd_data_u8s8s32x_kernel;

namespace mkldnn {
namespace impl {
namespace ocl {

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
        impl::data_type_t dst_type = src_type,
        impl::data_type_t acc_type = dst_type>
struct ref_convolution_fwd_t : public primitive_t {
    struct pd_t : public ocl_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : ocl_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:ncsp:any", ref_convolution_fwd_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;
            assert(this->engine()->kind() == engine_kind::gpu);

            bool ok = true && set_default_formats()
                    && utils::one_of(desc()->prop_kind, forward_training,
                               forward_inference)
                    && desc()->alg_kind == alg_kind::convolution_direct
                    && desc()->src_desc.data_type == src_type
                    && desc()->weights_desc.data_type == wei_type
                    && desc()->accum_data_type == acc_type
                    && desc()->dst_desc.data_type == dst_type;
            if (!ok)
                return status::unimplemented;

            const auto &p = attr()->post_ops_;
            with_sum = p.find(primitive_kind::sum) != -1;
            const int sum_idx = p.find(primitive_kind::sum);
            sum_scale = (sum_idx != -1) ? p.entry_[sum_idx].sum.scale : 1.0;

            const int eltwise_ind = p.find(primitive_kind::eltwise);
            bool with_eltwise = eltwise_ind != -1;
            with_relu = false;
            negative_slope = .0f;
            if (with_eltwise) {
                auto eltwise = p.entry_[eltwise_ind].eltwise;
                with_relu = eltwise.alg == alg_kind::eltwise_relu
                        && eltwise.alpha != 1.0f;
                if (with_relu)
                    negative_slope = eltwise.alpha;
            }

            with_sum_relu = (p.len_ == 2) 
                && with_relu && with_sum 
                && sum_idx == 0 && eltwise_ind == 1;

            gws[0] = OH() * OW() * OD();
            gws[1] = OC() / G();
            gws[2] = MB() * G();
            lws[0] = lws[1] = lws[2] = 1;
            return status::success;
        }

        float sum_scale, negative_slope;
        bool with_sum, with_relu, with_sum_relu;
        size_t gws[3];
        size_t lws[3];

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    ref_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    ~ref_convolution_fwd_t() = default;

    status_t init() override {
        auto jit = ocl_jit_t(ref_conv_fwd_data_u8s8s32x_kernel);

        jit.set_data_type(dst_type);

        const memory_desc_wrapper src_d(pd()->src_md());
        const memory_desc_wrapper dst_d(pd()->dst_md());
        const int ndims = src_d.ndims();

        jit.define_int("NDIMS", ndims);
        jit.define_int("G", pd()->G());
        jit.define_int("MB", pd()->MB());
        jit.define_int("IC", pd()->IC());
        jit.define_int("ID", pd()->ID());
        jit.define_int("IH", pd()->IH());
        jit.define_int("IW", pd()->IW());
        jit.define_int("OC", pd()->OC());
        jit.define_int("OD", pd()->OD());
        jit.define_int("OH", pd()->OH());
        jit.define_int("OW", pd()->OW());
        jit.define_int("KD", pd()->KD());
        jit.define_int("KH", pd()->KH());
        jit.define_int("KW", pd()->KW());
        jit.define_int("SD", pd()->KSD());
        jit.define_int("SH", pd()->KSH());
        jit.define_int("SW", pd()->KSW());
        jit.define_int("KDD", pd()->KDD());
        jit.define_int("KDH", pd()->KDH());
        jit.define_int("KDW", pd()->KDW());
        jit.define_int("PD", pd()->padFront());
        jit.define_int("PH", pd()->padT());
        jit.define_int("PW", pd()->padL());
        jit.define_int("WITH_BIAS", pd()->with_bias());
        jit.define_int("WITH_RELU", pd()->with_relu);
        jit.define_int("WITH_SUM", pd()->with_sum);
        jit.define_int("WITH_SUM_RELU", pd()->with_sum_relu);
        jit.define_int("SUM_SCALE", pd()->sum_scale == 1.0);
        jit.define_int("LWS_0", 1);
        jit.define_int("LWS_1", 1);
        jit.define_int("LWS_2", 1);
        jit.define_int("GWS_0", pd()->gws[0]);
        jit.define_int("GWS_1", pd()->gws[1]);
        jit.define_int("GWS_2", pd()->gws[2]);

        jit_offsets jit_off;
        set_offsets(src_d, jit_off.src_off);
        set_offsets(dst_d, jit_off.dst_off);
        def_offsets(jit_off.src_off, jit, "SRC", ndims);
        def_offsets(jit_off.dst_off, jit, "DST", ndims);

        status_t status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("ref_conv_fwd_kernel");
        if (!kernel_)
            return status::runtime_error;

        return status::success;
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    ocl_kernel_t kernel_;
};

template <impl::data_type_t diff_dst_type>
struct ref_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public ocl_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : ocl_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:ncsp:any", ref_convolution_bwd_data_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;
            assert(this->engine()->kind() == engine_kind::gpu);

            bool ok = true && set_default_formats()
                    && IMPLICATION(utils::one_of(diff_dst_type, u8, s8),
                               expect_data_types(
                                       u8, s8, f32, diff_dst_type, s32))
                    && desc()->prop_kind == prop_kind::backward_data
                    && desc()->alg_kind == alg_kind::convolution_direct;
            if (!ok)
                return status::unimplemented;

            const auto &p = attr()->post_ops_;
            with_sum = p.find(primitive_kind::sum) != -1;
            with_relu = p.find(primitive_kind::eltwise) != -1;
            const int sum_idx = p.find(primitive_kind::sum);
            sum_scale = (sum_idx != -1) ? p.entry_[sum_idx].sum.scale : 1.0;
            with_sum_relu = 0;
            if (p.len_ == 2) {
                with_sum_relu = p.entry_[sum_idx].is_sum(sum_scale == 1.0)
                        && p.entry_[1].is_relu();
            }

            negative_slope = .0f;

            gws[0] = IH() * IW() * ID();
            gws[1] = IC() / G();
            gws[2] = MB() * G();
            lws[0] = lws[1] = lws[2] = 1;
            return status::success;
        }

        float sum_scale, negative_slope;
        bool with_sum, with_relu, with_sum_relu;
        size_t gws[3];
        size_t lws[3];

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        bool support_bias() const override { return true; }
    };

    ref_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    ~ref_convolution_bwd_data_t() = default;

    status_t init() override {
        auto jit = ocl_jit_t(ref_conv_bwd_data_u8s8s32x_kernel);

        jit.set_data_type(diff_dst_type);

        const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
        const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
        const int ndims = diff_src_d.ndims();

        jit.define_int("NDIMS", ndims);
        jit.define_int("G", pd()->G());
        jit.define_int("MB", pd()->MB());
        jit.define_int("IC", pd()->IC());
        jit.define_int("ID", pd()->ID());
        jit.define_int("IH", pd()->IH());
        jit.define_int("IW", pd()->IW());
        jit.define_int("OC", pd()->OC());
        jit.define_int("OD", pd()->OD());
        jit.define_int("OH", pd()->OH());
        jit.define_int("OW", pd()->OW());
        jit.define_int("KD", pd()->KD());
        jit.define_int("KH", pd()->KH());
        jit.define_int("KW", pd()->KW());
        jit.define_int("SD", pd()->KSD());
        jit.define_int("SH", pd()->KSH());
        jit.define_int("SW", pd()->KSW());
        jit.define_int("KDD", pd()->KDD());
        jit.define_int("KDH", pd()->KDH());
        jit.define_int("KDW", pd()->KDW());
        jit.define_int("PD", pd()->padFront());
        jit.define_int("PH", pd()->padT());
        jit.define_int("PW", pd()->padL());
        jit.define_int("WITH_BIAS", pd()->with_bias());
        jit.define_int("WITH_RELU", pd()->with_relu);
        jit.define_int("WITH_SUM", pd()->with_sum);
        jit.define_int("WITH_SUM_RELU", pd()->with_sum_relu);
        jit.define_int("SUM_SCALE", pd()->sum_scale == 1.0);
        jit.define_int("LWS_0", 1);
        jit.define_int("LWS_1", 1);
        jit.define_int("LWS_2", 1);
        jit.define_int("GWS_0", pd()->gws[0]);
        jit.define_int("GWS_1", pd()->gws[1]);
        jit.define_int("GWS_2", pd()->gws[2]);

        jit_offsets jit_off;
        set_offsets(diff_src_d, jit_off.src_off);
        set_offsets(diff_dst_d, jit_off.dst_off);
        def_offsets(jit_off.src_off, jit, "SRC", ndims);
        def_offsets(jit_off.dst_off, jit, "DST", ndims);

        status_t status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("ref_conv_bwd_data_kernel");
        if (!kernel_)
            return status::runtime_error;

        return status::success;
    }

    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    ocl_kernel_t kernel_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn
#endif

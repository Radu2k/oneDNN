/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_PATTERNS_DATA_TYPE_CHECK_PASS_HPP
#define GRAPH_BACKEND_DNNL_PATTERNS_DATA_TYPE_CHECK_PASS_HPP

#include "graph/backend/dnnl/kernels/quantize.hpp"
#include "graph/backend/dnnl/patterns/pattern_matcher_pass.hpp"
#include "graph/backend/dnnl/platform.hpp"
#include "graph/backend/fake/pattern_utils.hpp"

#include "graph/backend/dnnl/patterns/utils.hpp"
#include "graph/utils/pm/nested_matcher.hpp"
#include "graph/utils/pm/pass_base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace {

inline platform::dir_t get_op_dir(const std::shared_ptr<op_t> &aop) {
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::impl::graph::dnnl_impl::platform;

    const auto &op_kind = aop->get_kind();
    const auto &num_inputs = aop->num_inputs();
    const auto &num_outputs = aop->num_outputs();

    dir_t dir = dir_t::DIR_UNDEF;
    switch (op_kind) {
        // BatchNorm
        case BatchNormForwardTraining: dir = dir_t::FWD_D; break;
        case BatchNormInference: dir = dir_t::FWD_I; break;
        case BatchNormTrainingBackward:
            dir = num_outputs == 1 ? dir_t::BWD_D : dir_t::BWD_DW;
            break;
        // Convolution
        case Convolution:
            dir = num_inputs > 2 ? dir_t::FWD_B : dir_t::FWD_I;
            break;
        case ConvolutionBackwardData: dir = dir_t::BWD_D; break;
        case ConvolutionBackwardWeights: dir = dir_t::BWD_W; break;
        // ConvTranspose
        case ConvTranspose:
            dir = num_inputs > 2 ? dir_t::FWD_B : dir_t::FWD_I;
            break;
        case ConvTransposeBackwardData: dir = dir_t::BWD_D; break;
        case ConvTransposeBackwardWeights: dir = dir_t::BWD_W; break;
        // Eltwise
        case Abs:
        case Clamp:
        case Elu:
        case Exp:
        case GELU:
        case HardSigmoid:
        case HardSwish:
        case LeakyReLU:
        case Log:
        case Mish:
        case Pow:
        case Reciprocal:
        case ReLU:
        case Round:
        case Sigmoid:
        case SoftPlus:
        case Sqrt:
        case Square:
        case SquaredDifference:
        case Tanh: dir = dir_t::FWD_I; break;
        case AbsBackward:
        case ClampBackward:
        case EluBackward:
        case GELUBackward:
        case HardSigmoidBackward:
        case HardSwishBackward:
        case MishBackward:
        case ReLUBackward:
        case SigmoidBackward:
        case SoftPlusBackward:
        case SqrtBackward:
        case TanhBackward: dir = dir_t::BWD_D; break;
        // LayerNorm
        case LayerNorm: dir = dir_t::FWD_I; break;
        case LayerNormBackward: dir = dir_t::BWD_DW; break;
        // Pool
        case MaxPool:
        case AvgPool: dir = dir_t::FWD_I; break;
        case MaxPoolBackward:
        case AvgPoolBackward: dir = dir_t::BWD_D; break;
        // PReLU
        case PReLU: dir = dir_t::FWD_I; break;
        case PReLUBackward: dir = dir_t::BWD_DW; break;
        // Resampling
        case Interpolate: dir = dir_t::FWD_I; break;
        case InterpolateBackward: dir = dir_t::BWD_D; break;
        // Softmax
        case SoftMax:
        case LogSoftmax: dir = dir_t::FWD_I; break;
        case SoftMaxBackward:
        case LogSoftmaxBackward: dir = dir_t::BWD_D; break;
        // Other ops lack of propagation kind, which are always considered as
        // forward, including Binary, Concat, Matmul, Reduction and Reorder.
        default: dir = dir_t::FWD_I; break;
    }

    return dir;
}

inline bool is_reorder_type(op_kind_t op_kind) {
    using namespace dnnl::impl::graph::op_kind;
    static const std::unordered_set<int> reorder_ops {Reorder, Quantize,
            Dequantize, DynamicDequantize, DynamicQuantize, TypeCast};

    return (reorder_ops.find(op_kind) != reorder_ops.end());
}

} // namespace

/*!
 * \brief dtype_check_pass_t generates a pass for checking unimplemented data 
 *        type.
 */
class dtype_check_pass_t : public graph::pass::pass_base_t {
public:
    explicit dtype_check_pass_t(std::string pbackend, std::string pname,
            std::vector<data_type_t> dtypes)
        : graph::pass::pass_base_t(std::move(pbackend), std::move(pname))
        , dt_to_check_(std::move(dtypes)) {
        // data type check passes should be executed first, hence should
        // have the highest priority.
        set_priority(50.f);
    }

    // the criteria of pass execution
    impl::status_t run(graph_t &agraph) override {
        using namespace dnnl::impl::graph::dnnl_impl::platform;

        // check if current pattern pass can be run on current graph
        engine_kind_t graph_engine_kind = agraph.get_engine_kind();
        if (get_engine_kind() != engine_kind::any_engine
                && get_engine_kind() != graph_engine_kind)
            return impl::status::success;

        std::unordered_map<int, std::vector<data_type_t>> unsupported_dt;
        for (const std::shared_ptr<op_t> &aop : agraph.get_ops()) {
            dir_t dir = get_op_dir(aop);
            if (unsupported_dt.find(dir) == unsupported_dt.end()) {
                unsupported_dt.emplace(dir, std::vector<data_type_t> {});
                for (const auto &dt : dt_to_check_) {
                    bool has_dtype_support = platform::get_dtype_support_status(
                            graph_engine_kind, dt, dir);
                    if (!has_dtype_support)
                        unsupported_dt.at(dir).emplace_back(dt);
                }
            }
        }

        std::vector<op_t *> matched_op_list;
        std::vector<std::vector<op_t *>> reorder_fusion_list;

        // NOTE(zhitao): Currenrly there is no special handling for patterns
        // which owns unsupported data type internally for older platforms
        // but of which the corresponding compiled partitions can be executed,
        // e.g. int8-bf16 patterns such as dequant->tc->matmul->tc->quant.

        for (const std::shared_ptr<op_t> &aop : agraph.get_ops()) {

            bool meet_unsupported_dt {false};
            bool meet_reorder {false};

            dir_t dir = get_op_dir(aop);
            const auto &op_kind = aop->get_kind();

            VCHECK_PATTERN_UTILS(
                    unsupported_dt.find(dir) != unsupported_dt.end(),
                    impl::status::unimplemented, "unsupported dir %d ", dir);
            const auto &dt_with_dir = unsupported_dt.at(dir);

            for (size_t i = 0; i < aop->num_inputs(); ++i) {
                const logical_tensor_t &iport
                        = aop->get_input_value(i)->get_logical_tensor();
                if (std::any_of(dt_with_dir.begin(), dt_with_dir.end(),
                            [&iport](data_type_t dt) {
                                return dt == iport.data_type;
                            })) {
                    if (is_reorder_type(op_kind)) {
                        meet_reorder = true;
                        break;
                    }
                    meet_unsupported_dt = true;
                    break;
                }
            }

            if (!meet_reorder && !meet_unsupported_dt) {
                for (size_t i = 0; i < aop->num_outputs(); ++i) {
                    const logical_tensor_t &oport
                            = aop->get_output_value(i)->get_logical_tensor();
                    if (std::any_of(dt_with_dir.begin(), dt_with_dir.end(),
                                [&oport](data_type_t dt) {
                                    return dt == oport.data_type;
                                })) {
                        if (is_reorder_type(op_kind)) {
                            meet_reorder = true;
                            break;
                        }
                        meet_unsupported_dt = true;
                        break;
                    }
                }
            }

            if (meet_reorder) {
                std::vector<op_t *> candidate_fusion(1, aop.get());
                reorder_fusion_list.emplace_back(candidate_fusion);
            }

            if (meet_unsupported_dt) matched_op_list.emplace_back(aop.get());
        }

        // For quantization patterns, if dnnl backend does not support the
        // fused op, the graph will be fused into separate single op fusions.
        if (!reorder_fusion_list.empty()) {
            pattern_utils_t dnnl_pu;
            const auto quantize_kernel_creater = []() -> kernel_ptr {
                return std::make_shared<
                        dnnl::impl::graph::dnnl_impl::quantize_dequantize_t>();
            };
            dnnl_pu.init_partition(agraph, reorder_fusion_list,
                    quantize_kernel_creater,
                    dnnl::impl::graph::partition_kind_t::misc_post_ops);
        }

        if (!matched_op_list.empty()) {
            dnnl::impl::graph::fake_impl::pattern_utils_t fake_pu;
            fake_pu.fuse(agraph, matched_op_list);
        }

        return impl::status::success;
    }

private:
    std::vector<data_type_t> dt_to_check_;
};

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

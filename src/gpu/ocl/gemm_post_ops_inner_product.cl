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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

__kernel void gemm_post_ops_inner_product(__global SRC_DATA_T *src,
        __global BIAS_DATA_T *bias, __global DST_DATA_T *dst,
        float eltwise_alpha, float eltwise_beta, float eltwise_scale,
        float sum_scale, __global SPAD_DATA_T *scratchpad,
        global float *scales) {
    const size_t mb = get_global_id(0) / OC;
    const size_t oc = get_global_id(0) % OC;

    const size_t data_idx = mb * OC + oc;
#if USE_TEMP_DST == 1
    ACC_DATA_T acc = TO_DEF_ACC_DATA_T(scratchpad[data_idx]);
#else
    ACC_DATA_T acc = TO_DEF_ACC_DATA_T(src[data_idx]);
#endif

#if WITH_BIAS == 1
    acc += TO_DEF_ACC_DATA_T(bias[oc]);
#endif

#if WITH_SCALES
#if SCALES_COMMON
    const float scale = scales[0];
#elif SCALES_PER_OC
    const float scale = scales[oc];
#else
#error "Unsupported scale type"
#endif
    acc *= scale;
#endif

    // Apply postops
#if WITH_SUM
    acc += sum_scale * TO_DEF_ACC_DATA_T(dst[data_idx]);
#endif

#if WITH_ELTWISE
    acc = fwd_eltwise(acc, eltwise_alpha, eltwise_beta, eltwise_scale);
#endif

    dst[data_idx] = TO_DST(acc);
}

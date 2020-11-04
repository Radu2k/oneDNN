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

#include "gpu/ocl/ocl_types.h"

#if defined(IS_MAX)
#define INIT_ACC -INFINITY
#elif defined(IS_MIN)
#define INIT_ACC INFINITY
#elif defined(IS_MUL)
#define INIT_ACC 1.0f
#else
#define INIT_ACC 0.0f
#endif

#if defined(IS_MAX)
#define ACCUMULATE(x, y) fmax(x, y)
#elif defined(IS_MIN)
#define ACCUMULATE(x, y) fmin(x, y)
#elif defined(IS_MEAN) || defined(IS_SUM)
#define ACCUMULATE(x, y) (x + y)
#elif defined(IS_MUL)
#define ACCUMULATE(x, y) (x * y)
#else
#define ACCUMULATE(x, y) (x + pow(fabs(y), POWER))
#endif

// We want to use some acc algorithms (like pow) only once
// for a given element
#if defined(IS_MAX) || defined(IS_MIN) || defined(IS_MUL)
#define ACCUMULATE_AGAIN(x, y) ACCUMULATE(x, y)
#else
#define ACCUMULATE_AGAIN(x, y) (x + y)
#endif

#if defined(IS_MEAN)
#define FINALIZE(x) (x / REDUCTION_SIZE)
#elif defined(IS_LP_MAX)
#define FINALIZE(x) rootn(fmax(x, EPS), POWER)
#elif defined(IS_LP_SUM)
#define FINALIZE(x) rootn(x + EPS, POWER)
#elif defined(IS_P_MAX)
#define FINALIZE(x) fmax(x, EPS)
#elif defined(IS_P_SUM)
#define FINALIZE(x) (x + EPS)
#else
#define FINALIZE(x) (x)
#endif

// Currently reduction works only for cases NxCxHxWxD -> NxCx1

// In the initial phase each work unit reduces HWD_BLOCK of HxWxD input.
// The result has shape NxCxFINAL_HWD_DIM, where FINAL_HWD_DIM is HxWxD / HWD_BLOCK
NAMED_KERNEL_ATTR(INITIAL)
__kernel void gen12hp_initial_reduce(
        __global SRC_DATA_T *src, __global float *dst) {
    const int n = GWS_GET_INITIAL_IN();
    const int c = GWS_GET_INITIAL_IC();
    const int hwd = GWS_GET_INITIAL_HWD_DIM();
    const int hwd_block_idx = hwd / HWD_BLOCK;

    const int c_block_idx = c / 16;
    const int n_offset = n * IC * INITIAL_HWD_DIM;
    const int c_offset = c_block_idx * INITIAL_HWD_DIM * 16;
    const int hwd_offset = hwd_block_idx * HWD_BLOCK * 16;
    const int src_offset = n_offset + c_offset + hwd_offset;

    src += src_offset;

    VECT_FLOAT_T vector_acc = INIT_ACC;
    for (int hwd_id = 0; hwd_id < HWD_BLOCK; hwd_id += VECT_DT_N) {
        VECT_FLOAT_T data = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                (const __global BLOCK_DATA_T *)&src[hwd_id * 16])));
        vector_acc = ACCUMULATE(vector_acc, data);
    }
    float acc = INIT_ACC;
    for (int i = 0; i < 8; i++) {
        acc = ACCUMULATE_AGAIN(acc, vector_acc[i]);
    }

    // N, HWD, C
    const int n_dst_offset = n * IC * FINAL_HWD_DIM;
    const int c_dst_offset = c_block_idx * FINAL_HWD_DIM * 16;
    const int dst_off = n_dst_offset + c_dst_offset + hwd_block_idx * 16
            + get_sub_group_local_id();
    dst[dst_off] = acc;
}

// In the final phase each WI reduces FINAL_HWD_DIM elements into 1.
// The result has shape NxCx1
NAMED_KERNEL_ATTR(FINAL)
__kernel void gen12hp_final_reduce(
        __global float *src, __global DST_DATA_T *dst) {
    const int n = GWS_GET_FINAL_IN();
    const int c = GWS_GET_FINAL_IC();

    const int c_block_idx = c / 16;
    const int c_inside_block_offset = c % 16;
    const int n_offset = n * IC * FINAL_HWD_DIM;
    const int c_offset = c_block_idx * FINAL_HWD_DIM * 16;

    // N, HWD, C
    const int offset = n_offset + c_offset + c_inside_block_offset;
    src += offset;

    float acc = INIT_ACC;
    for (int hwd_id = 0; hwd_id < FINAL_HWD_DIM; hwd_id++) {
        const float data = src[hwd_id * 16];
        acc = ACCUMULATE_AGAIN(acc, data);
    }

    const int dst_offset = n * IC + c;
    dst[dst_offset] = TO_DST(FINALIZE(acc));
}
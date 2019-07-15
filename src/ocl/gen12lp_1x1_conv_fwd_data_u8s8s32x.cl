/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* licensed under the apache license, version 2.0 (the "license");
* you may not use this file except in compliance with the license.
* you may obtain a copy of the license at
*
*     http://www.apache.org/licenses/license-2.0
*
* unless required by applicable law or agreed to in writing, software
* distributed under the license is distributed on an "as is" basis,
* without warranties or conditions of any kind, either express or implied.
* see the license for the specific language governing permissions and
* limitations under the license.
*******************************************************************************/

#include "ocl/ocl_math_utils.h"

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));


#define CHANNEL_OFFSET 1
#define MB_OFFSET IC_BLOCK
#define PIXEL_WIDTH_OFFSET (MB_OFFSET * MB_BLOCK)
#define PIXEL_HEIGHT_OFFSET (PIXEL_WIDTH_OFFSET * IW)
#define CHANNEL_BLOCK_OFFSET (PIXEL_HEIGHT_OFFSET * IH) // For NChw

// Weights offsets
#define WEIGHTS_WIDTH_OFFSET (4 * 8 * 8 * 4)
#define WEIGHTS_HEIGHT_OFFSET (WEIGHTS_WIDTH_OFFSET * 1)
#define KERNEL_BLOCK_OFFSET (WEIGHTS_HEIGHT_OFFSET * 1)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
gen12lp_1x1_conv_fwd_kernel(const __global uchar *src, const __global char *wei,
        const __global float *bias, __global uchar *dst,
        float relu_negative_slope, float sum_scale, float scales) {

    // Groups:
    const uint oc_group_id = get_group_id(0);
    const uint sp_group_id = get_group_id(1);
    const uint mb_group_id = get_group_id(2);

    // SIMD
    const uint sg_local_id = get_sub_group_local_id();
    const uint sg_id = get_sub_group_id();

    // Spatial
    const uint sp = get_global_id(1);
    const uint ow = sp % OW;
    const uint oh = sp / OW;
    const uint iw = ow;
    const uint ih = oh;

    // Source (At ic = 0)
    src += (mb_group_id % 2) * MB_BLOCK / 2 * MB_OFFSET; // MB block offset
    src += (mb_group_id / 2) * CHANNEL_BLOCK_OFFSET * IC_NCHUNK; // MB offset
    src += oh * PIXEL_HEIGHT_OFFSET; // height offset
    src += ow * PIXEL_WIDTH_OFFSET; // width offset
    
    // Destination
    dst += (mb_group_id % 2) * MB_BLOCK / 2 * MB_OFFSET; // MB block offset
    dst += (mb_group_id / 2) * CHANNEL_BLOCK_OFFSET * OC_NCHUNK; // MB offset
    dst += CHANNEL_BLOCK_OFFSET * oc_group_id; //OC offset
    dst += oh * PIXEL_HEIGHT_OFFSET;
    dst += ow * PIXEL_WIDTH_OFFSET;

    // Weights
    wei += oc_group_id * KERNEL_BLOCK_OFFSET * IC_NCHUNK;

    // Output accumulators:
    // 8 MB (0-7) x 4 Kernels  (32 8bit ints)
    int8 C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    // 8 MB (8-15) x 4 Kernels  (32 8bit ints)
    int8 C10 = 0, C11 = 0, C12 = 0, C13 = 0;
    
    for(uint ic_block_id = 0; ic_block_id < IC_NCHUNK; ++ic_block_id)
    {
        uint8 S0, S1;
        int8 W0, W1, W2, W3;

        BLOCK_READ_SRC(S0, 0 * IC_BLOCK);
        BLOCK_READ_SRC(S1, 8 * IC_BLOCK);

        BLOCK_READ_WHT(W0, 0);
        BLOCK_READ_WHT(W1, 8 * IC_BLOCK);
        BLOCK_READ_WHT(W2, 16 * IC_BLOCK);
        BLOCK_READ_WHT(W3, 24 * IC_BLOCK);

        C00 = mmad8x8(S0, W0, C00);
        C01 = mmad8x8(S0, W1, C01);
        C02 = mmad8x8(S0, W2, C02);
        C03 = mmad8x8(S0, W3, C03);
        C10 = mmad8x8(S1, W0, C10);
        C11 = mmad8x8(S1, W1, C11);
        C12 = mmad8x8(S1, W2, C12);
        C13 = mmad8x8(S1, W3, C13);

        src += CHANNEL_BLOCK_OFFSET;
        wei += KERNEL_BLOCK_OFFSET;
    }


#if WITH_BIAS
#if WITH_SUM_ELTWISE
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)DST, sum_scale, BIA);            \
    TMP = fma((float)ACC, SCALE, TMP);                \
    RES = convert_uchar_sat(TMP);
#else // WITH_SUM_ELTWISE
#if WITH_RELU
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)ACC, SCALE, BIA);                \
    RES = convert_uchar_sat(TMP);
#else // WITH_RELU
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)ACC, SCALE, BIA);                \
    RES = convert_uchar_sat(TMP);
#endif // WITH_RELU 
#endif // WITH_SUM_ELTWISE
#else // WITH_BIAS
#if WITH_SUM_ELTWISE
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE)     \
    TMP = fma((float)ACC, SCALE, (float)DST * sum_scale); \
    RES = convert_uchar_sat(TMP);
#else // WITH_SUM_ELTWISE
#if WITH_RELU
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = (float)ACC * SCALE;                         \
    RES = convert_uchar_sat(TMP);
#else // WITH_RELU
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = (float)ACC * SCALE;                         \
    RES = convert_uchar_sat(TMP);
#endif // WITH_RELU
#endif // WITH_SUM_ELTWISE
#endif // WITH_BIAS

#if WITH_SUM_ELTWISE || WITH_SUM
#define PACK(idx)                                             \
    D00 = as_uchar4(D0[idx]);                                 \
    BIAS_SUM_ELTWISE(S00[0], T00, C00[idx], b0, D00[0], scales); \
    BIAS_SUM_ELTWISE(S00[1], T01, C01[idx], b1, D00[1], scales); \
    BIAS_SUM_ELTWISE(S00[2], T02, C02[idx], b2, D00[2], scales); \
    BIAS_SUM_ELTWISE(S00[3], T03, C03[idx], b3, D00[3], scales); \
    T0[idx] = as_uint(S00);                                   \
    D01 = as_uchar4(D1[idx]);                                 \
    BIAS_SUM_ELTWISE(S01[0], T10, C10[idx], b0, D01[0], scales); \
    BIAS_SUM_ELTWISE(S01[1], T11, C11[idx], b1, D01[1], scales); \
    BIAS_SUM_ELTWISE(S01[2], T12, C12[idx], b2, D01[2], scales); \
    BIAS_SUM_ELTWISE(S01[3], T13, C13[idx], b3, D01[3], scales); \
    T1[idx] = as_uint(S01);                                   

#else // WITH_SUM_ELTWISE || WITH_SUM
#define PACK(idx)                                             \
    BIAS_SUM_ELTWISE(S00[0], T00, C00[idx], b0, D00[0], scales); \
    BIAS_SUM_ELTWISE(S00[1], T01, C01[idx], b1, D00[1], scales); \
    BIAS_SUM_ELTWISE(S00[2], T02, C02[idx], b2, D00[2], scales); \
    BIAS_SUM_ELTWISE(S00[3], T03, C03[idx], b3, D00[3], scales); \
    T0[idx] = as_uint(S00);                                   \
    BIAS_SUM_ELTWISE(S01[0], T10, C10[idx], b0, D01[0], scales); \
    BIAS_SUM_ELTWISE(S01[1], T11, C11[idx], b1, D01[1], scales); \
    BIAS_SUM_ELTWISE(S01[2], T12, C12[idx], b2, D01[2], scales); \
    BIAS_SUM_ELTWISE(S01[3], T13, C13[idx], b3, D01[3], scales); \
    T1[idx] = as_uint(S01);                                   
#endif // WITH_SUM_ELTWISE || WITH_SUM

#if WITH_BIAS
        bias += (oc_group_id + sg_id) * OC_BLOCK + get_sub_group_local_id() * 4;
        float b0 = bias[0] * scales;
        float b1 = bias[1] * scales;
        float b2 = bias[2] * scales;
        float b3 = bias[3] * scales;
#endif // WITH_BIAS
#if WITH_SUM_ELTWISE || WITH_SUM
        uchar4 D00, D01;
        uint8 D0 = intel_sub_group_block_read8((__global uint *)dst);
        uint8 D1 = intel_sub_group_block_read8(
                (__global uint *)&dst[8 * OC_BLOCK]);
#endif // WITH_SUM_ELTWISE || WITH_SUM
        uchar4 S00, S01;
        uint8 T0, T1;
        float T00, T01, T02, T03;
        float T10, T11, T12, T13;
        PACK(0);
        PACK(1);
        PACK(2);
        PACK(3);
        PACK(4);
        PACK(5);
        PACK(6);
        PACK(7);

        intel_sub_group_block_write8((__global uint *)&dst[0 * OC_BLOCK], T0);
        intel_sub_group_block_write8((__global uint *)&dst[8 * OC_BLOCK], T1);
}
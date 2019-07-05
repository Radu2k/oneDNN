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
#include "ocl/ocl_types.h"

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));

#define BLOCK_READ_WHT_FROM_SLM(data, idx) \
    data = as_int8(READ_LOCAL_8((__local uint *)&wei_slm[idx]));

#define CHANNEL_OFFSET 1
#define MB_OFFSET IC_BLOCK

#define INPUT_PIXEL_WIDTH_OFFSET (MB_OFFSET * MB_BLOCK)
#define INPUT_PIXEL_HEIGHT_OFFSET (INPUT_PIXEL_WIDTH_OFFSET * IW)
#define INPUT_CHANNEL_BLOCK_OFFSET (INPUT_PIXEL_HEIGHT_OFFSET * IH) // For NChw

#define OUTPUT_PIXEL_WIDTH_OFFSET (MB_OFFSET * MB_BLOCK)
#define OUTPUT_PIXEL_HEIGHT_OFFSET (OUTPUT_PIXEL_WIDTH_OFFSET * OW)
#define OUTPUT_CHANNEL_BLOCK_OFFSET (OUTPUT_PIXEL_HEIGHT_OFFSET * OH) // For NChw

// Weights offsets
#define WEIGHTS_WIDTH_OFFSET (4 * 8 * 8 * 4)
#define WEIGHTS_HEIGHT_OFFSET (WEIGHTS_WIDTH_OFFSET * 1)
#define KERNEL_BLOCK_OFFSET (WEIGHTS_HEIGHT_OFFSET * 1)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
gen12hp_1x1_conv_fwd_kernel(const __global uchar *src, const __global char *wei,
        const __global float *bias, __global DATA_T *dst,
        float relu_negative_slope, float sum_scale, float scales) {

    // Groups:
    const uint oc_group_id = get_group_id(0);
    const uint sp_group_id = get_group_id(1);
    const uint sp_local_id = get_local_id(1);
    const uint mb_group_id = get_group_id(2);

    // SIMD
    const uint sg_local_id = get_sub_group_local_id();
    const uint sg_id = get_sub_group_id();

    // Spatial
    const uint sp = get_global_id(1);
    const uint ow = sp % OW_PADDED;
    const uint oh = sp / OW_PADDED;
    const uint iw = ow * SW;
    const uint ih = oh * SH;

    // Source (At ic = 0)
    src += mb_group_id * INPUT_CHANNEL_BLOCK_OFFSET * IC_NCHUNK; // MB offset
    src += ih * INPUT_PIXEL_HEIGHT_OFFSET; // height offset
    src += iw * INPUT_PIXEL_WIDTH_OFFSET; // width offset
    
    // Destination
    dst += mb_group_id * OUTPUT_CHANNEL_BLOCK_OFFSET * OC_NCHUNK; // MB offset
    dst += OUTPUT_CHANNEL_BLOCK_OFFSET * oc_group_id; //OC offset
    dst += oh * OUTPUT_PIXEL_HEIGHT_OFFSET;
    dst += ow * OUTPUT_PIXEL_WIDTH_OFFSET;

    // Weights
    wei += oc_group_id * KERNEL_BLOCK_OFFSET * IC_NCHUNK;

#ifdef SLM_WEI
    __local char wei_slm[OC_BLOCK * IC_BLOCK];
#endif // SLM_WEI

    // Output accumulators:
    // 8 MB (0-7) x 4 Kernels  (32 8bit ints)
    int8 C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    // 8 MB (8-15) x 4 Kernels  (32 8bit ints)
    int8 C10 = 0, C11 = 0, C12 = 0, C13 = 0;
    // 8 MB (16-23) x 4 Kernels  (32 8bit ints)
    int8 C20 = 0, C21 = 0, C22 = 0, C23 = 0;
    // 8 MB (24-31) x 4 Kernels  (32 8bit ints)
    int8 C30 = 0, C31 = 0, C32 = 0, C33 = 0;
    
    uint8 S0, S1, S2, S3;
    int8 W0, W1, W2, W3;

    __attribute__((opencl_unroll_hint))
    for(uint ic_block_id = 0; ic_block_id < IC_NCHUNK; ++ic_block_id)
    {
#ifdef SLM_WEI
        barrier(CLK_LOCAL_MEM_FENCE);
        if(sp_local_id == 0)
        {
            WRITE_LOCAL_8((__local uint *)&wei_slm[0],
                intel_sub_group_block_read8((__global uint *)&wei[0]));
            WRITE_LOCAL_8((__local uint *)&wei_slm[8 * IC_BLOCK],
                intel_sub_group_block_read8((__global uint *)&wei[8 * IC_BLOCK]));
            WRITE_LOCAL_8((__local uint *)&wei_slm[16 * IC_BLOCK],
                intel_sub_group_block_read8((__global uint *)&wei[16 * IC_BLOCK]));
            WRITE_LOCAL_8((__local uint *)&wei_slm[24 * IC_BLOCK],
                intel_sub_group_block_read8((__global uint *)&wei[24 * IC_BLOCK]));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        BLOCK_READ_SRC(S0, 0 * IC_BLOCK);
        BLOCK_READ_SRC(S1, 8 * IC_BLOCK);
        BLOCK_READ_SRC(S2, 16 * IC_BLOCK);
        BLOCK_READ_SRC(S3, 24 * IC_BLOCK);

#ifdef SLM_WEI
        BLOCK_READ_WHT_FROM_SLM(W0, 0);
        BLOCK_READ_WHT_FROM_SLM(W1, 8 * IC_BLOCK);
        BLOCK_READ_WHT_FROM_SLM(W2, 16 * IC_BLOCK);
        BLOCK_READ_WHT_FROM_SLM(W3, 24 * IC_BLOCK);
#else
        BLOCK_READ_WHT(W0, 0);
        BLOCK_READ_WHT(W1, 8 * IC_BLOCK);
        BLOCK_READ_WHT(W2, 16 * IC_BLOCK);
        BLOCK_READ_WHT(W3, 24 * IC_BLOCK);
#endif

        C00 = mmad8x8(S0, W0, C00);
        C01 = mmad8x8(S0, W1, C01);
        C02 = mmad8x8(S0, W2, C02);
        C03 = mmad8x8(S0, W3, C03);
        C10 = mmad8x8(S1, W0, C10);
        C11 = mmad8x8(S1, W1, C11);
        C12 = mmad8x8(S1, W2, C12);
        C13 = mmad8x8(S1, W3, C13);
        C20 = mmad8x8(S2, W0, C20);
        C21 = mmad8x8(S2, W1, C21);
        C22 = mmad8x8(S2, W2, C22);
        C23 = mmad8x8(S2, W3, C23);
        C30 = mmad8x8(S3, W0, C30);
        C31 = mmad8x8(S3, W1, C31);
        C32 = mmad8x8(S3, W2, C32);
        C33 = mmad8x8(S3, W3, C33);

        src += INPUT_CHANNEL_BLOCK_OFFSET;
        wei += KERNEL_BLOCK_OFFSET;
    }


#if WITH_BIAS
#if WITH_SUM_RELU
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)DST, sum_scale, BIA);            \
    TMP = fma((float)ACC, SCALE, TMP);                \
    if (TMP < 0)                                      \
        TMP *= relu_negative_slope;                   \
    RES = CONVERT_DATA_T(TMP);
#else // WITH_SUM_RELU
#if WITH_RELU && WITH_SUM
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)ACC, SCALE, BIA);                \
    if (TMP < 0)                                      \
        TMP *= relu_negative_slope;                   \
    TMP = fma((float)DST, sum_scale, TMP);            \
    RES = CONVERT_DATA_T(TMP);
#else // WITH_RELU && WITH_SUM
#if WITH_RELU
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)ACC, SCALE, BIA);                \
    if (TMP < 0)                                      \
        TMP *= relu_negative_slope;                   \
    RES = CONVERT_DATA_T(TMP);
#endif // WITH_RELU
#if WITH_SUM
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)DST, sum_scale, BIA);            \
    TMP = fma((float)ACC, SCALE, TMP);                \
    RES = CONVERT_DATA_T(TMP);
#endif
#if WITH_RELU == 0 && WITH_SUM == 0
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)ACC, SCALE, BIA);                \
    RES = CONVERT_DATA_T(TMP);
#endif
#endif // WITH_RELU && WITH_SUM
#endif // WITH_SUM_RELU
#else // WITH_BIAS
#if WITH_SUM_RELU
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA, DST, SCALE)     \
    TMP = fma((float)ACC, SCALE, (float)DST * sum_scale); \
    if (TMP < 0)                                          \
        TMP *= relu_negative_slope;                       \
    RES = CONVERT_DATA_T(TMP);
#else // WITH_SUM_RELU
#if WITH_RELU && WITH_SUM
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = (float)ACC * SCALE;                         \
    if (TMP < 0)                                      \
        TMP *= relu_negative_slope;                   \
    TMP = fma((float)DST, sum_scale, TMP);            \
    RES = CONVERT_DATA_T(TMP);
#else // WITH_RELU && WITH_SUM
#if WITH_RELU
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = (float)ACC * SCALE;                         \
    if (TMP < 0)                                      \
        TMP *= relu_negative_slope;                   \
    RES = CONVERT_DATA_T(TMP);
#endif // WITH_RELU
#if WITH_SUM
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA, DST, SCALE)     \
    TMP = fma((float)ACC, SCALE, (float)DST * sum_scale); \
    RES = CONVERT_DATA_T(TMP);
#endif // WITH_SUM
#if WITH_RELU == 0 && WITH_SUM == 0
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = (float)ACC * SCALE;                         \
    RES = CONVERT_DATA_T(TMP);
#endif
#endif // WITH_RELU && WITH_SUM
#endif // WITH_SUM_RELU
#endif // WITH_BIAS

#if WITH_SUM_RELU || WITH_SUM
#define PACK(idx)                                             \
    D00 = AS_DATA4_T(D0[idx]);                                \
    BIAS_SUM_RELU(S00[0], T00, C00[idx], b0, D00[0], scales); \
    BIAS_SUM_RELU(S00[1], T01, C01[idx], b1, D00[1], scales); \
    BIAS_SUM_RELU(S00[2], T02, C02[idx], b2, D00[2], scales); \
    BIAS_SUM_RELU(S00[3], T03, C03[idx], b3, D00[3], scales); \
    T0[idx] = as_uint(S00);                                   \
    D01 = AS_DATA4_T(D1[idx]);                                \
    BIAS_SUM_RELU(S01[0], T10, C10[idx], b0, D01[0], scales); \
    BIAS_SUM_RELU(S01[1], T11, C11[idx], b1, D01[1], scales); \
    BIAS_SUM_RELU(S01[2], T12, C12[idx], b2, D01[2], scales); \
    BIAS_SUM_RELU(S01[3], T13, C13[idx], b3, D01[3], scales); \
    T1[idx] = as_uint(S01);                                   \
    D02 = AS_DATA4_T(D2[idx]);                                \
    BIAS_SUM_RELU(S02[0], T20, C20[idx], b0, D02[0], scales); \
    BIAS_SUM_RELU(S02[1], T21, C21[idx], b1, D02[1], scales); \
    BIAS_SUM_RELU(S02[2], T22, C22[idx], b2, D02[2], scales); \
    BIAS_SUM_RELU(S02[3], T23, C23[idx], b3, D02[3], scales); \
    T2[idx] = as_uint(S02);                                   \
    D03 = AS_DATA4_T(D3[idx]);                                \
    BIAS_SUM_RELU(S03[0], T30, C30[idx], b0, D03[0], scales); \
    BIAS_SUM_RELU(S03[1], T31, C31[idx], b1, D03[1], scales); \
    BIAS_SUM_RELU(S03[2], T32, C32[idx], b2, D03[2], scales); \
    BIAS_SUM_RELU(S03[3], T33, C33[idx], b3, D03[3], scales); \
    T3[idx] = as_uint(S03);

#else // WITH_SUM_RELU || WITH_SUM
#define PACK(idx)                                             \
    BIAS_SUM_RELU(S00[0], T00, C00[idx], b0, D00[0], scales); \
    BIAS_SUM_RELU(S00[1], T01, C01[idx], b1, D00[1], scales); \
    BIAS_SUM_RELU(S00[2], T02, C02[idx], b2, D00[2], scales); \
    BIAS_SUM_RELU(S00[3], T03, C03[idx], b3, D00[3], scales); \
    T0[idx] = as_uint(S00);                                   \
    BIAS_SUM_RELU(S01[0], T10, C10[idx], b0, D01[0], scales); \
    BIAS_SUM_RELU(S01[1], T11, C11[idx], b1, D01[1], scales); \
    BIAS_SUM_RELU(S01[2], T12, C12[idx], b2, D01[2], scales); \
    BIAS_SUM_RELU(S01[3], T13, C13[idx], b3, D01[3], scales); \
    T1[idx] = as_uint(S01);                                   \
    BIAS_SUM_RELU(S02[0], T20, C20[idx], b0, D02[0], scales); \
    BIAS_SUM_RELU(S02[1], T21, C21[idx], b1, D02[1], scales); \
    BIAS_SUM_RELU(S02[2], T22, C22[idx], b2, D02[2], scales); \
    BIAS_SUM_RELU(S02[3], T23, C23[idx], b3, D02[3], scales); \
    T2[idx] = as_uint(S02);                                   \
    BIAS_SUM_RELU(S03[0], T30, C30[idx], b0, D03[0], scales); \
    BIAS_SUM_RELU(S03[1], T31, C31[idx], b1, D03[1], scales); \
    BIAS_SUM_RELU(S03[2], T32, C32[idx], b2, D03[2], scales); \
    BIAS_SUM_RELU(S03[3], T33, C33[idx], b3, D03[3], scales); \
    T3[idx] = as_uint(S03);
#endif // WITH_SUM_RELU || WITH_SUM

if(ow < OW)
{
#if WITH_BIAS
        bias += oc_group_id * OC_BLOCK + get_sub_group_local_id() * 4;
        float b0 = bias[0] * scales;
        float b1 = bias[1] * scales;
        float b2 = bias[2] * scales;
        float b3 = bias[3] * scales;
#endif // WITH_BIAS
#if WITH_SUM_RELU || WITH_SUM
        DATA4_T D00, D01, D02, D03;
        uint8 D0 = intel_sub_group_block_read8((__global uint *)dst);
        uint8 D1 = intel_sub_group_block_read8(
                (__global uint *)&dst[8 * OC_BLOCK]);
        uint8 D2 = intel_sub_group_block_read8(
                (__global uint *)&dst[16 * OC_BLOCK]);
        uint8 D3 = intel_sub_group_block_read8(
                (__global uint *)&dst[24 * OC_BLOCK]); 
#endif // WITH_SUM_RELU || WITH_SUM
        DATA4_T S00, S01, S02, S03;
        uint8 T0, T1, T2, T3;
        float T00, T01, T02, T03;
        float T10, T11, T12, T13;
        float T20, T21, T22, T23;
        float T30, T31, T32, T33;
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
        intel_sub_group_block_write8((__global uint *)&dst[16 * OC_BLOCK], T2);
        intel_sub_group_block_write8((__global uint *)&dst[24 * OC_BLOCK], T3);
}
}

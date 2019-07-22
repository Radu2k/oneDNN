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

#include "ocl/ocl_math_utils.h"
#include "ocl/ocl_types.h"

#undef MB_FULL_BLOCK

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
conv_fwd_u8s8s32x_kernel(const __global uchar *src, const __global char *wei,
        const __global float *bias, __global DATA_T *dst,
        float relu_negative_slope, float sum_scale, float scales) {

#ifdef MB_FULL_BLOCK
    const int mb_blocks = 1;
#else // MB_FULL_BLOCK
    const int mb_blocks = 2;
#endif // MB_FULL_BLOCK

    const int group_oc = get_group_id(0) * OC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP / mb_blocks;
    const int group_sp = get_group_id(1) * SP_GROUP;

    const int sub_group_id = get_sub_group_id();
    const int mb = get_group_id(2) % mb_blocks;
    const int oc = (sub_group_id % OC_GROUP);
    const int sp = (sub_group_id / OC_GROUP);

    const int g = (group_oc + oc) / OC_NCHUNK;
    const int group_ic = IC_NCHUNK * g;

    const int god = group_sp / (OW_PADDED * OH);
    const int gohw = group_sp % (OW_PADDED * OH);
    const int goh = gohw / OW_PADDED;
    const int gow = gohw % OW_PADDED;

    const int gid = god * SD;
    const int gih = goh * SH;
    const int giw = gow * SW;

    const int local_oh = sp / OW_PADDED;
    const int local_ow = sp % OW_PADDED;
    const int local_ih = local_oh * SH;
    const int local_iw = local_ow * SW;

    const int od = god;
    const int ow = gow + local_ow;
    const int oh = goh + local_oh;
    const int id = gid - PD;
    const int iw = giw + local_iw - PW;
    const int ih = gih + local_ih - PH;

    if (ow >= OW)
        return;

    dst += OC_BLOCK * OD * OH * OW * MB_BLOCK * (group_oc + oc);
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK * group_mb;
    dst += OC_BLOCK * MB_BLOCK / 2 * mb;
    dst += OC_BLOCK * MB_BLOCK * (OW * OH * od + OW * oh + ow);

    src += IC_BLOCK * ID * IH * IW * MB_BLOCK * group_ic;
    src += IC_BLOCK * ID * IH * IW * IC_NCHUNK * G * MB_BLOCK * group_mb;
    src += IC_BLOCK * MB_BLOCK / 2 * mb;
    src += IC_BLOCK * MB_BLOCK * (IW * IH * id + IW * ih + iw);

    wei += IC_BLOCK * KD * KH * KW * OC_BLOCK * (group_oc + oc) * IC_NCHUNK;

    int8 C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    int8 C10 = 0, C11 = 0, C12 = 0, C13 = 0;
    int8 C20 = 0, C21 = 0, C22 = 0, C23 = 0;
    int8 C30 = 0, C31 = 0, C32 = 0, C33 = 0;

    __attribute__((opencl_unroll_hint))
    for (int ic_chunk = 0; ic_chunk < IC_NCHUNK; ic_chunk++) {
        uint8 S0, S1, S2, S3;
        int8 W0, W1, W2, W3;
        for (int kd = 0; kd < KD; kd++) {
            if (kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID) {
                src += IC_BLOCK * MB_BLOCK * IH * IW * (1 + DD);
                wei += IC_BLOCK * OC_BLOCK * KH * KW;
                continue;
            }
            for (int kh = 0; kh < KH; kh++) {
                if (kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH) {
                    src += IC_BLOCK * MB_BLOCK * IW * (1 + DH);
                    wei += IC_BLOCK * OC_BLOCK * KW;
                    continue;
                }
                __attribute__((opencl_unroll_hint))
                for (int kw = 0; kw < KW; kw++) {
                    if (kw * (1 + DW) + iw >= 0 && kw * (1 + DW) + iw < IW) {
                        BLOCK_READ_SRC(S0, 0);
#if MB > 8
                        BLOCK_READ_SRC(S1, 8 * IC_BLOCK);
#ifdef MB_FULL_BLOCK
                        BLOCK_READ_SRC(S2, 16 * IC_BLOCK);
                        BLOCK_READ_SRC(S3, 24 * IC_BLOCK);
#endif // MB_FULL_BLOCK
#endif // MB > 8
                        BLOCK_READ_WHT(W0, 0);
                        BLOCK_READ_WHT(W1, 8 * IC_BLOCK);
                        BLOCK_READ_WHT(W2, 16 * IC_BLOCK);
                        BLOCK_READ_WHT(W3, 24 * IC_BLOCK);
                        C00 = mmad8x8(S0, W0, C00);
                        C01 = mmad8x8(S0, W1, C01);
                        C02 = mmad8x8(S0, W2, C02);
                        C03 = mmad8x8(S0, W3, C03);
#if MB > 8
                        C10 = mmad8x8(S1, W0, C10);
                        C11 = mmad8x8(S1, W1, C11);
                        C12 = mmad8x8(S1, W2, C12);
                        C13 = mmad8x8(S1, W3, C13);
#ifdef MB_FULL_BLOCK
                        C20 = mmad8x8(S2, W0, C20);
                        C21 = mmad8x8(S2, W1, C21);
                        C22 = mmad8x8(S2, W2, C22);
                        C23 = mmad8x8(S2, W3, C23);
                        C30 = mmad8x8(S3, W0, C30);
                        C31 = mmad8x8(S3, W1, C31);
                        C32 = mmad8x8(S3, W2, C32);
                        C33 = mmad8x8(S3, W3, C33);
#endif // MB_FULL_BLOCK
#endif // MB > 8
                    }
                    src += IC_BLOCK * MB_BLOCK * (1 + DW);
                    wei += IC_BLOCK * OC_BLOCK;
                }
                src += IC_BLOCK * MB_BLOCK * (IW * (1 + DH) - KW * (1 + DW));
            }
            src += IC_BLOCK * MB_BLOCK * (IH * (1 + DD) - KH * (1 + DH)) * IW;
        }
        src += IC_BLOCK * MB_BLOCK * (ID - KD * (1 + DD)) * IH * IW;
    }
#if WITH_BIAS
#if WITH_SUM_ELTWISE
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)DST, sum_scale, BIA);            \
    TMP = fma((float)ACC, SCALE, TMP);                \
    if (TMP < 0)                                      \
        TMP *= relu_negative_slope;                   \
    RES = CONVERT_DATA_T(TMP);
#else // WITH_SUM_ELTWISE
#if WITH_RELU && WITH_SUM
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)ACC, SCALE, BIA);                \
    if (TMP < 0)                                      \
        TMP *= relu_negative_slope;                   \
    TMP = fma((float)DST, sum_scale, TMP);            \
    RES = CONVERT_DATA_T(TMP);
#else // WITH_RELU && WITH_SUM
#if WITH_RELU
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)ACC, SCALE, BIA);                \
    if (TMP < 0)                                      \
        TMP *= relu_negative_slope;                   \
    RES = CONVERT_DATA_T(TMP);
#endif // WITH_RELU
#if WITH_SUM
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)DST, sum_scale, BIA);            \
    TMP = fma((float)ACC, SCALE, TMP);                \
    RES = CONVERT_DATA_T(TMP);
#endif
#if WITH_RELU == 0 && WITH_SUM == 0
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = fma((float)ACC, SCALE, BIA);                \
    RES = CONVERT_DATA_T(TMP);
#endif
#endif // WITH_RELU && WITH_SUM
#endif // WITH_SUM_ELTWISE
#else // WITH_BIAS
#if WITH_SUM_ELTWISE
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE)     \
    TMP = fma((float)ACC, SCALE, (float)DST * sum_scale); \
    if (TMP < 0)                                          \
        TMP *= relu_negative_slope;                       \
    RES = CONVERT_DATA_T(TMP);
#else // WITH_SUM_ELTWISE
#if WITH_RELU && WITH_SUM
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = (float)ACC * SCALE;                         \
    if (TMP < 0)                                      \
        TMP *= relu_negative_slope;                   \
    TMP = fma((float)DST, sum_scale, TMP);            \
    RES = CONVERT_DATA_T(TMP);
#else // WITH_RELU && WITH_SUM
#if WITH_RELU
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = (float)ACC * SCALE;                         \
    if (TMP < 0)                                      \
        TMP *= relu_negative_slope;                   \
    RES = CONVERT_DATA_T(TMP);
#endif // WITH_RELU
#if WITH_SUM
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE)     \
    TMP = fma((float)ACC, SCALE, (float)DST * sum_scale); \
    RES = CONVERT_DATA_T(TMP);
#endif // WITH_SUM
#if WITH_RELU == 0 && WITH_SUM == 0
#define BIAS_SUM_ELTWISE(RES, TMP, ACC, BIA, DST, SCALE) \
    TMP = (float)ACC * SCALE;                         \
    RES = CONVERT_DATA_T(TMP);
#endif
#endif // WITH_RELU && WITH_SUM
#endif // WITH_SUM_ELTWISE
#endif // WITH_BIAS

#if WITH_SUM_ELTWISE || WITH_SUM
#define PACK(idx)                                             \
    D00 = AS_DATA4_T(D0[idx]);                                \
    BIAS_SUM_ELTWISE(S00[0], T00, C00[idx], b0, D00[0], scales); \
    BIAS_SUM_ELTWISE(S00[1], T01, C01[idx], b1, D00[1], scales); \
    BIAS_SUM_ELTWISE(S00[2], T02, C02[idx], b2, D00[2], scales); \
    BIAS_SUM_ELTWISE(S00[3], T03, C03[idx], b3, D00[3], scales); \
    T0[idx] = as_uint(S00);                                   \
    D01 = AS_DATA4_T(D1[idx]);                                \
    BIAS_SUM_ELTWISE(S01[0], T10, C10[idx], b0, D01[0], scales); \
    BIAS_SUM_ELTWISE(S01[1], T11, C11[idx], b1, D01[1], scales); \
    BIAS_SUM_ELTWISE(S01[2], T12, C12[idx], b2, D01[2], scales); \
    BIAS_SUM_ELTWISE(S01[3], T13, C13[idx], b3, D01[3], scales); \
    T1[idx] = as_uint(S01);                                   \
    D02 = AS_DATA4_T(D2[idx]);                                \
    BIAS_SUM_ELTWISE(S02[0], T20, C20[idx], b0, D02[0], scales); \
    BIAS_SUM_ELTWISE(S02[1], T21, C21[idx], b1, D02[1], scales); \
    BIAS_SUM_ELTWISE(S02[2], T22, C22[idx], b2, D02[2], scales); \
    BIAS_SUM_ELTWISE(S02[3], T23, C23[idx], b3, D02[3], scales); \
    T2[idx] = as_uint(S02);                                   \
    D03 = AS_DATA4_T(D3[idx]);                                \
    BIAS_SUM_ELTWISE(S03[0], T30, C30[idx], b0, D03[0], scales); \
    BIAS_SUM_ELTWISE(S03[1], T31, C31[idx], b1, D03[1], scales); \
    BIAS_SUM_ELTWISE(S03[2], T32, C32[idx], b2, D03[2], scales); \
    BIAS_SUM_ELTWISE(S03[3], T33, C33[idx], b3, D03[3], scales); \
    T3[idx] = as_uint(S03);

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
    T1[idx] = as_uint(S01);                                   \
    BIAS_SUM_ELTWISE(S02[0], T20, C20[idx], b0, D02[0], scales); \
    BIAS_SUM_ELTWISE(S02[1], T21, C21[idx], b1, D02[1], scales); \
    BIAS_SUM_ELTWISE(S02[2], T22, C22[idx], b2, D02[2], scales); \
    BIAS_SUM_ELTWISE(S02[3], T23, C23[idx], b3, D02[3], scales); \
    T2[idx] = as_uint(S02);                                   \
    BIAS_SUM_ELTWISE(S03[0], T30, C30[idx], b0, D03[0], scales); \
    BIAS_SUM_ELTWISE(S03[1], T31, C31[idx], b1, D03[1], scales); \
    BIAS_SUM_ELTWISE(S03[2], T32, C32[idx], b2, D03[2], scales); \
    BIAS_SUM_ELTWISE(S03[3], T33, C33[idx], b3, D03[3], scales); \
    T3[idx] = as_uint(S03);
#endif // WITH_SUM_ELTWISE || WITH_SUM

    if (ow < OW) {
#if WITH_BIAS
        bias += (group_oc + oc) * OC_BLOCK + get_sub_group_local_id() * 4;
        float b0 = bias[0] * scales;
        float b1 = bias[1] * scales;
        float b2 = bias[2] * scales;
        float b3 = bias[3] * scales;
#endif // WITH_BIAS
#if WITH_SUM_ELTWISE || WITH_SUM
        DATA4_T D00, D01, D02, D03;
        uint8 D0 = intel_sub_group_block_read8((__global uint *)dst);
        uint8 D1 = intel_sub_group_block_read8(
                (__global uint *)&dst[8 * OC_BLOCK]);
        uint8 D2 = intel_sub_group_block_read8(
                (__global uint *)&dst[16 * OC_BLOCK]);
        uint8 D3 = intel_sub_group_block_read8(
                (__global uint *)&dst[24 * OC_BLOCK]);
#endif // WITH_SUM_ELTWISE || WITH_SUM
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
#if MB > 8
        intel_sub_group_block_write8((__global uint *)&dst[8 * OC_BLOCK], T1);
#ifdef MB_FULL_BLOCK
        intel_sub_group_block_write8((__global uint *)&dst[16 * OC_BLOCK], T2);
        intel_sub_group_block_write8((__global uint *)&dst[24 * OC_BLOCK], T3);
#endif // MB_FULL_BLOCK
#endif // MB > 8
    }
}

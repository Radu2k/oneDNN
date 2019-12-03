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

#ifndef MB_FULL_BLOCK
#define MB_FULL_BLOCK
#endif

#if KW * OC_BLOCK * IC_BLOCK * IC_GROUP <= 8192
#define SLM_WEI
#endif

#ifdef SLM_WEI
#define WEI wei_tmp
#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(READ_LOCAL_8((__local uint *)&WEI[idx]));
#else
#define WEI wei
#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&WEI[idx]));
#endif

#define BLOCK_READ_DST(data, idx) \
    data = AS_INT8_T( \
            intel_sub_group_block_read8((__global uint *)&current_dst[idx]));

#define BLOCK_READ_BIA(data, idx) \
    data = as_float4(intel_sub_group_block_read4((__global uint *)&bias[idx]));

#undef CONVERT_DATA_T
#define CONVERT_DATA_T convert_uchar_sat

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
gen12hp_conv_bwd_data_u8s8s32x_kernel(const __global uchar *src,
        const __global char *wei, const __global float *bias,
        __global DATA_T *dst) {

#ifdef MB_FULL_BLOCK
    const int mb_blocks = 1;
#else // MB_FULL_BLOCK
    const int mb_blocks = 2;
#endif // MB_FULL_BLOCK

    const int group_ic = get_group_id(0) * IC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP / mb_blocks;
    const int group_sp = get_group_id(1) * SP_GROUP;

    const int sub_group_id = get_sub_group_id();
    const int mb = get_group_id(2) % mb_blocks;
    const int ic = (sub_group_id % IC_GROUP);
    const int sp = (sub_group_id / IC_GROUP);

    const int g = (group_ic + ic) / IC_NCHUNK;
    const int group_oc = OC_NCHUNK * g;

    const int gid = group_sp / (IW_PADDED * IH);
    const int gihw = group_sp % (IW_PADDED * IH);
    const int gih = gihw / IW_PADDED;
    const int giw = gihw % IW_PADDED;

    const int local_ih = sp / IW_PADDED;
    const int local_iw = sp % IW_PADDED;

    const int id = gid;
    const int iw = giw + local_iw;
    const int ih = gih + local_ih;

#ifndef SLM_WEI
    if (iw >= IW) return;
#endif // SLM_WEI

    src += IC_BLOCK * ID * IH * IW * MB_BLOCK * (group_ic + ic);
    src += IC_BLOCK * ID * IH * IW * IC_NCHUNK * G * MB_BLOCK * group_mb;
    src += IC_BLOCK * MB_BLOCK / 2 * mb;
    src += IC_BLOCK * MB_BLOCK * (IW * IH * id + IW * ih + iw);

    dst += OC_BLOCK * OD * OH * OW * MB_BLOCK * group_oc;
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK * group_mb;
    dst += OC_BLOCK * MB_BLOCK / 2 * mb;

    wei += OC_BLOCK * KD * KH * KW * IC_BLOCK * (group_ic + ic) * OC_NCHUNK;

    int8 C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    int8 C10 = 0, C11 = 0, C12 = 0, C13 = 0;
    int8 C20 = 0, C21 = 0, C22 = 0, C23 = 0;
    int8 C30 = 0, C31 = 0, C32 = 0, C33 = 0;

#ifdef SLM_WEI
    __local char wei_loc[KW * OC_BLOCK * IC_BLOCK * IC_GROUP];
    __local char *wei_loc_base = wei_loc + KW * IC_BLOCK * OC_BLOCK * ic;
#endif // SLM_WEI

    __attribute__((opencl_unroll_hint)) for (int oc_chunk = 0;
                                             oc_chunk < OC_NCHUNK; oc_chunk++) {
        INT8_T D0, D1, D2, D3;
        int8 W0, W1, W2, W3;
        for (int kd = 0; kd < KD; kd++) {
            if ((id + PD - kd * (1 + DD)) % SD != 0) {
                wei += IC_BLOCK * OC_BLOCK * KH * KW;
                continue;
            }
            const int od = (id + PD - kd * (1 + DD)) / SD;
            if (od < 0 || od >= OD) {
                wei += IC_BLOCK * OC_BLOCK * KH * KW;
                continue;
            }
            for (int kh = 0; kh < KH; kh++) {
                if ((ih + PH - kh * (1 + DH)) % SH != 0) {
                    wei += IC_BLOCK * OC_BLOCK * KW;
                    continue;
                }
                const int oh = (ih + PH - kh * (1 + DH)) / SH;
                if (oh < 0 || oh >= OH) {
                    wei += IC_BLOCK * OC_BLOCK * KW;
                    continue;
                }

#ifdef SLM_WEI
                barrier(CLK_LOCAL_MEM_FENCE);
                const __global char *wei_copy_from
                        = wei + sp * KW * OC_BLOCK * IC_BLOCK / 8;
                __local char *wei_copy_to
                        = wei_loc_base + sp * KW * OC_BLOCK * IC_BLOCK / 8;
                for (int bl = 0; bl < KW; bl++) {
                    WRITE_LOCAL_4(
                            (__local uint *)&wei_copy_to[bl * 4 * OC_BLOCK],
                            intel_sub_group_block_read4(
                                    (__global uint *)&wei_copy_from[bl * 4
                                            * OC_BLOCK]));
                }
                __local char *wei_tmp = wei_loc_base;
                barrier(CLK_LOCAL_MEM_FENCE);
#endif // SLM_WEI

                __attribute__((opencl_unroll_hint)) for (int kw = 0; kw < KW;
                                                         kw++) {
                    if ((iw + PW - kw * (1 + DW)) % SW == 0) {
                        const int ow = (iw + PW - kw * (1 + DW)) / SW;
                        if (ow >= 0 && ow < OW) {
                            __global DATA_T *current_dst = dst
                                    + OC_BLOCK * MB_BLOCK
                                            * (OW * OH * od + OW * oh + ow);
                            BLOCK_READ_DST(D0, 0);
#if MB > 8
                            BLOCK_READ_DST(D1, 8 * OC_BLOCK);
#ifdef MB_FULL_BLOCK
                            BLOCK_READ_DST(D2, 16 * OC_BLOCK);
                            BLOCK_READ_DST(D3, 24 * OC_BLOCK);
#endif // MB_FULL_BLOCK
#endif // MB > 8
                            BLOCK_READ_WHT(W0, 0);
                            BLOCK_READ_WHT(W1, 8 * IC_BLOCK);
                            BLOCK_READ_WHT(W2, 16 * IC_BLOCK);
                            BLOCK_READ_WHT(W3, 24 * IC_BLOCK);
                            C00 = MMAD8X8(D0, W0, C00);
                            C01 = MMAD8X8(D0, W1, C01);
                            C02 = MMAD8X8(D0, W2, C02);
                            C03 = MMAD8X8(D0, W3, C03);
#if MB > 8
                            C10 = MMAD8X8(D1, W0, C10);
                            C11 = MMAD8X8(D1, W1, C11);
                            C12 = MMAD8X8(D1, W2, C12);
                            C13 = MMAD8X8(D1, W3, C13);
#ifdef MB_FULL_BLOCK
                            C20 = MMAD8X8(D2, W0, C20);
                            C21 = MMAD8X8(D2, W1, C21);
                            C22 = MMAD8X8(D2, W2, C22);
                            C23 = MMAD8X8(D2, W3, C23);
                            C30 = MMAD8X8(D3, W0, C30);
                            C31 = MMAD8X8(D3, W1, C31);
                            C32 = MMAD8X8(D3, W2, C32);
                            C33 = MMAD8X8(D3, W3, C33);
#endif // MB_FULL_BLOCK
#endif // MB > 8
                        }
                    }
                    WEI += IC_BLOCK * OC_BLOCK;
                }
#ifdef SLM_WEI
                wei += IC_BLOCK * OC_BLOCK * KW;
#endif // SLM_WEI
            }
        }
        dst += OC_BLOCK * MB_BLOCK * OD * OH * OW;
    }

#if WITH_BIAS
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA) \
    TMP = (float)ACC + BIA; \
    RES = CONVERT_DATA_T(TMP);
#else // WITH_BIAS
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA) RES = CONVERT_DATA_T((float)ACC);
#endif // WITH_BIAS

#define PACK(idx) \
    BIAS_SUM_RELU(D00[0], T00, C00[idx], b0); \
    BIAS_SUM_RELU(D00[1], T01, C01[idx], b1); \
    BIAS_SUM_RELU(D00[2], T02, C02[idx], b2); \
    BIAS_SUM_RELU(D00[3], T03, C03[idx], b3); \
    T0[idx] = as_uint(D00); \
    BIAS_SUM_RELU(D01[0], T10, C10[idx], b0); \
    BIAS_SUM_RELU(D01[1], T11, C11[idx], b1); \
    BIAS_SUM_RELU(D01[2], T12, C12[idx], b2); \
    BIAS_SUM_RELU(D01[3], T13, C13[idx], b3); \
    T1[idx] = as_uint(D01); \
    BIAS_SUM_RELU(D02[0], T20, C20[idx], b0); \
    BIAS_SUM_RELU(D02[1], T21, C21[idx], b1); \
    BIAS_SUM_RELU(D02[2], T22, C22[idx], b2); \
    BIAS_SUM_RELU(D02[3], T23, C23[idx], b3); \
    T2[idx] = as_uint(D02); \
    BIAS_SUM_RELU(D03[0], T30, C30[idx], b0); \
    BIAS_SUM_RELU(D03[1], T31, C31[idx], b1); \
    BIAS_SUM_RELU(D03[2], T32, C32[idx], b2); \
    BIAS_SUM_RELU(D03[3], T33, C33[idx], b3); \
    T3[idx] = as_uint(D03);

#ifdef SLM_WEI
    if (iw < IW) {
#endif // SLM_WEI
#if WITH_BIAS
        float4 bia;
        BLOCK_READ_BIA(bia, (group_ic + ic) * IC_BLOCK);
#endif // WITH_BIAS

        uchar4 D00, D01, D02, D03;
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

        intel_sub_group_block_write_uc16(
                (__global uchar *)&src[0 * IC_BLOCK], as_uchar16(T0.s0123));
        intel_sub_group_block_write_uc16(
                (__global uchar *)&src[4 * IC_BLOCK], as_uchar16(T0.s4567));
#if MB > 8
        intel_sub_group_block_write_uc16(
                (__global uchar *)&src[8 * IC_BLOCK], as_uchar16(T1.s0123));
        intel_sub_group_block_write_uc16(
                (__global uchar *)&src[12 * IC_BLOCK], as_uchar16(T1.s4567));
#ifdef MB_FULL_BLOCK
        intel_sub_group_block_write_uc16(
                (__global uchar *)&src[16 * IC_BLOCK], as_uchar16(T2.s0123));
        intel_sub_group_block_write_uc16(
                (__global uchar *)&src[20 * IC_BLOCK], as_uchar16(T2.s4567));
        intel_sub_group_block_write_uc16(
                (__global uchar *)&src[24 * IC_BLOCK], as_uchar16(T3.s0123));
        intel_sub_group_block_write_uc16(
                (__global uchar *)&src[28 * IC_BLOCK], as_uchar16(T3.s4567));
#endif // MB_FULL_BLOCK
#endif // MB > 8
#ifdef SLM_WEI
    }
#endif // SLM_WEI
}

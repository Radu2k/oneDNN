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
#if WITH_ELTWISE == 1 || WITH_POST_SUM_ELTWISE == 1
#include "ocl/ocl_post_ops.h"
#endif

#ifndef MB_FULL_BLOCK
#define MB_FULL_BLOCK
#endif

#if KW * OC_BLOCK * IC_BLOCK * OC_GROUP <= 8192
#define SLM_WEI
#endif

#ifdef SLM_WEI
#define WEI wei_tmp
#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(READ_LOCAL_8((__local uint *)&wei_tmp[idx]));
#else
#define WEI wei
#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&WEI[idx]));
#endif

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
gen12hp_conv_fwd_u8s8s32x_kernel(const __global uchar *src,
        const __global char *wei, const __global float *bias,
        __global DATA_T *dst, float alpha, float beta, float sum_scale,
        float scales) {

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

#ifndef SLM_WEI
    if (ow >= OW) return;
#endif // SLM_WEI

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

#ifdef SLM_WEI
    __local char wei_loc[KW * OC_BLOCK * IC_BLOCK * OC_GROUP];
    __local char *wei_loc_base = wei_loc + KW * IC_BLOCK * OC_BLOCK * oc;
#endif // SLM_WEI

    __attribute__((opencl_unroll_hint)) for (int ic_chunk = 0;
                                             ic_chunk < IC_NCHUNK; ic_chunk++) {
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

#ifdef SLM_WEI
                barrier(CLK_LOCAL_MEM_FENCE);
                const __global char *wei_copy_from
                        = wei + sp * KW * OC_BLOCK * IC_BLOCK / 8;
                __local char *wei_copy_to
                        = wei_loc_base + sp * KW * OC_BLOCK * IC_BLOCK / 8;
                for (int bl = 0; bl < KW; bl++) {
                    WRITE_LOCAL_4(
                            (__local uint *)&wei_copy_to[bl * 4 * IC_BLOCK],
                            intel_sub_group_block_read4(
                                    (__global uint *)&wei_copy_from[bl * 4
                                            * IC_BLOCK]));
                }
                __local char *wei_tmp = wei_loc_base;
                barrier(CLK_LOCAL_MEM_FENCE);
#endif // SLM_WEI

                __attribute__((opencl_unroll_hint)) for (int kw = 0; kw < KW;
                                                         kw++) {
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
                        C00 = MMAD8X8(S0, W0, C00);
                        C01 = MMAD8X8(S0, W1, C01);
                        C02 = MMAD8X8(S0, W2, C02);
                        C03 = MMAD8X8(S0, W3, C03);
#if MB > 8
                        C10 = MMAD8X8(S1, W0, C10);
                        C11 = MMAD8X8(S1, W1, C11);
                        C12 = MMAD8X8(S1, W2, C12);
                        C13 = MMAD8X8(S1, W3, C13);
#ifdef MB_FULL_BLOCK
                        C20 = MMAD8X8(S2, W0, C20);
                        C21 = MMAD8X8(S2, W1, C21);
                        C22 = MMAD8X8(S2, W2, C22);
                        C23 = MMAD8X8(S2, W3, C23);
                        C30 = MMAD8X8(S3, W0, C30);
                        C31 = MMAD8X8(S3, W1, C31);
                        C32 = MMAD8X8(S3, W2, C32);
                        C33 = MMAD8X8(S3, W3, C33);
#endif // MB_FULL_BLOCK
#endif // MB > 8
                    }
                    src += IC_BLOCK * MB_BLOCK * (1 + DW);
                    WEI += IC_BLOCK * OC_BLOCK;
                }
#ifdef SLM_WEI
                wei += IC_BLOCK * OC_BLOCK * KW;
#endif // SLM_WEI
                src += IC_BLOCK * MB_BLOCK * (IW * (1 + DH) - KW * (1 + DW));
            }
            src += IC_BLOCK * MB_BLOCK * (IH * (1 + DD) - KH * (1 + DH)) * IW;
        }
        src += IC_BLOCK * MB_BLOCK * (ID - KD * (1 + DD)) * IH * IW;
    }

    float4 tmp;
    uint8 dst_pack;
    uint8 D0, D1, D2, D3;

#if WITH_BIAS
    bias += (group_oc + oc) * OC_BLOCK + get_sub_group_local_id() * 4;
    float4 bia = (float4)(bias[0], bias[1], bias[2], bias[3]);
    bia *= scales;
#define QUANTIZE_ADD_BIAS() tmp = fma(tmp, (float4)scales, bia);
#else
#define QUANTIZE_ADD_BIAS() tmp *= scales;
#endif

#if WITH_SUM
#define DO_SUM(d_pack) \
    do { \
        DATA4_T d = AS_DATA4_T(d_pack); \
        float4 df = convert_float4(d); \
        tmp = fma(df, (float4)sum_scale, tmp); \
    } while (0)
#else
#define DO_SUM(d) ;
#endif // with_sum

#define ELTWISE() \
    do { \
        tmp[0] = fwd_eltwise(tmp[0], alpha, beta); \
        tmp[1] = fwd_eltwise(tmp[1], alpha, beta); \
        tmp[2] = fwd_eltwise(tmp[2], alpha, beta); \
        tmp[3] = fwd_eltwise(tmp[3], alpha, beta); \
    } while (0)

#if WITH_ELTWISE
#define DO_ELTWISE() ELTWISE();
#else
#define DO_ELTWISE() ;
#endif

#if WITH_POST_SUM_ELTWISE
#define DO_POST_SUM_ELTWISE() ELTWISE();
#else
#define DO_POST_SUM_ELTWISE() ;
#endif

#define PACK(C0, C1, C2, C3, idx) \
    do { \
        tmp[0] = C0[idx]; \
        tmp[1] = C1[idx]; \
        tmp[2] = C2[idx]; \
        tmp[3] = C3[idx]; \
    } while (0)

#define CONVERT_PACK(idx) \
    do { \
        DATA4_T tmp_cvt \
                = (DATA4_T)(CONVERT_DATA_T(tmp.s0), CONVERT_DATA_T(tmp.s1), \
                        CONVERT_DATA_T(tmp.s2), CONVERT_DATA_T(tmp.s3)); \
        dst_pack[idx] = as_uint(tmp_cvt); \
    } while (0)

#define STORE_DST(C0, C1, C2, C3, D, mb_stride) \
    do { \
        for (int n_i = 0; n_i < 8; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS(); \
            DO_ELTWISE(); \
            DO_SUM(D[n_i]); \
            DO_POST_SUM_ELTWISE(); \
            CONVERT_PACK(n_i); \
        } \
        intel_sub_group_block_write8( \
                (__global uint *)&dst[mb_stride * OC_BLOCK], dst_pack); \
    } while (0)

    if (ow < OW) {
#if WITH_SUM
        D0 = intel_sub_group_block_read8((__global uint *)dst);
#if MB > 8
        D1 = intel_sub_group_block_read8((__global uint *)&dst[8 * OC_BLOCK]);
#ifdef MB_FULL_BLOCK
        D2 = intel_sub_group_block_read8((__global uint *)&dst[16 * OC_BLOCK]);
        D3 = intel_sub_group_block_read8((__global uint *)&dst[24 * OC_BLOCK]);
#endif // MB_FULL_BLOCK
#endif // MB > 8
#endif

        STORE_DST(C00, C01, C02, C03, D0, 0);
#if MB > 8
        STORE_DST(C10, C11, C12, C13, D1, 8);
#ifdef MB_FULL_BLOCK
        STORE_DST(C20, C21, C22, C23, D2, 16);
        STORE_DST(C30, C31, C32, C33, D3, 24);
#endif // MB_FULL_BLOCK
#endif // MB > 8
    }
}

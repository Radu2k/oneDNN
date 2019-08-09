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

#if OW_BLOCK == 4
#define ACC_DATA_BLOCK int4
#define DATA_BLOCK uint4
#define READ_BLOCK intel_sub_group_block_read4
#define WRITE_LOCAL WRITE_LOCAL_4
#define WRITE_BLOCK intel_sub_group_block_write4
#define MMAD mmad8x4
#else
#define ACC_DATA_BLOCK int8
#define DATA_BLOCK uint8
#define READ_BLOCK intel_sub_group_block_read8
#define WRITE_LOCAL WRITE_LOCAL_8
#define WRITE_BLOCK intel_sub_group_block_write8
#define MMAD mmad8x8
#endif

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);
#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
conv_fwd_ow_block_u8s8s32x_kernel(const __global uchar *src,
        const __global char *wei, const __global float *bias,
        __global DATA_T *dst, float alpha, float beta, float sum_scale,
        float scales) {

    const int group_oc = get_group_id(0) * OC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP;
    const int group_sp = get_group_id(1) * SP_GROUP;
    const int sub_group_id = get_sub_group_id();
    const int oc = (sub_group_id % OC_GROUP);
    const int sp = (sub_group_id / OC_GROUP);
    const int g = (group_oc + oc) / OC_NCHUNK;
    const int group_ic = IC_NCHUNK * g;
    const int god = group_sp / (OW_PADDED * OH);
    const int gohw = group_sp % (OW_PADDED * OH);
    const int goh = gohw / OW_PADDED;
    const int gow = OW_BLOCK * (gohw % OW_PADDED);
    const int gid = god * SD;
    const int gih = goh * SH;
    const int giw = gow * SW;
    const int local_ow = OW_BLOCK * sp;
    const int local_iw = local_ow * SW;
    const int od = god;
    const int ow = gow + local_ow;
    const int oh = goh;
    const int id = gid - PD;
    const int iw = giw + local_iw - PW;
    const int ih = gih - PH;

    __local uint S_slice[IC_BLOCK / 4
            * (LWS_1 * SW * OW_BLOCK + (KW - 1) * (1 + DW))];
    __local uint *S_part = S_slice + IC_BLOCK / 4 * (sp * SW * OW_BLOCK + PW);
    __local uint *S_work = S_slice + IC_BLOCK / 4 * (sp * SW * OW_BLOCK);

    const bool left_tail = iw < 0;
    const bool left_nozero_tail = sub_group_id == 0 && iw >= 0;
    const bool right_tail = (iw + PW + OW_SLM_TAIL >= IW);
    const bool right_nozero_tail
            = sp == (LWS_1 - 1) && (iw + PW + OW_SLM_TAIL < IW);

    dst += OC_BLOCK * OD * OH * OW * MB_BLOCK * (group_oc + oc);
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK * group_mb;
    dst += OC_BLOCK * MB_BLOCK * (OW * OH * od + OW * oh + ow);
    src += IC_BLOCK * ID * IH * IW * MB_BLOCK * group_ic;
    src += IC_BLOCK * ID * IH * IW * IC_NCHUNK * G * MB_BLOCK * group_mb;
    src += IC_BLOCK * MB_BLOCK * (IW * IH * id + IW * ih + iw + PW);
    wei += IC_BLOCK * KD * KH * KW * OC_BLOCK * (group_oc + oc) * IC_NCHUNK;

    /* Prepare S_slice tails */
#if PW > 0
    if (left_tail) {
        for (int i = 0; i < PW; i++) {
            WRITE_LOCAL_1(S_slice + i * 8, 0);
        }
    }
#endif

#if ZERO_TAIL > 0
    if (right_tail) {
        for (int i = OW_SLM_TAIL; i < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW;
                i++) {
            WRITE_LOCAL_1(S_part + i * 8, 0);
        }
    }
#endif

    ACC_DATA_BLOCK C00 = 0, C01 = 0, C02 = 0, C03 = 0;

    __attribute__((opencl_unroll_hint(1))) for (int ic_chunk = 0;
                                                ic_chunk < IC_NCHUNK;
                                                ic_chunk++) {
        DATA_BLOCK S0;

        int8 W0, W1, W2, W3;
        __attribute__((opencl_unroll_hint(1))) for (int kd = 0; kd < KD; kd++) {
            if (kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID) {
                src += IC_BLOCK * MB_BLOCK * IH * IW * (1 + DD);
                wei += IC_BLOCK * OC_BLOCK * KH * KW;
                continue;
            }
            __attribute__((opencl_unroll_hint(1))) for (int kh = 0; kh < KH;
                                                        kh++) {
                if (kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH) {
                    src += IC_BLOCK * MB_BLOCK * IW * (1 + DH);
                    wei += IC_BLOCK * OC_BLOCK * KW;
                    continue;
                }

                barrier(CLK_LOCAL_MEM_FENCE);

#if OW_GROUP > LWS_1
                /* Copy tails in case of multigroups */
                if (ow < OW) {
#if PW > 0
                    if (left_nozero_tail) {
                        for (int i = -PW; i < 0; i++) {
                            WRITE_LOCAL_1(S_part + i * 8,
                                    intel_sub_group_block_read((
                                            const __global uint
                                                    *)(&src[i * IC_BLOCK])));
                        }
                    }
#endif

#if OW_SLM_TAIL > 0
                    if (right_nozero_tail) {
                        for (int i = SW * OW_BLOCK;
                                i < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW;
                                i++) {
                            WRITE_LOCAL_1(S_part + i * 8,
                                    intel_sub_group_block_read((
                                            const __global uint
                                                    *)(&src[i * IC_BLOCK])));
                        }
                    }
#endif
#endif

#if OW_SLM_TAIL != OW_BLOCK * SW
                    /* Copy last block to SLM */
                    if (right_tail) {
                        __attribute__((opencl_unroll_hint)) for (int i = 0; i
                                                                 < OW_SLM_TAIL;
                                                                 i++) {
                            WRITE_LOCAL_1(S_part + i * 8,
                                    intel_sub_group_block_read((
                                            const __global uint
                                                    *)(&src[i * IC_BLOCK])));
                        }
                    } else {
#endif
                        /* Copy block to SLM */
                        __attribute__((
                                opencl_unroll_hint)) for (int i = 0;
                                                          i < SW * OW_BLOCK;
                                                          i += OW_BLOCK) {
                            WRITE_LOCAL(S_part + i * 8,
                                    READ_BLOCK((const __global uint
                                                    *)(&src[i * IC_BLOCK])));
                        }

#if OW_SLM_TAIL != OW_BLOCK * SW
                    }
#endif

#if OW_GROUP > LWS_1
                }
#endif
                barrier(CLK_LOCAL_MEM_FENCE);

                __attribute__((opencl_unroll_hint)) for (int kw = 0; kw < KW;
                                                         kw++) {
                    __attribute__((opencl_unroll_hint(
                            OW_BLOCK))) for (int i = 0; i < OW_BLOCK; i++) {
                        S0[i] = READ_LOCAL_1(
                                S_work + (kw * (1 + DW) + SW * i) * 8);
                    }

                    BLOCK_READ_WHT(W0, 0);
                    BLOCK_READ_WHT(W1, 8 * IC_BLOCK);
                    BLOCK_READ_WHT(W2, 16 * IC_BLOCK);
                    BLOCK_READ_WHT(W3, 24 * IC_BLOCK);

                    C00 = MMAD(S0, W0, C00);
                    C01 = MMAD(S0, W1, C01);
                    C02 = MMAD(S0, W2, C02);
                    C03 = MMAD(S0, W3, C03);

                    wei += IC_BLOCK * OC_BLOCK;
                }
                src += IC_BLOCK * MB_BLOCK * IW * (1 + DH);
            }
            src += IC_BLOCK * MB_BLOCK * (IH * (1 + DD) - KH * (1 + DH)) * IW;
        }
        src += IC_BLOCK * MB_BLOCK * (ID - KD * (1 + DD)) * IH * IW;
    }

    if (ow < OW) {
        float4 tmp;

        DATA_BLOCK dst_pack;
        DATA_BLOCK D0, D1, D2, D3;

#if WITH_BIAS
        bias += (group_oc + oc) * OC_BLOCK + get_sub_group_local_id() * 4;
        float4 bia = (float4)(bias[0], bias[1], bias[2], bias[3]);
        bia *= scales;
#define QUANTIZE_ADD_BIAS() tmp = fma(tmp, (float4)scales, bia);
#else
#define QUANTIZE_ADD_BIAS() tmp *= scales;
#endif

#if WITH_SUM
        D0 = READ_BLOCK((__global uint *)dst);

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

#define PACK_DST(C0, C1, C2, C3, D) \
    do { \
        for (int n_i = 0; n_i < OW_BLOCK; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS(); \
            DO_ELTWISE(); \
            DO_SUM(D[n_i]); \
            DO_POST_SUM_ELTWISE(); \
            CONVERT_PACK(n_i); \
        } \
    } while (0)

        PACK_DST(C00, C01, C02, C03, D0);
#if OW_TAIL
        if (right_tail) {
            __attribute__((opencl_unroll_hint(OW_TAIL))) for (int i = 0;
                                                              i < OW_TAIL;
                                                              i++) {
                intel_sub_group_block_write(
                        (__global uint *)&dst[i * 32], dst_pack[i]);
            }
        } else {
#endif
            WRITE_BLOCK((__global uint *)&dst[0], dst_pack);
#if OW_TAIL
        }
#endif
    }
}

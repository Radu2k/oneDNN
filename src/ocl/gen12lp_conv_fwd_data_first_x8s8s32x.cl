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

#define KDHW_SIZE (KH * KW * KD)

#define GET_INT_BLOCK(SRC_SLM, SLM_INDEX, SRC_GLOBAL, GLOBAL_INDEX) \
    uchar4 res = 0; \
    for (int j = 0; j < IC; j++) { \
        res[j] = SRC_GLOBAL[GLOBAL_INDEX + j * IH * IW * ID]; \
    } \
    SRC_SLM[SLM_INDEX] = as_int(res);

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

#define BLOCK_READ_WHT(data, idx) \
    data = as_int(intel_sub_group_block_read((__global uint *)&wei[idx]));

#define BLOCK_READ_WHT8(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
conv_fwd_first_x8s8s32x_kernel(const __global uchar *src,
        const __global char *wei, const __global float *bias,
        __global DST_DATA_T *dst, float alpha, float beta, float sum_scale,
        float scales) {

    const int group_oc = get_group_id(0) * OC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP;
    const int group_sp = get_group_id(1) * SP_GROUP;
    const int sub_group_id = get_sub_group_id();
    const int sub_local_id = get_sub_group_local_id();
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

    __local uint S_slice[SRC_SLM_SIZE * KH * KD];
    __local uint *S_part = S_slice + (sp * SW * OW_BLOCK + PW);
    __local MMAD_DATA_T *S_work = S_slice + (sp * SW * OW_BLOCK);

    dst += OC_BLOCK * OD * OH * OW * MB_BLOCK * (group_oc + oc);
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK
            * (group_mb / MB_BLOCK);
    dst += OC_BLOCK * (group_mb % MB_BLOCK);
    dst += OC_BLOCK * MB_BLOCK * (OW * OH * od + OW * oh + ow);

    src += IC_BLOCK * ID * IH * IW * G * group_mb;
    src += IC_BLOCK * (IW * IH * id + IW * ih + iw + PW);

    wei += 4 * KDHW_SIZE * OC_BLOCK * (group_oc + oc);

    bias += (group_oc + oc) * OC_BLOCK;

    /* WORK WITH SLM */
    const bool left_tail = iw < 0;
    const bool left_nozero_tail = sub_group_id == 0 && iw >= 0;
    const bool right_tail = (iw + PW + OW_SLM_TAIL >= IW) && (iw + PW < IW);
    const bool empty = (iw + PW >= IW);
    const bool right_nozero_tail
            = sp == (LWS_1 - 1) && (iw + PW + OW_SLM_TAIL < IW);

    barrier(CLK_LOCAL_MEM_FENCE);
    /* KD */
#if KD > 1
    for (int kd = 0; kd < KD; kd++) {
        if (kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID) {
            S_part += SRC_SLM_SIZE * KH;
            src += IC_BLOCK * IW * IH * (1 + DD);
            continue;
        }
#endif
        /* KH */
#if KH > 1
        for (int kh = 0; kh < KH; kh++) {
            if (kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH) {
                S_part += SRC_SLM_SIZE;
                src += IC_BLOCK * IW * (1 + DH);
                continue;
            }
#endif
            /* KW */
            /* left tail */
#if PW > 0
            if (left_tail) {
                for (int i = -PW; i < 0; i++) {
                    S_part[i] = 0;
                }
            }
#endif
            /* right tail */
#if ZERO_TAIL > 0
            if (right_tail) {
                for (int i = OW_SLM_TAIL;
                        i < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW; i++) {
                    S_part[i] = 0;
                }
            }
#if SLM_WORKING_GROUPS < OW_NCHUNK
            if (empty) {
                for (int i = 0; i < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW;
                        i++) {
                    WRITE_LOCAL_1(S_part + i * 8, 0);
                }
            }
#endif
#endif
#if SLM_WORKING_GROUPS < OW_NCHUNK
            if (iw + PW < IW) {
#endif
#if OW_NCHUNK > LWS_1
                /* Copy tails in case of multigroups */
                if (ow < OW) {
#if PW > 0
                    if (left_nozero_tail) {
                        for (int i = -PW; i < 0; i++) {
#if NCHW == 1
                            GET_INT_BLOCK(S_part, i, src, i);

#else
                            S_part[i] = ((__global uint *)src)[i];
#endif
                        }
                    }
#endif

                    if (right_nozero_tail) {
                        for (int i = SW * OW_BLOCK;
                                i < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW;
                                i++) {
#if NCHW == 1
                            GET_INT_BLOCK(S_part, i, src, i);
#else
                            S_part[i] = ((__global uint *)src)[i];
#endif
                        }
                    }
#endif

#if OW_SLM_TAIL != OW_BLOCK * SW
                    /* Copy last block to SLM */
                    if (right_tail) {
                        __attribute__((opencl_unroll_hint)) for (int i = 0; i
                                                                 < OW_SLM_TAIL;
                                                                 i++) {
#if NCHW == 1
                            GET_INT_BLOCK(S_part, i, src, i);
#else
                            S_part[i] = ((__global uint *)src)[i];
#endif
                        }
                    } else {
#endif
#if (SW * OW_BLOCK) % 8 == 0
                        /* Copy block to SLM */
                        __attribute__((
                                opencl_unroll_hint)) for (int i = 0;
                                                          i < SW * OW_BLOCK;
                                                          i += 8) {
#if NCHW == 1
                            uchar4 res = 0;
                            for (int j = 0; j < IC; j++) {
                                res[j] = intel_sub_group_block_read_uc(
                                        src + i + j * IH * IW * ID);
                            }
                            WRITE_LOCAL_1(S_part + i, as_int(res));
#else
                            WRITE_LOCAL_1(S_part + i,
                                    intel_sub_group_block_read((
                                            const __global uint
                                                    *)(&src[i * IC_BLOCK])));
#endif
                        }
#elif (SW * OW_BLOCK) % 4 == 0 && NCHW == 0
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < SW * OW_BLOCK;
                                             i += 4) {
        WRITE_LOCAL_SHORT_1(S_part + i,
                intel_sub_group_block_read_us(
                        (const __global ushort *)(&src[i * IC_BLOCK])));
    }
#else
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < SW * OW_BLOCK;
                                             i++) {
        GET_INT_BLOCK(S_part, i, src, i);
    }
#endif

#if OW_SLM_TAIL != OW_BLOCK * SW
                    }
#endif

#if OW_NCHUNK > LWS_1
                }
#endif
#if SLM_WORKING_GROUPS < OW_NCHUNK
            }
#endif
#if KH > 1
            S_part += SRC_SLM_SIZE;
            src += IC_BLOCK * IW * (1 + DH);
        }
        S_part -= SRC_SLM_SIZE * KH;
        src -= IC_BLOCK * KH * IW * (1 + DH);
#endif
#if KD > 1
        S_part += SRC_SLM_SIZE * KH;
        src += IC_BLOCK * IW * IH * (1 + DD);
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

    MMAD_DATA8_T S;
    int8 W0, W1, W2, W3;
    int W00, W10, W20, W30;
    int8 C00 = 0;
    int8 C10 = 0;
    int8 C20 = 0;
    int8 C30 = 0;
#if OW_BLOCK == 12
    MMAD_DATA4_T SS;
    int4 C01 = 0;
    int4 C11 = 0;
    int4 C21 = 0;
    int4 C31 = 0;
#endif

#if OW_BLOCK == 16
    int8 C01 = 0;
    int8 C11 = 0;
    int8 C21 = 0;
    int8 C31 = 0;
#endif

    for (int i = 0; i < KDHW_SIZE - KDHW_SIZE % 8; i += 8) {
        const int ihw = (i + sub_local_id) % (KW * KH);
        const int filter_iw = (ihw % KW) * (1 + DW);
        const int filter_ih = ihw / KW;
        const int filter_id = (i + sub_local_id) / (KH * KW);
        const int filter = (filter_ih * (1 + DH) + ih >= 0)
                && (filter_ih * (1 + DH) + ih < IH)
                && (filter_id * (1 + DD) + id >= 0
                        && filter_id * (1 + DD) + id < ID);

        BLOCK_READ_WHT8(W0, 0);
        BLOCK_READ_WHT8(W1, KDHW_SIZE * OC_BLOCK);
        BLOCK_READ_WHT8(W2, 2 * KDHW_SIZE * OC_BLOCK);
        BLOCK_READ_WHT8(W3, 3 * KDHW_SIZE * OC_BLOCK);
        if (filter) {
            S.s0 = S_work[SW * 0 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s1 = S_work[SW * 1 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s2 = S_work[SW * 2 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s3 = S_work[SW * 3 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s4 = S_work[SW * 4 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s5 = S_work[SW * 5 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s6 = S_work[SW * 6 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s7 = S_work[SW * 7 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
#if OW_BLOCK == 12
            SS.s0 = S_work[SW * 8 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            SS.s1 = S_work[SW * 9 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            SS.s2 = S_work[SW * 10 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            SS.s3 = S_work[SW * 11 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
#endif
        } else {
            S = 0;
#if OW_BLOCK == 12
            SS = 0;
#endif
        }

        C00 = mmad8x8(S, W0, C00);
        C10 = mmad8x8(S, W1, C10);
        C20 = mmad8x8(S, W2, C20);
        C30 = mmad8x8(S, W3, C30);
#if OW_BLOCK == 12
        C01 = mmad8x4(SS, W0, C01);
        C11 = mmad8x4(SS, W1, C11);
        C21 = mmad8x4(SS, W2, C21);
        C31 = mmad8x4(SS, W3, C31);
#endif

#if OW_BLOCK == 16
        if (filter) {
            S.s0 = S_work[SW * 8 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s1 = S_work[SW * 9 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s2 = S_work[SW * 10 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s3 = S_work[SW * 11 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s4 = S_work[SW * 12 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s5 = S_work[SW * 13 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s6 = S_work[SW * 14 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s7 = S_work[SW * 15 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
        } else {
            S = 0;
        }

        C01 = mmad8x8(S, W0, C01);
        C11 = mmad8x8(S, W1, C11);
        C21 = mmad8x8(S, W2, C21);
        C31 = mmad8x8(S, W3, C31);
#endif
        wei += OC_BLOCK * 8;
    }

    for (int i = KDHW_SIZE - KDHW_SIZE % 8; i < KDHW_SIZE; i++) {
        const int ihw = (i) % (KW * KH);
        const int filter_iw = (ihw % KW) * (1 + DW);
        const int filter_ih = ihw / KW;
        const int filter_id = (i) / (KH * KW);
        const int filter = (filter_ih * (1 + DH) + ih >= 0)
                && (filter_ih * (1 + DH) + ih < IH)
                && (filter_id * (1 + DD) + id >= 0
                        && filter_id * (1 + DD) + id < ID);
        if (filter) {
            BLOCK_READ_WHT(W00, 0);
            BLOCK_READ_WHT(W10, KDHW_SIZE * OC_BLOCK);
            BLOCK_READ_WHT(W20, 2 * KDHW_SIZE * OC_BLOCK);
            BLOCK_READ_WHT(W30, 3 * KDHW_SIZE * OC_BLOCK);

            S.s0 = S_work[SW * 0 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s1 = S_work[SW * 1 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s2 = S_work[SW * 2 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s3 = S_work[SW * 3 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s4 = S_work[SW * 4 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s5 = S_work[SW * 5 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s6 = S_work[SW * 6 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s7 = S_work[SW * 7 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
#if OW_BLOCK == 12
            SS.s0 = S_work[SW * 8 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            SS.s1 = S_work[SW * 9 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            SS.s2 = S_work[SW * 10 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            SS.s3 = S_work[SW * 11 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
#endif
            C00.s0 = IMAD(AS_SRC_DATA4_T(S.s0), as_char4(W00), C00.s0);
            C00.s1 = IMAD(AS_SRC_DATA4_T(S.s1), as_char4(W00), C00.s1);
            C00.s2 = IMAD(AS_SRC_DATA4_T(S.s2), as_char4(W00), C00.s2);
            C00.s3 = IMAD(AS_SRC_DATA4_T(S.s3), as_char4(W00), C00.s3);
            C00.s4 = IMAD(AS_SRC_DATA4_T(S.s4), as_char4(W00), C00.s4);
            C00.s5 = IMAD(AS_SRC_DATA4_T(S.s5), as_char4(W00), C00.s5);
            C00.s6 = IMAD(AS_SRC_DATA4_T(S.s6), as_char4(W00), C00.s6);
            C00.s7 = IMAD(AS_SRC_DATA4_T(S.s7), as_char4(W00), C00.s7);

            C10.s0 = IMAD(AS_SRC_DATA4_T(S.s0), as_char4(W10), C10.s0);
            C10.s1 = IMAD(AS_SRC_DATA4_T(S.s1), as_char4(W10), C10.s1);
            C10.s2 = IMAD(AS_SRC_DATA4_T(S.s2), as_char4(W10), C10.s2);
            C10.s3 = IMAD(AS_SRC_DATA4_T(S.s3), as_char4(W10), C10.s3);
            C10.s4 = IMAD(AS_SRC_DATA4_T(S.s4), as_char4(W10), C10.s4);
            C10.s5 = IMAD(AS_SRC_DATA4_T(S.s5), as_char4(W10), C10.s5);
            C10.s6 = IMAD(AS_SRC_DATA4_T(S.s6), as_char4(W10), C10.s6);
            C10.s7 = IMAD(AS_SRC_DATA4_T(S.s7), as_char4(W10), C10.s7);

            C20.s0 = IMAD(AS_SRC_DATA4_T(S.s0), as_char4(W20), C20.s0);
            C20.s1 = IMAD(AS_SRC_DATA4_T(S.s1), as_char4(W20), C20.s1);
            C20.s2 = IMAD(AS_SRC_DATA4_T(S.s2), as_char4(W20), C20.s2);
            C20.s3 = IMAD(AS_SRC_DATA4_T(S.s3), as_char4(W20), C20.s3);
            C20.s4 = IMAD(AS_SRC_DATA4_T(S.s4), as_char4(W20), C20.s4);
            C20.s5 = IMAD(AS_SRC_DATA4_T(S.s5), as_char4(W20), C20.s5);
            C20.s6 = IMAD(AS_SRC_DATA4_T(S.s6), as_char4(W20), C20.s6);
            C20.s7 = IMAD(AS_SRC_DATA4_T(S.s7), as_char4(W20), C20.s7);

            C30.s0 = IMAD(AS_SRC_DATA4_T(S.s0), as_char4(W30), C30.s0);
            C30.s1 = IMAD(AS_SRC_DATA4_T(S.s1), as_char4(W30), C30.s1);
            C30.s2 = IMAD(AS_SRC_DATA4_T(S.s2), as_char4(W30), C30.s2);
            C30.s3 = IMAD(AS_SRC_DATA4_T(S.s3), as_char4(W30), C30.s3);
            C30.s4 = IMAD(AS_SRC_DATA4_T(S.s4), as_char4(W30), C30.s4);
            C30.s5 = IMAD(AS_SRC_DATA4_T(S.s5), as_char4(W30), C30.s5);
            C30.s6 = IMAD(AS_SRC_DATA4_T(S.s6), as_char4(W30), C30.s6);
            C30.s7 = IMAD(AS_SRC_DATA4_T(S.s7), as_char4(W30), C30.s7);
#if OW_BLOCK == 12
            C01.s0 = IMAD(AS_SRC_DATA4_T(SS.s0), as_char4(W00), C01.s0);
            C01.s1 = IMAD(AS_SRC_DATA4_T(SS.s1), as_char4(W00), C01.s1);
            C01.s2 = IMAD(AS_SRC_DATA4_T(SS.s2), as_char4(W00), C01.s2);
            C01.s3 = IMAD(AS_SRC_DATA4_T(SS.s3), as_char4(W00), C01.s3);

            C11.s0 = IMAD(AS_SRC_DATA4_T(SS.s0), as_char4(W10), C11.s0);
            C11.s1 = IMAD(AS_SRC_DATA4_T(SS.s1), as_char4(W10), C11.s1);
            C11.s2 = IMAD(AS_SRC_DATA4_T(SS.s2), as_char4(W10), C11.s2);
            C11.s3 = IMAD(AS_SRC_DATA4_T(SS.s3), as_char4(W10), C11.s3);

            C21.s0 = IMAD(AS_SRC_DATA4_T(SS.s0), as_char4(W20), C21.s0);
            C21.s1 = IMAD(AS_SRC_DATA4_T(SS.s1), as_char4(W20), C21.s1);
            C21.s2 = IMAD(AS_SRC_DATA4_T(SS.s2), as_char4(W20), C21.s2);
            C21.s3 = IMAD(AS_SRC_DATA4_T(SS.s3), as_char4(W20), C21.s3);

            C31.s0 = IMAD(AS_SRC_DATA4_T(SS.s0), as_char4(W30), C31.s0);
            C31.s1 = IMAD(AS_SRC_DATA4_T(SS.s1), as_char4(W30), C31.s1);
            C31.s2 = IMAD(AS_SRC_DATA4_T(SS.s2), as_char4(W30), C31.s2);
            C31.s3 = IMAD(AS_SRC_DATA4_T(SS.s3), as_char4(W30), C31.s3);
#endif

#if OW_BLOCK == 16
            S.s0 = S_work[SW * 8 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s1 = S_work[SW * 9 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s2 = S_work[SW * 10 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s3 = S_work[SW * 11 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s4 = S_work[SW * 12 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s5 = S_work[SW * 13 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s6 = S_work[SW * 14 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s7 = S_work[SW * 15 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];

            C01.s0 = IMAD(AS_SRC_DATA4_T(S.s0), as_char4(W00), C01.s0);
            C01.s1 = IMAD(AS_SRC_DATA4_T(S.s1), as_char4(W00), C01.s1);
            C01.s2 = IMAD(AS_SRC_DATA4_T(S.s2), as_char4(W00), C01.s2);
            C01.s3 = IMAD(AS_SRC_DATA4_T(S.s3), as_char4(W00), C01.s3);
            C01.s4 = IMAD(AS_SRC_DATA4_T(S.s4), as_char4(W00), C01.s4);
            C01.s5 = IMAD(AS_SRC_DATA4_T(S.s5), as_char4(W00), C01.s5);
            C01.s6 = IMAD(AS_SRC_DATA4_T(S.s6), as_char4(W00), C01.s6);
            C01.s7 = IMAD(AS_SRC_DATA4_T(S.s7), as_char4(W00), C01.s7);

            C11.s0 = IMAD(AS_SRC_DATA4_T(S.s0), as_char4(W10), C11.s0);
            C11.s1 = IMAD(AS_SRC_DATA4_T(S.s1), as_char4(W10), C11.s1);
            C11.s2 = IMAD(AS_SRC_DATA4_T(S.s2), as_char4(W10), C11.s2);
            C11.s3 = IMAD(AS_SRC_DATA4_T(S.s3), as_char4(W10), C11.s3);
            C11.s4 = IMAD(AS_SRC_DATA4_T(S.s4), as_char4(W10), C11.s4);
            C11.s5 = IMAD(AS_SRC_DATA4_T(S.s5), as_char4(W10), C11.s5);
            C11.s6 = IMAD(AS_SRC_DATA4_T(S.s6), as_char4(W10), C11.s6);
            C11.s7 = IMAD(AS_SRC_DATA4_T(S.s7), as_char4(W10), C11.s7);

            C21.s0 = IMAD(AS_SRC_DATA4_T(S.s0), as_char4(W20), C21.s0);
            C21.s1 = IMAD(AS_SRC_DATA4_T(S.s1), as_char4(W20), C21.s1);
            C21.s2 = IMAD(AS_SRC_DATA4_T(S.s2), as_char4(W20), C21.s2);
            C21.s3 = IMAD(AS_SRC_DATA4_T(S.s3), as_char4(W20), C21.s3);
            C21.s4 = IMAD(AS_SRC_DATA4_T(S.s4), as_char4(W20), C21.s4);
            C21.s5 = IMAD(AS_SRC_DATA4_T(S.s5), as_char4(W20), C21.s5);
            C21.s6 = IMAD(AS_SRC_DATA4_T(S.s6), as_char4(W20), C21.s6);
            C21.s7 = IMAD(AS_SRC_DATA4_T(S.s7), as_char4(W20), C21.s7);

            C31.s0 = IMAD(AS_SRC_DATA4_T(S.s0), as_char4(W30), C31.s0);
            C31.s1 = IMAD(AS_SRC_DATA4_T(S.s1), as_char4(W30), C31.s1);
            C31.s2 = IMAD(AS_SRC_DATA4_T(S.s2), as_char4(W30), C31.s2);
            C31.s3 = IMAD(AS_SRC_DATA4_T(S.s3), as_char4(W30), C31.s3);
            C31.s4 = IMAD(AS_SRC_DATA4_T(S.s4), as_char4(W30), C31.s4);
            C31.s5 = IMAD(AS_SRC_DATA4_T(S.s5), as_char4(W30), C31.s5);
            C31.s6 = IMAD(AS_SRC_DATA4_T(S.s6), as_char4(W30), C31.s6);
            C31.s7 = IMAD(AS_SRC_DATA4_T(S.s7), as_char4(W30), C31.s7);
#endif
        }
        wei += OC_BLOCK;
    }
    DST_DATA16_T R1, R2, R3, R4;

#if WITH_BIAS
    float4 bia = as_float4(intel_sub_group_block_read4((__global uint *)bias));
    bia *= scales;
#define QUANTIZE_ADD_BIAS() tmp = fma(tmp, (float4)scales, bia);
#define QUANTIZE_ADD_BIAS_4() \
    tmp0 = fma(tmp0, (float8)scales, bia.s01230123); \
    tmp1 = fma(tmp1, (float8)scales, bia.s01230123);
#else
#define QUANTIZE_ADD_BIAS() tmp *= scales;
#define QUANTIZE_ADD_BIAS_4() \
    tmp0 *= scales; \
    tmp1 *= scales;
#endif

#if WITH_SUM
#define DO_SUM() \
    do { \
        DST_DATA4_T d = AS_DST_DATA4_T(intel_sub_group_block_read_uc4(dst)); \
        float4 df = convert_float4(d); \
        tmp = fma(df, (float4)sum_scale, tmp); \
    } while (0)

#define DO_SUM_4() \
    do { \
        DST_DATA16_T d \
                = AS_DST_DATA16_T(intel_sub_group_block_read_uc16(dst)); \
        float8 df0 = convert_float8(d.s01234567); \
        float8 df1 = convert_float8(d.s89abcdef); \
        tmp0 = fma(df0, (float8)sum_scale, tmp0); \
        tmp1 = fma(df1, (float8)sum_scale, tmp1); \
    } while (0)
#else
#define DO_SUM() ;
#define DO_SUM_4() ;
#endif

#define ELTWISE() \
    do { \
        tmp[0] = fwd_eltwise(tmp[0], alpha, beta); \
        tmp[1] = fwd_eltwise(tmp[1], alpha, beta); \
        tmp[2] = fwd_eltwise(tmp[2], alpha, beta); \
        tmp[3] = fwd_eltwise(tmp[3], alpha, beta); \
    } while (0)

#define ELTWISE_4() \
    do { \
        tmp0.s0 = fwd_eltwise(tmp0.s0, alpha, beta); \
        tmp0.s1 = fwd_eltwise(tmp0.s1, alpha, beta); \
        tmp0.s2 = fwd_eltwise(tmp0.s2, alpha, beta); \
        tmp0.s3 = fwd_eltwise(tmp0.s3, alpha, beta); \
        tmp0.s4 = fwd_eltwise(tmp0.s4, alpha, beta); \
        tmp0.s5 = fwd_eltwise(tmp0.s5, alpha, beta); \
        tmp0.s6 = fwd_eltwise(tmp0.s6, alpha, beta); \
        tmp0.s7 = fwd_eltwise(tmp0.s7, alpha, beta); \
\
        tmp1.s0 = fwd_eltwise(tmp1.s0, alpha, beta); \
        tmp1.s1 = fwd_eltwise(tmp1.s1, alpha, beta); \
        tmp1.s2 = fwd_eltwise(tmp1.s2, alpha, beta); \
        tmp1.s3 = fwd_eltwise(tmp1.s3, alpha, beta); \
        tmp1.s4 = fwd_eltwise(tmp1.s4, alpha, beta); \
        tmp1.s5 = fwd_eltwise(tmp1.s5, alpha, beta); \
        tmp1.s6 = fwd_eltwise(tmp1.s6, alpha, beta); \
        tmp1.s7 = fwd_eltwise(tmp1.s7, alpha, beta); \
    } while (0)

#if WITH_ELTWISE
#define DO_ELTWISE() ELTWISE();
#define DO_ELTWISE_4() ELTWISE_4();
#else
#define DO_ELTWISE() ;
#define DO_ELTWISE_4() ;
#endif

#if WITH_POST_SUM_ELTWISE
#define DO_POST_SUM_ELTWISE() ELTWISE();
#define DO_POST_SUM_ELTWISE_4() ELTWISE_4();
#else
#define DO_POST_SUM_ELTWISE() ;
#define DO_POST_SUM_ELTWISE_4() ;
#endif

#define PACK(C0, C1, C2, C3, idx) \
    do { \
        tmp[0] = C0[idx]; \
        tmp[1] = C1[idx]; \
        tmp[2] = C2[idx]; \
        tmp[3] = C3[idx]; \
    } while (0)

#define PACK_4(C0, C1, C2, C3, idx) \
    do { \
        tmp0.s0 = C0[idx]; \
        tmp0.s1 = C1[idx]; \
        tmp0.s2 = C2[idx]; \
        tmp0.s3 = C3[idx]; \
\
        tmp0.s4 = C0[idx + 1]; \
        tmp0.s5 = C1[idx + 1]; \
        tmp0.s6 = C2[idx + 1]; \
        tmp0.s7 = C3[idx + 1]; \
\
        tmp1.s0 = C0[idx + 2]; \
        tmp1.s1 = C1[idx + 2]; \
        tmp1.s2 = C2[idx + 2]; \
        tmp1.s3 = C3[idx + 2]; \
\
        tmp1.s4 = C0[idx + 3]; \
        tmp1.s5 = C1[idx + 3]; \
        tmp1.s6 = C2[idx + 3]; \
        tmp1.s7 = C3[idx + 3]; \
    } while (0)

#define CONVERT_PACK() \
    do { \
        tmp_cvt = (DST_DATA4_T)(TO_DST(tmp.s0), TO_DST(tmp.s1), \
                TO_DST(tmp.s2), TO_DST(tmp.s3)); \
    } while (0)

#define CONVERT_PACK_4() \
    do { \
        R.s01234567 = TO_DST8(tmp0); \
        R.s89abcdef = TO_DST8(tmp1); \
    } while (0)

#define STORE_DST(C0, C1, C2, C3, i) \
    do { \
        PACK(C0, C1, C2, C3, i); \
        QUANTIZE_ADD_BIAS(); \
        DO_ELTWISE(); \
        DO_SUM(); \
        DO_POST_SUM_ELTWISE(); \
        CONVERT_PACK(); \
        intel_sub_group_block_write_uc4(dst, as_uchar4(tmp_cvt)); \
        dst += OC_BLOCK * MB_BLOCK; \
    } while (0)

#define STORE_DST_4(C0, C1, C2, C3, i) \
    do { \
        PACK_4(C0, C1, C2, C3, i); \
        QUANTIZE_ADD_BIAS_4(); \
        DO_ELTWISE_4(); \
        DO_SUM_4(); \
        DO_POST_SUM_ELTWISE_4(); \
        CONVERT_PACK_4(); \
        intel_sub_group_block_write_uc16(dst, as_uchar16(R)); \
        dst += 4 * OC_BLOCK; \
    } while (0)

    if (ow < OW) {
        float4 tmp;
        DST_DATA4_T tmp_cvt;
        float8 tmp0, tmp1;
        DST_DATA16_T R;

#if OW_TAIL
        if (ow + OW_BLOCK < OW) {
#endif
#if MB_BLOCK == 32
            STORE_DST(C00, C10, C20, C30, 0);
            STORE_DST(C00, C10, C20, C30, 1);
            STORE_DST(C00, C10, C20, C30, 2);
            STORE_DST(C00, C10, C20, C30, 3);

            STORE_DST(C00, C10, C20, C30, 4);
            STORE_DST(C00, C10, C20, C30, 5);
            STORE_DST(C00, C10, C20, C30, 6);
            STORE_DST(C00, C10, C20, C30, 7);
#if OW_BLOCK >= 12
            STORE_DST(C01, C11, C21, C31, 0);
            STORE_DST(C01, C11, C21, C31, 1);
            STORE_DST(C01, C11, C21, C31, 2);
            STORE_DST(C01, C11, C21, C31, 3);
#endif
#if OW_BLOCK == 16
            STORE_DST(C01, C11, C21, C31, 4);
            STORE_DST(C01, C11, C21, C31, 5);
            STORE_DST(C01, C11, C21, C31, 6);
            STORE_DST(C01, C11, C21, C31, 7);
#endif

#else
        STORE_DST_4(C00, C10, C20, C30, 0);
        STORE_DST_4(C00, C10, C20, C30, 4);
#if OW_BLOCK >= 12
        STORE_DST_4(C01, C11, C21, C31, 0);
#endif
#if OW_BLOCK >= 16
        STORE_DST_4(C01, C11, C21, C31, 4);
#endif
#endif
#if OW_TAIL
        } else {

#if OW_TAIL < 4
            for (int i = 0; i < OW_TAIL; i++) {
                STORE_DST(C00, C10, C20, C30, i);
            }
#else
#if MB_BLOCK == 32
            STORE_DST(C00, C10, C20, C30, 0);
            STORE_DST(C00, C10, C20, C30, 1);
            STORE_DST(C00, C10, C20, C30, 2);
            STORE_DST(C00, C10, C20, C30, 3);
#else
            STORE_DST_4(C00, C10, C20, C30, 0);
#endif
#endif
#if OW_TAIL > 4
#if OW_TAIL < 8
            for (int i = 4; i < OW_TAIL; i++) {
                STORE_DST(C00, C10, C20, C30, i);
            }
#else
#if MB_BLOCK == 32
            STORE_DST(C00, C10, C20, C30, 4);
            STORE_DST(C00, C10, C20, C30, 5);
            STORE_DST(C00, C10, C20, C30, 6);
            STORE_DST(C00, C10, C20, C30, 7);
#else
            STORE_DST_4(C00, C10, C20, C30, 4);
#endif
#endif
#if OW_TAIL > 8
#if OW_TAIL < 12
            for (int i = 8; i < OW_TAIL; i++) {
                STORE_DST(C01, C11, C21, C31, i);
            }
#else
#if MB_BLOCK == 32
            STORE_DST(C01, C11, C21, C31, 0);
            STORE_DST(C01, C11, C21, C31, 1);
            STORE_DST(C01, C11, C21, C31, 2);
            STORE_DST(C01, C11, C21, C31, 3);
#else
            STORE_DST_4(C01, C11, C21, C31, 0);
#endif
#endif
#if OW_TAIL > 12
#if OW_TAIL < 16
            for (int i = 12; i < OW_TAIL; i++) {
                STORE_DST(C01, C11, C21, C31, i);
            }
#else
#if MB_BLOCK == 32
            STORE_DST(C01, C11, C21, C31, 4);
            STORE_DST(C01, C11, C21, C31, 5);
            STORE_DST(C01, C11, C21, C31, 6);
            STORE_DST(C01, C11, C21, C31, 7);
#else
            STORE_DST_4(C01, C11, C21, C31, 4);
#endif
#endif
#endif
#endif
#endif
        }
#endif
    }
}

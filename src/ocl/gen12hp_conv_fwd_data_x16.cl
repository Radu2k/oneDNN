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

#include "ocl/ocl_types.h"
#if WITH_ELTWISE == 1 || WITH_POST_SUM_ELTWISE == 1
#include "ocl/ocl_post_ops.h"
#endif

#if SLM_WEI
#define WEI wei_tmp
#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(READ_LOCAL_8((__local uint *)&wei_tmp[idx]));
#else
#define WEI wei
#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));
#endif

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
gen12hp_conv_fwd_x16_kernel(const __global DATA_T *src,
        const __global DATA_T *wei, const __global BIA_DATA_T *bias,
        __global DST_DATA_T *dst, float eltwise_alpha, float eltwise_beta,
        float sum_scale, float scales) {
    const int group_oc = get_group_id(0) * OC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP;
    const int group_sp = get_group_id(1) * SP_GROUP;

    const int sub_group_id = get_sub_group_id();
    const int oc = (sub_group_id % OC_GROUP);
    const int sp = (sub_group_id / OC_GROUP);

    const int g = (group_oc + oc) * (OC_CALC_BLOCK / OC_BLOCK) / OC_NCHUNK;
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

#if !SLM_WEI
    if (ow >= OW) return;
#endif // SLM_WEI

    dst += OC_CALC_BLOCK * MB_BLOCK * OD * OH * OW * (group_oc + oc);
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK * group_mb;
    dst += OC_BLOCK * MB_BLOCK * (OW * OH * od + OW * oh + ow);

    src += IC_BLOCK * ID * IH * IW * MB_BLOCK * group_ic;
    src += IC_BLOCK * ID * IH * IW * IC_NCHUNK * G * MB_BLOCK * group_mb;
    src += IC_BLOCK * MB_BLOCK * (IW * IH * id + IW * ih + iw);

    wei += WEI_BLOCK * KD * KH * KW * (group_oc + oc) * IC_NCHUNK;

    float8 C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    float8 C10 = 0, C11 = 0, C12 = 0, C13 = 0;
    float8 C20 = 0, C21 = 0, C22 = 0, C23 = 0;
    float8 C30 = 0, C31 = 0, C32 = 0, C33 = 0;

#if SLM_WEI
    __local DATA_T wei_loc[KW * OC_GROUP * WEI_BLOCK];
    __local DATA_T *wei_loc_base = wei_loc + KW * WEI_BLOCK * oc;
#endif // SLM_WEI

    __attribute__((opencl_unroll_hint(1))) for (int ic_chunk = 0;
                                                ic_chunk < IC_NCHUNK;
                                                ic_chunk++) {
        uint8 S0, S1, S2, S3;
        int8 W0, W1, W2, W3;
        __attribute__((opencl_unroll_hint(1))) for (int kd = 0; kd < KD; kd++) {
            if (kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID) {
                src += IC_BLOCK * MB_BLOCK * IH * IW * (1 + DD);
                wei += WEI_BLOCK * KH * KW;
                continue;
            }
            __attribute__((opencl_unroll_hint)) for (int kh = 0; kh < KH;
                                                     kh++) {
                if (kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH) {
                    src += IC_BLOCK * MB_BLOCK * IW * (1 + DH);
                    wei += WEI_BLOCK * KW;
                    continue;
                }

#if SLM_WEI
                barrier(CLK_LOCAL_MEM_FENCE);
                const __global DATA_T *wei_copy_from
                        = wei + sp * KW * WEI_BLOCK / LWS_1;
                __local DATA_T *wei_copy_to
                        = wei_loc_base + sp * KW * WEI_BLOCK / LWS_1;
                for (int bl = 0; bl < KW; bl++) {
                    WRITE_LOCAL_4(
                            (__local uint *)&wei_copy_to[bl * 4 * IC_BLOCK],
                            intel_sub_group_block_read4(
                                    (__global uint *)&wei_copy_from[bl * 4
                                            * IC_BLOCK]));
                }
                __local DATA_T *wei_tmp = wei_loc_base;
                barrier(CLK_LOCAL_MEM_FENCE);
#endif // SLM_WEI

                __attribute__((opencl_unroll_hint)) for (int kw = 0; kw < KW;
                                                         kw++) {
                    if (kw * (1 + DW) + iw >= 0 && kw * (1 + DW) + iw < IW) {
                        BLOCK_READ_SRC(S0, 0);
#if MB > 8
                        BLOCK_READ_SRC(S1, 8 * IC_BLOCK);
#if MB > 16
                        BLOCK_READ_SRC(S2, 16 * IC_BLOCK);
#if MB > 24
                        BLOCK_READ_SRC(S3, 24 * IC_BLOCK);
#endif // MB > 24
#endif // MB > 16
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
#if MB > 16
                        C20 = MMAD8X8(S2, W0, C20);
                        C21 = MMAD8X8(S2, W1, C21);
                        C22 = MMAD8X8(S2, W2, C22);
                        C23 = MMAD8X8(S2, W3, C23);
#if MB > 24
                        C30 = MMAD8X8(S3, W0, C30);
                        C31 = MMAD8X8(S3, W1, C31);
                        C32 = MMAD8X8(S3, W2, C32);
                        C33 = MMAD8X8(S3, W3, C33);
#endif // MB > 24
#endif // MB > 16
#endif // MB > 8
                    }
                    src += IC_BLOCK * MB_BLOCK * (1 + DW);
                    WEI += WEI_BLOCK;
                }
#if SLM_WEI
                wei += WEI_BLOCK * KW;
#endif // SLM_WEI
                src += IC_BLOCK * MB_BLOCK * (IW * (1 + DH) - KW * (1 + DW));
            }
            src += IC_BLOCK * MB_BLOCK * (IH * (1 + DD) - KH * (1 + DH)) * IW;
        }
        src += IC_BLOCK * MB_BLOCK * (ID - KD * (1 + DD)) * IH * IW;
    }

#if WITH_BIAS
    bias += (group_oc + oc) * OC_CALC_BLOCK
            + get_sub_group_local_id() * OC_CALC_BLOCK / OC_BLOCK;
    float4 bia = (float4)(BIA_TO_REF(bias[0]), BIA_TO_REF(bias[1]),
            BIA_TO_REF(bias[16]), BIA_TO_REF(bias[17]));
    bia *= scales;
#define QUANTIZE_ADD_BIAS() tmp = fma(tmp, (float4)scales, bia);
#else
#define QUANTIZE_ADD_BIAS() tmp *= scales;
#endif

#if WITH_SUM
#define DO_SUM(d_pack0, d_pack1) \
    do { \
        DST_DATA2_T d0 = AS_DST_DATA2_T(d_pack0); \
        DST_DATA2_T d1 = AS_DST_DATA2_T(d_pack1); \
        tmp.s01 = fma(DST_TO_REF2(d0), (float2)sum_scale, tmp.s01); \
        tmp.s23 = fma(DST_TO_REF2(d1), (float2)sum_scale, tmp.s23); \
    } while (0)
#else
#define DO_SUM(d_pack0, d_pack1) ;
#endif // WITH_SUM

#define ELTWISE() \
    do { \
        tmp[0] = fwd_eltwise(tmp[0], eltwise_alpha, eltwise_beta); \
        tmp[1] = fwd_eltwise(tmp[1], eltwise_alpha, eltwise_beta); \
        tmp[2] = fwd_eltwise(tmp[2], eltwise_alpha, eltwise_beta); \
        tmp[3] = fwd_eltwise(tmp[3], eltwise_alpha, eltwise_beta); \
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
        DST_DATA2_T tmp_cvt0 \
                = (DST_DATA2_T)(REF_TO_DST(tmp.s0), REF_TO_DST(tmp.s1)); \
        dst_pack0[idx] = AS_DST_PACK(tmp_cvt0); \
        DST_DATA2_T tmp_cvt1 \
                = (DST_DATA2_T)(REF_TO_DST(tmp.s2), REF_TO_DST(tmp.s3)); \
        dst_pack1[idx] = AS_DST_PACK(tmp_cvt1); \
    } while (0)

#define STORE_DST(C0, C1, C2, C3, D0, D1) \
    do { \
        for (int n_i = 0; n_i < 8; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS(); \
            DO_ELTWISE(); \
            DO_SUM(D0[n_i], D1[n_i]); \
            DO_POST_SUM_ELTWISE(); \
            CONVERT_PACK(n_i); \
        } \
        BLOCK_WRITE_DST((__global uint *)&dst[0], dst_pack0); \
        __global DST_DATA_T *dst1 = dst + OC_BLOCK * MB_BLOCK * OD * OH * OW; \
        BLOCK_WRITE_DST((__global uint *)&dst1[0], dst_pack1); \
    } while (0)

    if (ow < OW) {
        float4 tmp;

#if DST_DT_F32
        ulong8 dst_pack0, dst_pack1;
        ulong8 D00, D01, D02, D03;
        ulong8 D10, D11, D12, D13;
#define AS_DST_PACK as_ulong
#define BLOCK_WRITE_DST intel_sub_group_block_write_ul8
#define BLOCK_READ_DST intel_sub_group_block_read_ul8
#else
        uint8 dst_pack0, dst_pack1;
        uint8 D00, D01, D02, D03;
        uint8 D10, D11, D12, D13;
#define AS_DST_PACK as_uint
#define BLOCK_WRITE_DST intel_sub_group_block_write8
#define BLOCK_READ_DST intel_sub_group_block_read8
#endif

#if WITH_SUM
        D00 = BLOCK_READ_DST((__global uint *)dst);
        D10 = BLOCK_READ_DST(
                (__global uint *)&dst[OC_BLOCK * MB_BLOCK * OD * OH * OW]);
#if MB > 8
        D01 = BLOCK_READ_DST((__global uint *)&dst[8 * OC_BLOCK]);
        D11 = BLOCK_READ_DST((__global uint *)&dst[8 * OC_BLOCK
                + OC_BLOCK * MB_BLOCK * OD * OH * OW]);
#if MB > 16
        D02 = BLOCK_READ_DST((__global uint *)&dst[16 * OC_BLOCK]);
        D12 = BLOCK_READ_DST((__global uint *)&dst[16 * OC_BLOCK
                + OC_BLOCK * MB_BLOCK * OD * OH * OW]);
#if MB > 24
        D03 = BLOCK_READ_DST((__global uint *)&dst[24 * OC_BLOCK]);
        D13 = BLOCK_READ_DST((__global uint *)&dst[24 * OC_BLOCK
                + OC_BLOCK * MB_BLOCK * OD * OH * OW]);
#endif // MB > 24
#endif // MB > 16
#endif // MB > 8
#endif

        STORE_DST(C00, C01, C02, C03, D00, D10);
#if MB > 8
        dst += 8 * OC_BLOCK;
        STORE_DST(C10, C11, C12, C13, D01, D11);
#if MB > 16
        dst += 8 * OC_BLOCK;
        STORE_DST(C20, C21, C22, C23, D02, D12);
#if MB > 24
        dst += 8 * OC_BLOCK;
        STORE_DST(C30, C31, C32, C33, D03, D13);
#endif // MB > 24
#endif // MB > 16
#endif // MB > 8
    }
}

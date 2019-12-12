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

#define DST_DATA_BLOCK_T MMAD_DATA8_T
#define AS_DST_DATA_BLOCK_T AS_MMAD_DATA8_T

#if SLM_WEI
#define WEI wei_tmp
#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(READ_LOCAL_8((__local uint *)&WEI[idx]));
#else
#define WEI wei
#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&WEI[idx]));
#endif

#define BLOCK_READ_DST(data, idx) \
    data = AS_DST_DATA_BLOCK_T( \
            intel_sub_group_block_read8((__global uint *)&current_dst[idx]));

#if BIA_DT_F32
#define BLOCK_READ_BIA(data, idx) \
    data = as_float4(intel_sub_group_block_read4((__global uint *)&bias[idx]));
#elif BIA_DT_F16
#define BLOCK_READ_BIA(data, idx) \
    data = convert_float4(as_half4( \
            intel_sub_group_block_read_us4((__global ushort *)&bias[idx])));
#elif BIA_DT_BF16
#define BLOCK_READ_BIA(data, idx) \
    data = convert_bf16_to_f32_vec4( \
            intel_sub_group_block_read_us4((__global ushort *)&bias[idx]));
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
gen12hp_conv_bwd_data_kernel(__global SRC_DATA_T *src,
        const __global WEI_DATA_T *wei, const __global BIA_DATA_T *bias,
        const __global DST_DATA_T *dst) {
    const int group_ic = get_group_id(0) * IC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP;
    const int group_sp = get_group_id(1) * SP_GROUP;

    const int sub_group_id = get_sub_group_id();
    const int ic = (sub_group_id % IC_GROUP);
    const int sp = (sub_group_id / IC_GROUP);

    const int g = (group_ic + ic) * (IC_CALC_BLOCK / IC_BLOCK) / IC_NCHUNK;
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

#if SLM_WEI == 0
    if (iw >= IW) return;
#endif // SLM_WEI

    src += IC_CALC_BLOCK * ID * IH * IW * MB_BLOCK * (group_ic + ic);
    src += IC_BLOCK * ID * IH * IW * IC_NCHUNK * G * MB_BLOCK * group_mb;
    src += IC_BLOCK * MB_BLOCK * (IW * IH * id + IW * ih + iw);

    dst += OC_BLOCK * OD * OH * OW * MB_BLOCK * group_oc;
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK * group_mb;

    wei += WEI_BLOCK * KD * KH * KW * (group_ic + ic) * OC_NCHUNK;

    MMAD_ACC_DATA8_T C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    MMAD_ACC_DATA8_T C10 = 0, C11 = 0, C12 = 0, C13 = 0;
    MMAD_ACC_DATA8_T C20 = 0, C21 = 0, C22 = 0, C23 = 0;
    MMAD_ACC_DATA8_T C30 = 0, C31 = 0, C32 = 0, C33 = 0;

#if SLM_WEI

#if KW == 1 && KH == 1 && KD == 1
// Enable double buffer
#define IND_GROUP 2
#define READ_SLM() \
    barrier(CLK_LOCAL_MEM_FENCE); \
    if (wei_last > wei) { \
        const __global WEI_DATA_T *wei_copy_from \
                = wei + sp * KW * WEI_BLOCK / LWS_1; \
        __local WEI_DATA_T *wei_copy_to = wei_loc_base \
                + sp * KW * WEI_BLOCK / LWS_1 + read_ind * KW * WEI_BLOCK; \
        for (int bl = 0; bl < KW; bl++) { \
            WRITE_LOCAL_4((__local uint *)&wei_copy_to[bl * 4 * IC_BLOCK], \
                    intel_sub_group_block_read4((__global uint \
                                    *)&wei_copy_from[bl * 4 * IC_BLOCK])); \
        } \
    } \
    __local WEI_DATA_T *wei_tmp = wei_loc_base + calc_ind * KW * WEI_BLOCK; \
    wei += WEI_BLOCK * KW; \
    read_ind ^= 1; \
    calc_ind ^= 1;

#else
#define IND_GROUP 1
#define READ_SLM() \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const __global WEI_DATA_T *wei_copy_from \
            = wei + sp * KW * WEI_BLOCK / LWS_1; \
    __local WEI_DATA_T *wei_copy_to \
            = wei_loc_base + sp * KW * WEI_BLOCK / LWS_1; \
    for (int bl = 0; bl < KW; bl++) { \
        WRITE_LOCAL_4((__local uint *)&wei_copy_to[bl * 4 * IC_BLOCK], \
                intel_sub_group_block_read4( \
                        (__global uint *)&wei_copy_from[bl * 4 * IC_BLOCK])); \
    } \
    __local WEI_DATA_T *wei_tmp = wei_loc_base; \
    wei += WEI_BLOCK * KW; \
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    __local WEI_DATA_T wei_loc[KW * IC_GROUP * WEI_BLOCK * IND_GROUP];
    __local WEI_DATA_T *wei_loc_base = wei_loc + KW * WEI_BLOCK * ic;
    const __global WEI_DATA_T *wei_last
            = wei + WEI_BLOCK * KD * KH * KW * OC_NCHUNK;
#endif // SLM_WEI
    bool run = true;
#if KD == 1
    const int od = (id + PD) / SD;
#if SD != 1
    run = (((id + PD) % SD == 0) && (od >= 0) && (od < OD)) && run;
#endif
#endif
#if KH == 1
    const int oh = (ih + PH) / SH;
#if SH != 1
    run = (((ih + PH) % SH == 0) && (oh >= 0) && (oh < OH)) && run;
#endif
#endif
#if KW == 1
    const int ow = (iw + PW) / SW;
#if SW != 1
    run = (((iw + PW) % SW == 0) && (ow >= 0) && (ow < OW)) && run;
#endif
#endif
    int read_ind = 0;
    int calc_ind = 1;

#if SLM_WEI && KW == 1 && KH == 1 && KD == 1
    READ_SLM()
#endif
    for (int oc_chunk = 0; oc_chunk < OC_NCHUNK; oc_chunk++) {
        DST_DATA_BLOCK_T D0, D1, D2, D3;
        int8 W0, W1, W2, W3;
#if KD != 1
        for (int kd = 0; kd < KD; kd++) {
            if ((id + PD - kd * (1 + DD)) % SD != 0) {
                wei += WEI_BLOCK * KH * KW;
                continue;
            }
            const int od = (id + PD - kd * (1 + DD)) / SD;
            if (od < 0 || od >= OD) {
                wei += WEI_BLOCK * KH * KW;
                continue;
            }
#endif // KD != 1
#if KH != 1
            for (int kh = 0; kh < KH; kh++) {
                if ((ih + PH - kh * (1 + DH)) % SH != 0) {
                    wei += WEI_BLOCK * KW;
                    continue;
                }
                const int oh = (ih + PH - kh * (1 + DH)) / SH;
                if (oh < 0 || oh >= OH) {
                    wei += WEI_BLOCK * KW;
                    continue;
                }
#endif // KH != 1

#if SLM_WEI
                READ_SLM()
#endif // SLM_WEI
#if KW != 1
                __attribute__((opencl_unroll_hint)) for (int kw = 0; kw < KW;
                                                         kw++) {
                    if ((iw + PW - kw * (1 + DW)) % SW == 0) {
                        const int ow = (iw + PW - kw * (1 + DW)) / SW;
                        if (run && ow >= 0 && ow < OW) {
#else
        if (run) {
#endif // KW != 1
                            __global DST_DATA_T *current_dst = dst
                                    + OC_BLOCK * MB_BLOCK
                                            * (OW * OH * od + OW * oh + ow);
                            BLOCK_READ_DST(D0, 0);
#if MB > 8
                            BLOCK_READ_DST(D1, 8 * OC_BLOCK);
#if MB > 16
                            BLOCK_READ_DST(D2, 16 * OC_BLOCK);
#if MB > 24
                            BLOCK_READ_DST(D3, 24 * OC_BLOCK);
#endif // MB > 24
#endif // MB > 16
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
#if MB > 16
                            C20 = MMAD8X8(D2, W0, C20);
                            C21 = MMAD8X8(D2, W1, C21);
                            C22 = MMAD8X8(D2, W2, C22);
                            C23 = MMAD8X8(D2, W3, C23);
#if MB > 24
                            C30 = MMAD8X8(D3, W0, C30);
                            C31 = MMAD8X8(D3, W1, C31);
                            C32 = MMAD8X8(D3, W2, C32);
                            C33 = MMAD8X8(D3, W3, C33);
#endif // MB > 24
#endif // MB > 16
#endif // MB > 8
                        }

#if KW != 1
                    }
#endif
                    WEI += WEI_BLOCK;
#if KW != 1
                }
#endif
#if KH != 1
            }
#endif
#if KD != 1
        }
#endif
        dst += OC_BLOCK * MB_BLOCK * OD * OH * OW;
    }

#if WITH_BIAS
    float4 bia;
    BLOCK_READ_BIA(bia, (group_ic + ic) * IC_CALC_BLOCK);
#define QUANTIZE_ADD_BIAS() tmp += bia;
#else
#define QUANTIZE_ADD_BIAS()
#endif

#define PACK(C0, C1, C2, C3, idx) \
    do { \
        tmp[0] = C0[idx]; \
        tmp[1] = C1[idx]; \
        tmp[2] = C2[idx]; \
        tmp[3] = C3[idx]; \
    } while (0)

#if DT_F16 || DT_BF16
#define CONVERT_PACK(idx) \
    do { \
        SRC_DATA2_T tmp_cvt0 = (SRC_DATA2_T)(TO_SRC(tmp.s0), TO_SRC(tmp.s1)); \
        src_pack0[idx] = as_uint(tmp_cvt0); \
        SRC_DATA2_T tmp_cvt1 = (SRC_DATA2_T)(TO_SRC(tmp.s2), TO_SRC(tmp.s3)); \
        src_pack1[idx] = as_uint(tmp_cvt1); \
    } while (0)

#define WRITE_SRC() \
    do { \
        BLOCK_WRITE_SRC(src, src_pack0.s0123, 0) \
        BLOCK_WRITE_SRC(src, src_pack0.s4567, 4 * IC_BLOCK) \
        __global SRC_DATA_T *src1 = src + IC_BLOCK * MB_BLOCK * ID * IH * IW; \
        BLOCK_WRITE_SRC(src1, src_pack1.s0123, 0) \
        BLOCK_WRITE_SRC(src1, src_pack1.s4567, 4 * IC_BLOCK) \
    } while (0)
#else
#define CONVERT_PACK(idx) \
    do { \
        SRC_DATA4_T tmp_cvt = (SRC_DATA4_T)(TO_SRC(tmp.s0), TO_SRC(tmp.s1), \
                TO_SRC(tmp.s2), TO_SRC(tmp.s3)); \
        src_pack0[idx] = as_uint(tmp_cvt); \
    } while (0)

#define WRITE_SRC() \
    do { \
        BLOCK_WRITE_SRC(src, src_pack0.s0123, 0) \
        BLOCK_WRITE_SRC(src, src_pack0.s4567, 4 * IC_BLOCK) \
    } while (0)
#endif

#define STORE_SRC(C0, C1, C2, C3) \
    do { \
        for (int n_i = 0; n_i < 8; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS(); \
            CONVERT_PACK(n_i); \
        } \
        WRITE_SRC(); \
    } while (0)

    if (iw < IW) {
        float4 tmp;

#if DT_F16 || DT_BF16
        uint8 src_pack0, src_pack1;
#define BLOCK_WRITE_SRC(p, data, idx) \
    intel_sub_group_block_write_us8( \
            (__global ushort *)&p[idx], as_ushort8(data));
#else
        uint8 src_pack0;
#define BLOCK_WRITE_SRC(p, data, idx) \
    intel_sub_group_block_write_uc16( \
            (__global uchar *)&p[idx], as_uchar16(data));
#endif

        STORE_SRC(C00, C01, C02, C03);
#if MB > 8
        src += 8 * IC_BLOCK;
        STORE_SRC(C10, C11, C12, C13);
#if MB > 16
        src += 8 * IC_BLOCK;
        STORE_SRC(C20, C21, C22, C23);
#if MB > 24
        src += 8 * IC_BLOCK;
        STORE_SRC(C30, C31, C32, C33);
#endif // MB > 24
#endif // MB > 16
#endif // MB > 8
    }
}

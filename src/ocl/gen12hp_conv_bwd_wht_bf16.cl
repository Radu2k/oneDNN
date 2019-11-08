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

#define write_local8(mem_loc, mem_glob) \
    do { \
        uint8 tmp = intel_sub_group_block_read8(mem_glob); \
        WRITE_LOCAL_8(mem_loc, tmp); \
    } while (0)

#define write_local4(mem_loc, mem_glob) \
    do { \
        uint4 tmp = intel_sub_group_block_read4(mem_glob); \
        WRITE_LOCAL_4(mem_loc, tmp); \
    } while (0)

#define read_dst_bias(mem_glob) \
    do { \
        uint4 tmp = intel_sub_group_block_read4(mem_glob); \
        if (compute_bias) { \
            for (int i = 0; i < 4; ++i) { \
                ushort2 tu = as_ushort2(tmp[i]); \
                BIAS_ACC[0] += convert_bf16_to_f32(tu.s0); \
                BIAS_ACC[1] += convert_bf16_to_f32(tu.s1); \
            } \
        } \
    } while (0)

#define write_dst_bias(mem_loc, mem_glob) \
    do { \
        uint4 tmp = intel_sub_group_block_read4(mem_glob); \
        if (compute_bias) { \
            for (int i = 0; i < 4; ++i) { \
                ushort2 tu = as_ushort2(tmp[i]); \
                BIAS_ACC[0] += convert_bf16_to_f32(tu.s0); \
                BIAS_ACC[1] += convert_bf16_to_f32(tu.s1); \
            } \
        } \
        WRITE_LOCAL_4(mem_loc, tmp); \
    } while (0)

#define INTS_PER_READ (8 * SUB_GROUP_SIZE)
#define INTS_PER_READ_DST (INTS_PER_READ / 2)

#if IC_BLK_UNROLL == 8
#define write_src write_local8
#define INTS_PER_READ_SRC INTS_PER_READ

#elif IC_BLK_UNROLL == 4
#define write_src write_local4
#define INTS_PER_READ_SRC (INTS_PER_READ / 2)

#else
#error Unexpected IC_BLK_UNROLL
#endif

#if WITH_BIAS
#define write_dst write_dst_bias
#else
#define write_dst write_local4
#endif

#define GEMM_IC_blk(o, i) \
    do { \
        ACC[o][2 * i] \
                = MMAD8X8(as_uint8(D[o]), as_int8(S[i][0]), ACC[o][2 * i]); \
        ACC[o][2 * i + 1] = MMAD8X8( \
                as_uint8(D[o]), as_int8(S[i][1]), ACC[o][2 * i + 1]); \
    } while (0)

#define TRANSPOSE_DST() \
    do { \
        D_tmp[0] = vload8( \
                2 * sg_loc_id, &diff_dst_loc_read[loc_dst_slice_idx]); \
        D_tmp[1] = vload8( \
                2 * sg_loc_id, &diff_dst_loc_read[loc_dst_slice_idx + 8]); \
        for (int i = 0; i < 8; i++) { \
            ushort2 tn0 = as_ushort2(D_tmp[0][i]); \
            ushort2 tn1 = as_ushort2(D_tmp[1][i]); \
            D[i / 4][(2 * i) % 8] = as_uint((ushort2)(tn0.s0, tn1.s0)); \
            D[i / 4][(2 * i) % 8 + 1] = as_uint((ushort2)(tn0.s1, tn1.s1)); \
        } \
    } while (0)

#define TRANSPOSE_SRC(i_c) \
    do { \
        S_tmp[0] = READ_LOCAL_8( \
                &src_loc_read[loc_src_slice_idx + 2 * i_c * INTS_PER_READ]); \
        S_tmp[1] = READ_LOCAL_8(&src_loc_read[loc_src_slice_idx \
                + (2 * i_c + 1) * INTS_PER_READ]); \
        for (int i = 0; i < 8; i++) { \
            ushort2 tn0 = as_ushort2(S_tmp[i / 4][(2 * i) % 8]); \
            ushort2 tn1 = as_ushort2(S_tmp[i / 4][(2 * i) % 8 + 1]); \
            S[i_c][0][i] = as_uint((ushort2)(tn0.s0, tn1.s0)); \
            S[i_c][1][i] = as_uint((ushort2)(tn0.s1, tn1.s1)); \
        } \
    } while (0)

#if MB_BLK_UNROLL == 2
#define WRITE_TO_SLM() \
    do { \
        write_src(&src_loc_write[sg_id * INTS_PER_READ_SRC], src_read); \
        write_src(&src_loc_write[src_slm_offset + sg_id * INTS_PER_READ_SRC], \
                &src_read[src_nblock_off]); \
        write_dst(&diff_dst_loc_write[sg_id * INTS_PER_READ_DST], \
                diff_dst_read); \
        write_dst(&diff_dst_loc_write[dst_slm_offset \
                          + sg_id * INTS_PER_READ_DST], \
                &diff_dst_read[dst_nblock_off]); \
    } while (0)
#else
#define WRITE_TO_SLM() \
    do { \
        write_src(&src_loc_write[sg_id * INTS_PER_READ_SRC], src_read); \
        write_dst(&diff_dst_loc_write[sg_id * INTS_PER_READ_DST], \
                diff_dst_read); \
    } while (0)
#endif

#define WRITE_SECOND_SLM_BUFFER() \
    do { \
        src_loc_write = &src_slm[(k_blk_iter % 2) * (SRC_SLM_SIZE / 2)]; \
        diff_dst_loc_write \
                = &diff_dst_slm[(k_blk_iter % 2) * (DST_SLM_SIZE / 2)]; \
        ow++; \
        if (ow == ow_end) { \
            ow = ow_start; \
            oh++; \
        } \
        if (oh == oh_end) { \
            oh = oh_start; \
            od++; \
        } \
        if (od == od_end) { \
            od = od_start; \
            n += MB_BLK_UNROLL * MB_BLOCK; \
        } \
        src_read = &src[SRC_OFF(n, 0, SD * od - id_pad, SH * oh - ih_pad, \
                                SW * ow - iw_pad) \
                / 2]; \
        diff_dst_read = &diff_dst[DST_OFF(n, 0, od, oh, ow) / 2]; \
        WRITE_TO_SLM(); \
    } while (0)

#if IC_BLK_UNROLL == 8
#define COMPUTE() \
    do { \
        TRANSPOSE_SRC(0); \
        GEMM_IC_blk(0, 0); \
        GEMM_IC_blk(1, 0); \
        TRANSPOSE_SRC(1); \
        GEMM_IC_blk(0, 1); \
        GEMM_IC_blk(1, 1); \
    } while (0)
#elif IC_BLK_UNROLL == 4
#define COMPUTE() \
    do { \
        TRANSPOSE_SRC(0); \
        GEMM_IC_blk(0, 0); \
        GEMM_IC_blk(1, 0); \
    } while (0)
#endif

#define COMPUTE_HALF_MB_BLOCK() \
    do { \
        TRANSPOSE_DST(); \
        COMPUTE(); \
    } while (0)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
gen12hp_conv_bwd_wht_kernel_bf16(const __global uint *src,
        __global WEI_DATA_T *diff_wei, __global uint *diff_bias,
        const __global uint *diff_dst) {

    const int group_oc = get_group_id(0) * OC_BLK_UNROLL * OC_BLOCK;
    const int group_ic = get_group_id(1) * IC_BLK_UNROLL * IC_BLOCK;
    const int group_g = get_group_id(2) / (KD * KH * KW);
    const int group_kd = (get_group_id(2) % (KD * KH * KW)) / (KH * KW);
    const int group_kh = (get_group_id(2) % (KH * KW)) / KW;
    const int group_kw = get_group_id(2) % KW;
    const int sg_id = get_sub_group_id();
    const int sg_loc_id = get_sub_group_local_id();

    const int id_pad = PD - group_kd * (1 + DD);
    const int ih_pad = PH - group_kh * (1 + DH);
    const int iw_pad = PW - group_kw * (1 + DW);

    const int od_start = max(0, (id_pad + SD - 1) / SD);
    const int oh_start = max(0, (ih_pad + SH - 1) / SH);
    const int ow_start = max(0, (iw_pad + SW - 1) / SW);

    const int id_start = od_start * SD - id_pad;
    const int ih_start = oh_start * SH - ih_pad;
    const int iw_start = ow_start * SW - iw_pad;

    const int id_pad_end = PD_R - (KD - 1 - group_kd) * (DD + 1);
    const int ih_pad_end = PH_R - (KH - 1 - group_kh) * (DH + 1);
    const int iw_pad_end = PW_R - (KW - 1 - group_kw) * (DW + 1);

    const int od_end = OD - max(0, (id_pad_end + SD - 1) / SD);
    const int oh_end = OH - max(0, (ih_pad_end + SH - 1) / SH);
    const int ow_end = OW - max(0, (iw_pad_end + SW - 1) / SW);

    const int k_blocks = (od_end - od_start) * (ow_end - ow_start)
            * (oh_end - oh_start)
            * ((MB + (MB_BLK_UNROLL * MB_BLOCK - 1))
                    / (MB_BLK_UNROLL * MB_BLOCK));

    const int k_blk_unrolls = k_blocks / K_UNROLL;
    const int k_blk_tails = k_blocks % K_UNROLL;

#if IC_BLK_UNROLL == 8
    src += SRC_OFF((sg_id % 2) * 8,
                   group_g * IC + group_ic + (sg_id / 2) * IC_BLOCK, 0, 0, 0)
            / 2;
#elif IC_BLK_UNROLL == 4
    src += SRC_OFF((sg_id % 4) * 4,
                   group_g * IC + group_ic + (sg_id / 4) * IC_BLOCK, 0, 0, 0)
            / 2;
#endif

    diff_wei += WHT_OFF(group_g, group_oc + (sg_id % 4) * OC_BLOCK,
            group_ic + (sg_id / 4) * (IC_BLK_UNROLL / 4) * IC_BLOCK, group_kd,
            group_kh, group_kw);

    diff_dst += DST_OFF((sg_id % 4) * 4,
                        group_g * OC + group_oc + (sg_id / 4) * OC_BLOCK, 0, 0,
                        0)
            / 2;

#if WITH_BIAS
    bool compute_bias
            = group_kh == 0 && group_kw == 0 && group_kd == 0 && group_ic == 0;
    __local uint bias_slm[OC_BLK_UNROLL * OC_BLOCK * 4];
    __local uint *bias_loc_write = &bias_slm[sg_id * OC_BLOCK];
    float2 BIAS_ACC = 0.0;
    if (compute_bias && sg_id % 4 == 0) {
#if BIA_DT_BF16
        diff_bias += (group_g * OC + group_oc + (sg_id / 4) * OC_BLOCK) / 2;
#else
        diff_bias += (group_g * OC + group_oc + (sg_id / 4) * OC_BLOCK);
#endif
    }
#endif

    uint8 S[2][2], S_tmp[2];

    uint8 D[2], D_tmp[2];

    float8 ACC[2][4];

    uint8 tmp[2];

    for (int i_o = 0; i_o < 2; i_o++)
        for (int i_i = 0; i_i < 4; i_i++) {
            ACC[i_o][i_i] = 0.0;
        }

    __local uint src_slm[SRC_SLM_SIZE];
    __local uint diff_dst_slm[DST_SLM_SIZE];

    int src_slm_offset = MB_BLOCK * IC_BLK_UNROLL * IC_BLOCK / 2;
    int dst_slm_offset = MB_BLOCK * OC_BLK_UNROLL * OC_BLOCK / 2;

    const int loc_src_slice_idx
            = sg_id / 4 * (MB_BLOCK * (IC_BLK_UNROLL / 4) * IC_BLOCK / 2);

    const int loc_dst_slice_idx = (sg_id % 4) * MB_BLOCK * OC_BLOCK / 2;

    const int src_nblock_off
            = SRC_OFF(MB_BLOCK, 0, 0, 0, 0) / 2; // 2 = num of DATA_T per int
    const int dst_nblock_off = DST_OFF(MB_BLOCK, 0, 0, 0, 0) / 2;

    __local uint *src_loc_write = src_slm;
    __local uint *diff_dst_loc_write = diff_dst_slm;
    const __local uint *src_loc_read = src_slm;
    const __local uint *diff_dst_loc_read = diff_dst_slm;

    const __global uint *src_read
            = &src[SRC_OFF(0, 0, id_start, ih_start, iw_start) / 2];
    const __global uint *diff_dst_read
            = &diff_dst[DST_OFF(0, 0, od_start, oh_start, ow_start) / 2];

    WRITE_TO_SLM();

    barrier(CLK_LOCAL_MEM_FENCE);

    int n = 0;
    int od = od_start;
    int oh = oh_start;
    int ow = ow_start;

    int k_blk_iter = 0;

    __attribute__((opencl_unroll_hint(1))) // attr:no-format
    for (int ki = 0; ki < k_blk_unrolls; ++ki) {
        __attribute__((opencl_unroll_hint(K_UNROLL))) // attr:no-format
        for (int ur = 0; ur < K_UNROLL; ++ur) {

            src_loc_read = &src_slm[(k_blk_iter % 2)
                    * (SRC_SLM_SIZE / 2)]; // Fixme give name to magic number
            diff_dst_loc_read
                    = &diff_dst_slm[(k_blk_iter % 2) * (DST_SLM_SIZE / 2)];

            k_blk_iter++;

            COMPUTE_HALF_MB_BLOCK();

            if (k_blk_iter < k_blocks) { WRITE_SECOND_SLM_BUFFER(); }

#if MB_BLK_UNROLL == 2
            src_loc_read += src_slm_offset;
            diff_dst_loc_read += dst_slm_offset;

            COMPUTE_HALF_MB_BLOCK();
#endif

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for (int ti = 0; ti < k_blk_tails; ti++) {
        src_loc_read = &src_slm[(k_blk_iter % 2) * SRC_SLM_SIZE
                / 2]; // Fixme give name to magic number 2= num_buffers
        diff_dst_loc_read = &diff_dst_slm[(k_blk_iter % 2) * DST_SLM_SIZE / 2];

        k_blk_iter++;

        COMPUTE_HALF_MB_BLOCK();

        if (k_blk_iter < k_blocks) { WRITE_SECOND_SLM_BUFFER(); }

#if MB_BLK_UNROLL == 2
        src_loc_read += src_slm_offset;
        diff_dst_loc_read += dst_slm_offset;

        COMPUTE_HALF_MB_BLOCK();
#endif

        barrier(CLK_LOCAL_MEM_FENCE);
    }

#if WEI_DT_BF16
#define WRITE_WEI(i_o, i_i) \
    do { \
        for (int i = 0; i < 8; ++i) { \
            tmp[i_i][i] = as_uint( \
                    (ushort2)(convert_f32_to_bf16(ACC[i_o][2 * i_i][i]), \
                            convert_f32_to_bf16(ACC[i_o][2 * i_i + 1][i]))); \
        } \
        intel_sub_group_block_write8( \
                (__global uint *)&diff_wei[WHT_OFF( \
                        0, i_o * 8, i_i * IC_BLOCK, 0, 0, 0)], \
                tmp[i_i]); \
    } while (0)
#else // wei data type is f32
    const int delta_up
            = ((sg_loc_id + 1) % 2) * (SUB_GROUP_SIZE + sg_loc_id / 2)
            + (sg_loc_id % 2) * (sg_loc_id - sg_loc_id / 2);
    const int delta_down
            = ((sg_loc_id + 1) % 2) * (SUB_GROUP_SIZE / 2 - sg_loc_id / 2)
            + (sg_loc_id % 2)
                    * (3 * SUB_GROUP_SIZE / 2 + sg_loc_id / 2 - sg_loc_id);
#define WRITE_WEI(i_o, i_i) \
    do { \
        tmp[0] = as_uint8(intel_sub_group_shuffle_up( \
                ACC[i_o][2 * i_i], ACC[i_o][2 * i_i + 1], delta_up)); \
        tmp[1] = as_uint8(intel_sub_group_shuffle_down( \
                ACC[i_o][2 * i_i], ACC[i_o][2 * i_i + 1], delta_down)); \
        intel_sub_group_block_write8( \
                (__global uint *)&diff_wei[WHT_OFF( \
                        0, i_o * 8, i_i * IC_BLOCK, 0, 0, 0)], \
                tmp[0]); \
        intel_sub_group_block_write8( \
                (__global uint *)&diff_wei[WHT_OFF( \
                        0, i_o * 8, i_i * IC_BLOCK + 8, 0, 0, 0)], \
                tmp[1]); \
    } while (0)
#endif

    WRITE_WEI(0, 0);
    WRITE_WEI(1, 0);
#if IC_BLK_UNROLL == 8
    WRITE_WEI(0, 1);
    WRITE_WEI(1, 1);
#endif

#if WITH_BIAS
    if (compute_bias) {
        // tail processing,
        // when either of od_start, oh_start, ow_start > 0
        const int num_mb_blks = (MB + MB_BLOCK - 1) / MB_BLOCK;
        for (int n = 0; n < num_mb_blks; n++) {
            for (int oh = 0; oh < oh_start; oh++) {
                for (int od = 0; od < OD; od++) {
                    for (int ow = 0; ow < OW; ow++) {
                        read_dst_bias(
                                &diff_dst[DST_OFF(n * MB_BLOCK, 0, od, oh, ow)
                                        / 2]);
                    }
                }
            }
        }
        for (int n = 0; n < num_mb_blks; n++) {
            for (int oh = oh_start; oh < OH; oh++) {
                for (int od = 0; od < od_start; od++) {
                    for (int ow = 0; ow < OW; ow++) {
                        read_dst_bias(
                                &diff_dst[DST_OFF(n * MB_BLOCK, 0, od, oh, ow)
                                        / 2]);
                    }
                }
            }
        }
        for (int n = 0; n < num_mb_blks; n++) {
            for (int oh = oh_start; oh < OH; oh++) {
                for (int od = od_start; od < OD; od++) {
                    for (int ow = 0; ow < ow_start; ow++) {
                        read_dst_bias(
                                &diff_dst[DST_OFF(n * MB_BLOCK, 0, od, oh, ow)
                                        / 2]);
                    }
                }
            }
        }

        WRITE_LOCAL_2(bias_loc_write, as_uint2(BIAS_ACC));
        barrier(CLK_LOCAL_MEM_FENCE);

        if (sg_id % 4 == 0) {
            float8 btmp = as_float8(READ_LOCAL_8(bias_loc_write));
            for (int ni = 1; ni < 4; ++ni) {
                BIAS_ACC[0] += btmp[2 * ni];
                BIAS_ACC[1] += btmp[2 * ni + 1];
            }
#if BIA_DT_BF16
            uint tmp = as_uint((ushort2)(convert_f32_to_bf16(BIAS_ACC.s0),
                    convert_f32_to_bf16(BIAS_ACC.s1)));
            intel_sub_group_block_write(diff_bias, tmp);
#else
            diff_bias[2 * sg_loc_id] = as_uint(BIAS_ACC[0]);
            diff_bias[2 * sg_loc_id + 1] = as_uint(BIAS_ACC[1]);
#endif
        }
    }
#endif
}

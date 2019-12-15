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

#define USHORT_PER_READ (16 * SUB_GROUP_SIZE)
#define INT_PER_READ (USHORT_PER_READ / 2)

// Using hard-code strides instead of SRC_OFF/DST_OFF/WHT_OFF
// because compiler generates ugly code for SRC_OFF
#define SRC_W_STRIDE (2 * MB_BLOCK * IC_BLOCK)
#define SRC_H_STRIDE (IW * SRC_W_STRIDE)
#define SRC_D_STRIDE (IH * SRC_H_STRIDE)
#define SRC_C_STRIDE (ID * SRC_D_STRIDE)
#define SRC_MB_STRIDE (G * IC / IC_BLOCK * SRC_C_STRIDE)

#define DST_W_STRIDE (2 * MB_BLOCK * OC_BLOCK)
#define DST_H_STRIDE (OW * DST_W_STRIDE)
#define DST_D_STRIDE (OH * DST_H_STRIDE)
#define DST_C_STRIDE (OD * DST_D_STRIDE)
#define DST_MB_STRIDE (G * OC / OC_BLOCK * DST_C_STRIDE)

#if WEI_DT_BF16
#define WEI_W_STRIDE (8 * 8 / 2)
#else
#define WEI_W_STRIDE (8 * 8)
#endif
#define WEI_H_STRIDE (KW * WEI_W_STRIDE)
#define WEI_D_STRIDE (KH * WEI_H_STRIDE)
#define WEI_IC_STRIDE (KD * WEI_D_STRIDE)
#define WEI_OC_STRIDE (IC / 8 * WEI_D_STRIDE)
#define WEI_G_STRIDE (OC / 8 * WEI_OC_STRIDE)

#define GEMM_IC_blk(o, i) \
    do { \
        ACC[o][2 * i] \
                = MMAD8X8(as_uint8(D[o]), as_int8(S[i][0]), ACC[o][2 * i]); \
        ACC[o][2 * i + 1] = MMAD8X8( \
                as_uint8(D[o]), as_int8(S[i][1]), ACC[o][2 * i + 1]); \
    } while (0)

#define READ_DST() \
    do { \
        D[0] = READ_LOCAL_8(&diff_dst_loc_read[loc_dst_slice_idx]); \
        D[1] = READ_LOCAL_8( \
                &diff_dst_loc_read[loc_dst_slice_idx + INT_PER_READ]); \
        D[2] = READ_LOCAL_8( \
                &diff_dst_loc_read[loc_dst_slice_idx + 2 * INT_PER_READ]); \
        D[3] = READ_LOCAL_8( \
                &diff_dst_loc_read[loc_dst_slice_idx + 3 * INT_PER_READ]); \
    } while (0)

#define READ_SRC(i_c) \
    do { \
        S[i_c][0] = READ_LOCAL_8( \
                &src_loc_read[loc_src_slice_idx + 2 * i_c * INT_PER_READ]); \
        S[i_c][1] = READ_LOCAL_8(&src_loc_read[loc_src_slice_idx \
                + (2 * i_c + 1) * INT_PER_READ]); \
    } while (0)

#define PACK(i) as_uint((short2)(D_tmp[0][i], D_tmp[1][i]))

#if WITH_BIAS
#define CONVERT_TO_F32(x) convert_bf16_to_f32(x)

#define WRITE_DST() \
    do { \
        dst_off = (size_t)n_block * DST_MB_STRIDE + od * DST_D_STRIDE \
                + oh * DST_H_STRIDE + ow * DST_W_STRIDE; \
        Dt[0] = __builtin_IB_simd_block_read_16_global_h(&diff_dst[dst_off]); \
        Dt[1] = __builtin_IB_simd_block_read_16_global_h( \
                &diff_dst[dst_off + USHORT_PER_READ]); \
        BIAS_ACC[0] += (CONVERT_TO_F32(Dt[0].s0) + CONVERT_TO_F32(Dt[1].s0) \
                + CONVERT_TO_F32(Dt[0].s2) + CONVERT_TO_F32(Dt[1].s2) \
                + CONVERT_TO_F32(Dt[0].s4) + CONVERT_TO_F32(Dt[1].s4) \
                + CONVERT_TO_F32(Dt[0].s6) + CONVERT_TO_F32(Dt[1].s6) \
                + CONVERT_TO_F32(Dt[0].s8) + CONVERT_TO_F32(Dt[1].s8) \
                + CONVERT_TO_F32(Dt[0].sa) + CONVERT_TO_F32(Dt[1].sa) \
                + CONVERT_TO_F32(Dt[0].sc) + CONVERT_TO_F32(Dt[1].sc) \
                + CONVERT_TO_F32(Dt[0].se) + CONVERT_TO_F32(Dt[1].se)); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s0) \
                + CONVERT_TO_F32(Dt[1].odd.s0); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s1) \
                + CONVERT_TO_F32(Dt[1].odd.s1); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s2) \
                + CONVERT_TO_F32(Dt[1].odd.s2); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s3) \
                + CONVERT_TO_F32(Dt[1].odd.s3); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s4) \
                + CONVERT_TO_F32(Dt[1].odd.s4); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s5) \
                + CONVERT_TO_F32(Dt[1].odd.s5); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s6) \
                + CONVERT_TO_F32(Dt[1].odd.s6); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s7) \
                + CONVERT_TO_F32(Dt[1].odd.s7); \
        vstore16((ushort16)(Dt[0].even, Dt[1].even), sg_loc_id, \
                diff_dst_loc_write[k_blk_iter % 2]); \
        vstore16((ushort16)(Dt[0].odd, Dt[1].odd), sg_loc_id + 8, \
                diff_dst_loc_write[k_blk_iter % 2]); \
    } while (0)

#else //WITHOUT  BIAS
#define WRITE_DST() \
    do { \
        dst_off = (size_t)n_block * DST_MB_STRIDE + od * DST_D_STRIDE \
                + oh * DST_H_STRIDE + ow * DST_W_STRIDE; \
        Dt[0] = __builtin_IB_simd_block_read_16_global_h(&diff_dst[dst_off]); \
        Dt[1] = __builtin_IB_simd_block_read_16_global_h( \
                &diff_dst[dst_off + USHORT_PER_READ]); \
        vstore16((ushort16)(Dt[0].even, Dt[1].even), sg_loc_id, \
                diff_dst_loc_write[k_blk_iter % 2]); \
        vstore16((ushort16)(Dt[0].odd, Dt[1].odd), sg_loc_id + 8, \
                diff_dst_loc_write[k_blk_iter % 2]); \
    } while (0)
#endif // WITH_BIAS

#define WRITE_SRC() \
    do { \
        src_off = (size_t)n_block * SRC_MB_STRIDE + id * SRC_D_STRIDE \
                + ih * SRC_H_STRIDE + iw * SRC_W_STRIDE; \
        Dt[0] = __builtin_IB_simd_block_read_16_global_h(&src[src_off]); \
        Dt[1] = __builtin_IB_simd_block_read_16_global_h( \
                &src[src_off + USHORT_PER_READ]); \
        WRITE_LOCAL_8(src_loc_write[k_blk_iter % 2], \
                (uint8)(as_uint(Dt[0].s02), as_uint(Dt[0].s46), \
                        as_uint(Dt[0].s8A), as_uint(Dt[0].sCE), \
                        as_uint(Dt[1].s02), as_uint(Dt[1].s46), \
                        as_uint(Dt[1].s8A), as_uint(Dt[1].sCE))); \
        WRITE_LOCAL_8(&src_loc_write[k_blk_iter % 2][INT_PER_READ], \
                (uint8)(as_uint(Dt[0].s13), as_uint(Dt[0].s57), \
                        as_uint(Dt[0].s9B), as_uint(Dt[0].sDF), \
                        as_uint(Dt[1].s13), as_uint(Dt[1].s57), \
                        as_uint(Dt[1].s9B), as_uint(Dt[1].sDF))); \
    } while (0)

#if OC_BLK_UNROLL == 8
#define COMPUTE(i_c) \
    do { \
        READ_SRC(i_c); \
        GEMM_IC_blk(0, i_c); \
        GEMM_IC_blk(1, i_c); \
        GEMM_IC_blk(2, i_c); \
        GEMM_IC_blk(3, i_c); \
    } while (0)
#elif OC_BLK_UNROLL == 4
#define COMPUTE(i_c) \
    do { \
        READ_SRC(i_c); \
        GEMM_IC_blk(0, i_c); \
        GEMM_IC_blk(1, i_c); \
    } while (0)
#else
#error UNEXPECTED OC_BLK_UNROLL
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
gen12hp_conv_bwd_wht_kernel_bf16(const __global ushort *src,
        __global float *diff_wei, __global float *diff_bias,
        const __global ushort *diff_dst) {

    const int gid[3] = {get_group_id(0), get_group_id(1), get_group_id(2)};
    const int sg_id = get_sub_group_id();
    const int sg_loc_id = get_sub_group_local_id();

    const int sgid_mod_2 = sg_id % 2;
    const int sgid_div_2 = sg_id / 2;

    const int sgid_mod_4 = sg_id % 4;
    const int sgid_div_4 = sg_id / 4;

    const int group_ic = gid[0] * IC_BLK_UNROLL;
    const int group_oc = gid[1] * OC_BLK_UNROLL;

    const int group_g = (gid[2] / K_WORKGROUPS) / (KD * KH * KW);
    const int group_k_block = (gid[2] % K_WORKGROUPS) * K_BLOCKS;
    const int kd = (gid[2] / K_WORKGROUPS / KH / KW) % KD;
    const int kh = (gid[2] / K_WORKGROUPS / KW) % KH;
    const int kw = (gid[2] / K_WORKGROUPS) % KW;

    const int od_start = max((PD - kd * (1 + DD) + SD - 1) / SD, 0);
    const int oh_start = max((PH - kh * (1 + DH) + SH - 1) / SH, 0);
    const int ow_start = max((PW - kw * (1 + DW) + SW - 1) / SW, 0);

    const int od_end
            = OD - max(0, (PD_R - (KD - 1 - kd) * (1 + DD) + SD - 1) / SD) - 1;
    const int oh_end
            = OH - max(0, (PH_R - (KH - 1 - kh) * (1 + DH) + SH - 1) / SH) - 1;
    const int ow_end
            = OW - max(0, (PW_R - (KW - 1 - kw) * (1 + DW) + SW - 1) / SW) - 1;

    const int total_od = od_end - od_start + 1;
    const int total_oh = oh_end - oh_start + 1;
    const int total_ow = ow_end - ow_start + 1;
    const int total_k_blocks = (MB / (MB_BLK_UNROLL * MB_BLOCK)) * total_od
            * total_oh * total_ow;

    const int max_k_blocks = min(K_BLOCKS, total_k_blocks - group_k_block);

    int od = od_start + ((group_k_block / total_ow / total_oh) % total_od);
    int oh = oh_start + ((group_k_block / total_ow) % total_oh);
    int ow = ow_start + (group_k_block % total_ow);

    int n_block = group_k_block / (total_od * total_oh * total_ow);

    const int group_id = od * SD - PD + kd * (1 + DD);
    const int group_ih = oh * SH - PH + kh * (1 + DH);
    const int group_iw = ow * SW - PW + kw * (1 + DW);
    int id = group_id;
    int ih = group_ih;
    int iw = group_iw;
#if IC_BLK_UNROLL == 4
    if (sg_id < 8)
#endif
        src += sgid_mod_2 * MB_BLOCK * IC_BLOCK
                + (group_g * IC / IC_BLOCK + group_ic + sgid_div_2)
                        * SRC_C_STRIDE;
#if OC_BLK_UNROLL == 4
    if (sg_id < 8)
#endif
        diff_dst += sgid_mod_2 * MB_BLOCK * OC_BLOCK
                + (group_g * OC / OC_BLOCK + group_oc + sgid_div_2)
                        * DST_C_STRIDE;
    const int wei_off = WHT_OFF(group_g,
            (group_oc + (sgid_mod_4) * (OC_BLK_UNROLL / 4)) * OC_BLOCK,
            (group_ic + (sgid_div_4) * (IC_BLK_UNROLL / 4)) * IC_BLOCK, kd, kh,
            kw);
    diff_wei += WHT_OFF(group_g,
            (group_oc + (sgid_mod_4) * (OC_BLK_UNROLL / 4)) * OC_BLOCK,
            (group_ic + (sgid_div_4) * (IC_BLK_UNROLL / 4)) * IC_BLOCK, kd, kh,
            kw);

#if WITH_BIAS
    float2 BIAS_ACC = 0.0f;
    bool compute_bias = group_ic == 0 && kd == min(PD, KD - 1)
            && kh == min(PH, KH - 1) && kw == min(PW, KW - 1);
#if OC_BLK_UNROLL == 4
    compute_bias &= sg_id < 8;
#endif
    size_t bia_off;
    volatile __global atomic_float *dbias;
#if OC_BLK_UNROLL == 4
    bia_off = group_g * OC + (group_oc + (sg_id % 8) / 2) * OC_BLOCK;
#else
    bia_off = group_g * OC + (group_oc + sg_id / 2) * OC_BLOCK;
#endif
    dbias = (volatile __global atomic_float *)&diff_bias[bia_off];
#endif // WITH_BIAS

    uint8 S[2][2];

    uint8 D[4];
    ushort16 Dt[2];

    float8 ACC[4][4] = {0.0f};

    __local uint src_slm[SRC_SLM_SIZE];
    __local uint diff_dst_slm[DST_SLM_SIZE];

    int src_slm_offset = MB_BLOCK * IC_BLK_UNROLL * IC_BLOCK / 2;
    int dst_slm_offset = MB_BLOCK * OC_BLK_UNROLL * OC_BLOCK / 2;

    const int loc_src_slice_idx
            = sgid_div_4 * (MB_BLOCK * (IC_BLK_UNROLL / 4) * IC_BLOCK / 2);

    const int loc_dst_slice_idx
            = (sgid_mod_4)*MB_BLOCK * (OC_BLK_UNROLL / 4) * OC_BLOCK / 2;

    const int src_loc_offset
            = sgid_div_2 * USHORT_PER_READ + sgid_mod_2 * src_slm_offset;
    __local uint *src_loc_write[2] = {&src_slm[src_loc_offset],
            &src_slm[SRC_SLM_SIZE / 2 + src_loc_offset]};

    const int dst_loc_offset
            = sgid_div_2 * USHORT_PER_READ + sgid_mod_2 * dst_slm_offset;
    __local ushort *diff_dst_loc_write[2] = {
            (__local ushort *)&diff_dst_slm[dst_loc_offset],
            (__local ushort *)&diff_dst_slm[DST_SLM_SIZE / 2 + dst_loc_offset]};

    const __local uint *src_loc_read = src_slm;
    const __local uint *diff_dst_loc_read = diff_dst_slm;

    int k_blk_iter = 0;

    size_t src_off, dst_off;

    if (max_k_blocks > 0) {
#if IC_BLK_UNROLL == 4
        if (sg_id < 8)
#endif
            WRITE_SRC();
#if OC_BLK_UNROLL == 4
        if (sg_id < 8)
#endif
            WRITE_DST();
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint(1))) // attr:no-format
    for (int k_blk = 0; k_blk < max_k_blocks; ++k_blk) {

        src_loc_read = &src_slm[(k_blk_iter % 2) * (SRC_SLM_SIZE / 2)];
        diff_dst_loc_read
                = &diff_dst_slm[(k_blk_iter % 2) * (DST_SLM_SIZE / 2)];

        ow++;
        iw += SW;
        if (ow == ow_end + 1) {
            ow = max((PW - kw * (1 + DW) + SW - 1) / SW, 0);
            oh++;
            iw = ow * SW - PW + kw * (1 + DW);
            ih += SH;
        }
        if (oh == oh_end + 1) {
            oh = max((PH - kh * (1 + DH) + SH - 1) / SH, 0);
            od++;
            ih = oh * SH - PH + kh * (1 + DH);
            id += SD;
        }
        if (od == od_end + 1) {
            od = max((PD - kd * (1 + DD) + SD - 1) / SD, 0);
            id = od * SD - PD + kd * (1 + DD);
            n_block++;
        }

        k_blk_iter++;

        // Read first 16n block of diff_dst (block size: 2c16n16c) from SLM
        READ_DST();
        // Compute 32o32i with reduction on first block of 16n
        COMPUTE(0);
#if IC_BLK_UNROLL == 8
        COMPUTE(1);
#endif

        if (k_blk < max_k_blocks - 1) {
#if IC_BLK_UNROLL == 4
            if (sg_id < 8)
#endif
                WRITE_SRC();
#if OC_BLK_UNROLL == 4
            if (sg_id < 8)
#endif
                WRITE_DST();
        }

        src_loc_read += src_slm_offset;
        diff_dst_loc_read += dst_slm_offset;

        // Read second 16n block of diff_dst (block size: 2c16n16c) from SLM
        READ_DST();
        // Reduce on the same block(32o32i) with reduction on second block of 16n
        COMPUTE(0);
#if IC_BLK_UNROLL == 8
        COMPUTE(1);
#endif

        barrier(CLK_LOCAL_MEM_FENCE);
    }

#if K_BLOCK_TAIL != 0
    if (group_k_block / K_BLOCKS == K_WORKGROUPS - 1) {
        WRITE_SECOND_SLM_BUFFER();
        barrier(CLK_LOCAL_MEM_FENCE);
        __attribute__((opencl_unroll_hint(1))) // attr:no-format
        for (int k_blk = 0; k_blk < K_BLOCK_TAIL; ++k_blk) {

            src_loc_read = &src_slm[(k_blk_iter % 2) * (SRC_SLM_SIZE / 2)];
            diff_dst_loc_read
                    = &diff_dst_slm[(k_blk_iter % 2) * (DST_SLM_SIZE / 2)];

            ow++;
            iw += SW;
            if (ow == ow_end + 1) {
                ow = max((PW - kw * (1 + DW) + SW - 1) / SW, 0);
                oh++;
                iw = ow * SW - PW + kw * (1 + DW);
                ih += SH;
            }
            if (oh == oh_end + 1) {
                oh = max((PH - kh * (1 + DH) + SH - 1) / SH, 0);
                od++;
                ih = oh * SH - PH + kh * (1 + DH);
                id += SD;
            }
            if (od == od_end + 1) {
                od = max((PD - kd * (1 + DD) + SD - 1) / SD, 0);
                id = od * SD - PD + kd * (1 + DD);
                n_block++;
            }

            k_blk_iter++;

            READ_DST();
            COMPUTE(0);
#if IC_BLK_UNROLL == 8
            COMPUTE(1);
#endif

            if (k_blk < K_BLOCK_TAIL - 1) {
#if IC_BLK_UNROLL == 4
                if (sg_id < 8)
#endif
                    WRITE_SRC();
#if OC_BLK_UNROLL == 4
                if (sg_id < 8)
#endif
                    WRITE_DST();
            }

            src_loc_read += src_slm_offset;
            diff_dst_loc_read += dst_slm_offset;

            READ_DST();
            COMPUTE(0);
#if IC_BLK_UNROLL == 8
            COMPUTE(1);
#endif

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
#endif

    volatile __global atomic_float *diff_wei_write;

#define WRITE_WEI(i_o, i_i) \
    do { \
        diff_wei_write = (volatile __global atomic_float *)&diff_wei[WHT_OFF( \
                0, i_o * 8, i_i * IC_BLOCK + sg_loc_id, 0, 0, 0)]; \
        atomic_add_global(&diff_wei_write[0], ACC[i_o][2 * i_i].s0); \
        atomic_add_global(&diff_wei_write[16], ACC[i_o][2 * i_i].s1); \
        atomic_add_global(&diff_wei_write[32], ACC[i_o][2 * i_i].s2); \
        atomic_add_global(&diff_wei_write[48], ACC[i_o][2 * i_i].s3); \
        atomic_add_global(&diff_wei_write[64], ACC[i_o][2 * i_i].s4); \
        atomic_add_global(&diff_wei_write[80], ACC[i_o][2 * i_i].s5); \
        atomic_add_global(&diff_wei_write[96], ACC[i_o][2 * i_i].s6); \
        atomic_add_global(&diff_wei_write[112], ACC[i_o][2 * i_i].s7); \
        diff_wei_write += SUB_GROUP_SIZE; \
        atomic_add_global(&diff_wei_write[0], ACC[i_o][2 * i_i + 1].s0); \
        atomic_add_global(&diff_wei_write[16], ACC[i_o][2 * i_i + 1].s1); \
        atomic_add_global(&diff_wei_write[32], ACC[i_o][2 * i_i + 1].s2); \
        atomic_add_global(&diff_wei_write[48], ACC[i_o][2 * i_i + 1].s3); \
        atomic_add_global(&diff_wei_write[64], ACC[i_o][2 * i_i + 1].s4); \
        atomic_add_global(&diff_wei_write[80], ACC[i_o][2 * i_i + 1].s5); \
        atomic_add_global(&diff_wei_write[96], ACC[i_o][2 * i_i + 1].s6); \
        atomic_add_global(&diff_wei_write[112], ACC[i_o][2 * i_i + 1].s7); \
    } while (0)

    WRITE_WEI(0, 0);
    WRITE_WEI(1, 0);
#if OC_BLK_UNROLL == 8
    WRITE_WEI(2, 0);
    WRITE_WEI(3, 0);
#endif

#if IC_BLK_UNROLL == 8
    WRITE_WEI(0, 1);
    WRITE_WEI(1, 1);
#if OC_BLK_UNROLL == 8
    WRITE_WEI(2, 1);
    WRITE_WEI(3, 1);
#endif
#endif

#if WITH_BIAS
#define COMPUTE_BIAS() \
    do { \
        dst_off = n * DST_MB_STRIDE + od * DST_D_STRIDE + oh * DST_H_STRIDE \
                + ow * DST_W_STRIDE; \
        Dt[0] = __builtin_IB_simd_block_read_16_global_h( \
                (__global ushort *)&diff_dst[dst_off]); \
        Dt[1] = __builtin_IB_simd_block_read_16_global_h( \
                (__global ushort *)&diff_dst[dst_off + USHORT_PER_READ]); \
        BIAS_ACC[0] += (CONVERT_TO_F32(Dt[0].s0) + CONVERT_TO_F32(Dt[1].s0) \
                + CONVERT_TO_F32(Dt[0].s2) + CONVERT_TO_F32(Dt[1].s2) \
                + CONVERT_TO_F32(Dt[0].s4) + CONVERT_TO_F32(Dt[1].s4) \
                + CONVERT_TO_F32(Dt[0].s6) + CONVERT_TO_F32(Dt[1].s6) \
                + CONVERT_TO_F32(Dt[0].s8) + CONVERT_TO_F32(Dt[1].s8) \
                + CONVERT_TO_F32(Dt[0].sa) + CONVERT_TO_F32(Dt[1].sa) \
                + CONVERT_TO_F32(Dt[0].sc) + CONVERT_TO_F32(Dt[1].sc) \
                + CONVERT_TO_F32(Dt[0].se) + CONVERT_TO_F32(Dt[1].se)); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s0) \
                + CONVERT_TO_F32(Dt[1].odd.s0); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s1) \
                + CONVERT_TO_F32(Dt[1].odd.s1); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s2) \
                + CONVERT_TO_F32(Dt[1].odd.s2); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s3) \
                + CONVERT_TO_F32(Dt[1].odd.s3); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s4) \
                + CONVERT_TO_F32(Dt[1].odd.s4); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s5) \
                + CONVERT_TO_F32(Dt[1].odd.s5); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s6) \
                + CONVERT_TO_F32(Dt[1].odd.s6); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s7) \
                + CONVERT_TO_F32(Dt[1].odd.s7); \
    } while (0)

    // handle padded region for bias computation
    // first thread in spatial gws dimension, handles the left padding
    if (compute_bias && gid[2] % K_WORKGROUPS == 0) {
        for (int n = 0; n < MB / (MB_BLOCK * MB_BLK_UNROLL); ++n) {
            for (od = 0; od < od_start; ++od) {
                for (oh = 0; oh < OH; ++oh) {
                    for (ow = 0; ow < OW; ++ow) {
                        COMPUTE_BIAS();
                    }
                }
            }
        }
        for (int n = 0; n < MB / (MB_BLOCK * MB_BLK_UNROLL); ++n) {
            for (od = od_start; od < OD; ++od) {
                for (oh = 0; oh < oh_start; ++oh) {
                    for (ow = 0; ow < OW; ++ow) {
                        COMPUTE_BIAS();
                    }
                }
            }
        }
        for (int n = 0; n < MB / (MB_BLOCK * MB_BLK_UNROLL); ++n) {
            for (od = od_start; od < OD; ++od) {
                for (oh = oh_start; oh < OH; ++oh) {
                    for (ow = 0; ow < ow_start; ++ow) {
                        COMPUTE_BIAS();
                    }
                }
            }
        }
    }

    // last thread handles the right padding
    if (compute_bias && gid[2] % K_WORKGROUPS == K_WORKGROUPS - 1) {
        for (int n = 0; n < MB / (MB_BLOCK * MB_BLK_UNROLL); ++n) {
            for (od = od_start; od < OD; ++od) {
                for (oh = oh_end + 1; oh < OH; ++oh) {
                    for (ow = ow_start; ow < OW; ++ow) {
                        COMPUTE_BIAS();
                    }
                }
            }
        }
        for (int n = 0; n < MB / (MB_BLOCK * MB_BLK_UNROLL); ++n) {
            for (od = od_end + 1; od < OD; ++od) {
                for (oh = oh_start; oh < oh_end + 1; ++oh) {
                    for (ow = ow_start; ow < OW; ++ow) {
                        COMPUTE_BIAS();
                    }
                }
            }
        }
        for (int n = 0; n < MB / (MB_BLOCK * MB_BLK_UNROLL); ++n) {
            for (od = od_start; od < od_end + 1; ++od) {
                for (oh = oh_start; oh < oh_end + 1; ++oh) {
                    for (ow = ow_end + 1; ow < OW; ++ow) {
                        COMPUTE_BIAS();
                    }
                }
            }
        }
    }
    if (compute_bias) {
        atomic_add_global(&dbias[sg_loc_id], BIAS_ACC.s0);
        atomic_add_global(&dbias[sg_loc_id + SUB_GROUP_SIZE], BIAS_ACC.s1);
    }
#endif
}

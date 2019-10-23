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

#define POST_OP_DATA_T float

#include "ocl/ocl_math_utils.h"
#include "ocl/ocl_post_ops.h"
#include "ocl/ocl_types.h"

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

#define BLOCK_READ_SRC_FROM_SLM(data, idx) \
    data = READ_LOCAL_8((__local uint *)&src_slm[idx]);

#define BLOCK_READ_WHT(data, idx) \
    data = intel_sub_group_block_read_us8((__global ushort *)&wei[idx]);

#define BLOCK_READ_WHT_FROM_SLM(data, idx) \
    data = READ_LOCAL_US_8((__local ushort *)&wei_slm[idx]);

#define INPUT_PIXEL_WIDTH_OFFSET (IC_BLOCK * MB_BLOCK)
#define INPUT_PIXEL_HEIGHT_OFFSET (INPUT_PIXEL_WIDTH_OFFSET * IW)
#define INPUT_CHANNEL_BLOCK_OFFSET (INPUT_PIXEL_HEIGHT_OFFSET * IH) // For NChw

#define OUTPUT_PIXEL_WIDTH_OFFSET (IC_BLOCK * MB_BLOCK)
#define OUTPUT_PIXEL_HEIGHT_OFFSET (OUTPUT_PIXEL_WIDTH_OFFSET * OW)
#define OUTPUT_CHANNEL_BLOCK_OFFSET \
    (OUTPUT_PIXEL_HEIGHT_OFFSET * OH) // For NChw

// Weights offsets
#define KERNEL_BLOCK_OFFSET (8 * 8)
#define NEXT_KERNEL_OFFSET (KERNEL_BLOCK_OFFSET * IC_NCHUNK * 2)

#define OC_BLOCK_NUMBER (2)
#define WEI_IC_NCHUNK (IC_NCHUNK * 2)
#define WEI_IC_BLOCK (8)
#define WEI_IC_BLOCK_NUMBER (2)
#define OC_PER_WI (4)

#if DST_DT_F16
#define TO_DST(_x) as_ushort8(convert_half8(_x))
#else
#if DST_DT_F32
#define TO_DST(_x) as_uint8(_x)
#else
#define TO_DST(_x) convert_f32_to_bf16_vec8(_x)
#endif
#endif

#if DST_DT_F32
#define DST_BLOCK_WRITE_8 intel_sub_group_block_write8
#define DST_BLOCK_READ_8 intel_sub_group_block_read8
#define DST_BLOCK_TYPE uint
#else
#define DST_BLOCK_WRITE_8 intel_sub_group_block_write_us8
#define DST_BLOCK_READ_8 intel_sub_group_block_read_us8
#define DST_BLOCK_TYPE ushort
#endif

#if BIA_DT_F32
#define CONVERT_BIAS(_bias) (_bias)
#else
#define CONVERT_BIAS(_bias) CONVERT_FLOAT2_T(_bias)
#endif

inline float8 POST_OPS_PASS(float8 val, float2 bias, float sum_scale,
        float alpha, float beta, __global DST_DATA_T *dst) {
#if WITH_BIAS
    val[0] += bias[0];
    val[2] += bias[0];
    val[4] += bias[0];
    val[6] += bias[0];
    val[1] += bias[1];
    val[3] += bias[1];
    val[5] += bias[1];
    val[7] += bias[1];
#endif // WITH_BIAS

#if WITH_ELTWISE == 1
    val[0] = fwd_eltwise(val[0], alpha, beta);
    val[1] = fwd_eltwise(val[1], alpha, beta);
    val[2] = fwd_eltwise(val[2], alpha, beta);
    val[3] = fwd_eltwise(val[3], alpha, beta);
    val[4] = fwd_eltwise(val[4], alpha, beta);
    val[5] = fwd_eltwise(val[5], alpha, beta);
    val[6] = fwd_eltwise(val[6], alpha, beta);
    val[7] = fwd_eltwise(val[7], alpha, beta);
#endif // WITH_ELTWISE

#if DT_F16
#define SUM_CONVERT(_sum) CONVERT_FLOAT8_T(as_half8(_sum))
#else DT_BF16
#if DST_DT_F32
#define SUM_CONVERT(_sum) as_float8(_sum)
#else
#define SUM_CONVERT(_sum) CONVERT_FLOAT8_T(_sum)
#endif
#endif

#if WITH_SUM
    val += SUM_CONVERT(DST_BLOCK_READ_8((__global ushort *)&dst[0]))
            * sum_scale;
#endif // WITH_SUM

#if WITH_POST_SUM_ELTWISE
    val[0] = fwd_eltwise(val[0], alpha, beta);
    val[1] = fwd_eltwise(val[1], alpha, beta);
    val[2] = fwd_eltwise(val[2], alpha, beta);
    val[3] = fwd_eltwise(val[3], alpha, beta);
    val[4] = fwd_eltwise(val[4], alpha, beta);
    val[5] = fwd_eltwise(val[5], alpha, beta);
    val[6] = fwd_eltwise(val[6], alpha, beta);
    val[7] = fwd_eltwise(val[7], alpha, beta);
#endif // WITH_SUM_ELTWISE

    return val;
}

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
gen12hp_1x1_conv_fwd_kernel_x16(const __global DATA_T *src,
        const __global DATA_T *wei, const __global BIA_DATA_T *bias,
        __global DST_DATA_T *dst, float eltwise_alpha, float eltwise_beta,
        float sum_scale) {

    // Groups:
    const uint oc_group_id = get_global_id(0) / 8;
    const uint mb_group_id = get_group_id(2);
    const int sp_local_id = get_local_id(1);
    const uint slm_oc_group_id = get_local_id(0) / 8;

    // Spatial
    const uint sp = get_global_id(1);
    const uint ow = sp % OW_PADDED;
    const uint oh = sp / OW_PADDED;
    const uint iw = ow * SW;
    const uint ih = oh * SH;

    // Source (At ic = 0)
    src += mb_group_id * INPUT_CHANNEL_BLOCK_OFFSET * IC_NCHUNK; // MB off
    src += ih * INPUT_PIXEL_HEIGHT_OFFSET; // height offset
    src += iw * INPUT_PIXEL_WIDTH_OFFSET; // width offset

    // Destination
    dst += mb_group_id * OUTPUT_CHANNEL_BLOCK_OFFSET * OC_NCHUNK; // MB off
    dst += OUTPUT_CHANNEL_BLOCK_OFFSET * oc_group_id
            * OC_BLOCK_NUMBER; //OC offset
    dst += oh * OUTPUT_PIXEL_HEIGHT_OFFSET; // height offset
    dst += ow * OUTPUT_PIXEL_WIDTH_OFFSET; // width offset

    // Weights
    wei += oc_group_id * KERNEL_BLOCK_OFFSET * WEI_IC_NCHUNK * OC_PER_WI;

#ifdef XF16_SRC_SLM
    __local ushort src_slm[IC_BLOCK * MB_BLOCK * LWS_1];
    __local ushort *slm_src_ptr;
    __global DATA_T *glob_src_ptr;
#endif

#ifdef XF16_WEI_SLM
    __local ushort wei_slm[IC_BLOCK * OC_BLOCK * (LWS_0 / SUB_GROUP_SIZE)];
    __local ushort *slm_wei_ptr;
    __global DATA_T *glob_wei_ptr;
#endif

#define SLM_SRC_GR_OFFSET (sp_local_id * IC_BLOCK * MB_BLOCK)
#define SLM_WEI_GR_OFFSET (slm_oc_group_id * IC_BLOCK * OC_BLOCK)

    // Output accumulators:
    // C00 -> 8 MB(0-7), 1 OC(sg_lid)
    // C01 -> 8 MB(8-15), 1 OC(sg_lid)
    // C02 -> 8 MB(16-23), 1 OC(sg_lid)
    // C03 -> 8 MB(24-31), 1 OC(sg_lid)
    float8 C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    // C10 -> 8 MB(0-7), 1 OC(sg_lid + sg_size)
    // C11 -> 8 MB(8-15), 1 OC(sg_lid + sg_size)
    // C12 -> 8 MB(16-23), 1 OC(sg_lid + sg_size)
    // C13 -> 8 MB(24-31), 1 OC(sg_lid + sg_size)
    float8 C10 = 0, C11 = 0, C12 = 0, C13 = 0;
    // C20 -> 8 MB(0-7), 1 OC(sg_lid + 2 * sg_size)
    // C21 -> 8 MB(8-15), 1 OC(sg_lid + 2 *sg_size)
    // C22 -> 8 MB(16-23), 1 OC(sg_lid + 2 *sg_size)
    // C23 -> 8 MB(24-31), 1 OC(sg_lid + 2 *sg_size)
    float8 C20 = 0, C21 = 0, C22 = 0, C23 = 0;
    // C30 -> 8 MB(0-7), 1 OC(sg_lid + 3 * sg_size)
    // C31 -> 8 MB(8-15), 1 OC(sg_lid + 3 *sg_size)
    // C32 -> 8 MB(16-23), 1 OC(sg_lid + 3 *sg_size)
    // C33 -> 8 MB(24-31), 1 OC(sg_lid + 3 *sg_size)
    float8 C30 = 0, C31 = 0, C32 = 0, C33 = 0;

    // Source:
    // S0 -> 8 MB(0-7), 16 IC
    // S1 -> 8 MB(8-15), 16 IC
    // S2 -> 8 MB(16-23), 16 IC
    // S3 -> 8 MB(24-31), 16 IC
    uint8 S0, S1, S2, S3;

    // Weights:
    // W00 -> 8 IC(0-7), 1 OC(sg_lid)
    // W01 -> 8 IC(8-15), 1 OC(sg_lid)
    ushort8 W00, W01;
    // W10 -> 8 IC(0-7), 1 OC(sg_lid + sg_size)
    // W11 -> 8 IC(8-15), 1 OC(sg_lid + sg_size)
    ushort8 W10, W11;
    // W10 -> 8 IC(0-7), 1 OC(sg_lid + 2 * sg_size)
    // W11 -> 8 IC(8-15), 1 OC(sg_lid + 2 * sg_size)
    ushort8 W20, W21;
    // W10 -> 8 IC(0-7), 1 OC(sg_lid + 3 * sg_size)
    // W11 -> 8 IC(8-15), 1 OC(sg_lid + 3 * sg_size)
    ushort8 W30, W31;

    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (uint ic_block_id = 0; ic_block_id < IC_NCHUNK; ++ic_block_id) {

#if XF16_SRC_SLM || XF16_WEI_SLM
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if XF16_SRC_SLM
        if (ow < OW) {
            slm_src_ptr = SLM_SRC_GR_OFFSET + src_slm
                    + (slm_oc_group_id * SUB_GROUP_SIZE * IC_BLOCK);
            glob_src_ptr = src + slm_oc_group_id * (SUB_GROUP_SIZE * IC_BLOCK);
            WRITE_LOCAL_8((__local uint *)&slm_src_ptr[0],
                    intel_sub_group_block_read8(
                            (__global uint *)&glob_src_ptr[0]));
        }
#endif

#if XF16_WEI_SLM
        slm_wei_ptr = SLM_WEI_GR_OFFSET + wei_slm
                + (sp_local_id * 8 * SUB_GROUP_SIZE);
        glob_wei_ptr = wei + ((sp_local_id % 2) * KERNEL_BLOCK_OFFSET)
                + ((sp_local_id / 2) * NEXT_KERNEL_OFFSET);
        WRITE_LOCAL_US_8((__local ushort *)&slm_wei_ptr[0],
                intel_sub_group_block_read_us8(
                        (__global ushort *)&glob_wei_ptr[0]));
#endif

#if XF16_SRC_SLM || XF16_WEI_SLM
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        if (ow < OW) {
#if XF16_SRC_SLM
            BLOCK_READ_SRC_FROM_SLM(S0, SLM_SRC_GR_OFFSET);
            BLOCK_READ_SRC_FROM_SLM(S1, SLM_SRC_GR_OFFSET + 8 * IC_BLOCK);
            BLOCK_READ_SRC_FROM_SLM(S2, SLM_SRC_GR_OFFSET + 16 * IC_BLOCK);
            BLOCK_READ_SRC_FROM_SLM(S3, SLM_SRC_GR_OFFSET + 24 * IC_BLOCK);
#else
            BLOCK_READ_SRC(S0, 0);
            BLOCK_READ_SRC(S1, 8 * IC_BLOCK);
            BLOCK_READ_SRC(S2, 16 * IC_BLOCK);
            BLOCK_READ_SRC(S3, 24 * IC_BLOCK);
#endif

#if XF16_WEI_SLM
            BLOCK_READ_WHT_FROM_SLM(W00, SLM_WEI_GR_OFFSET);
            BLOCK_READ_WHT_FROM_SLM(W01, SLM_WEI_GR_OFFSET + 8 * WEI_IC_BLOCK);
            BLOCK_READ_WHT_FROM_SLM(W10, SLM_WEI_GR_OFFSET + 16 * WEI_IC_BLOCK);
            BLOCK_READ_WHT_FROM_SLM(W11, SLM_WEI_GR_OFFSET + 24 * WEI_IC_BLOCK);
            BLOCK_READ_WHT_FROM_SLM(W20, SLM_WEI_GR_OFFSET + 32 * WEI_IC_BLOCK);
            BLOCK_READ_WHT_FROM_SLM(W21, SLM_WEI_GR_OFFSET + 40 * WEI_IC_BLOCK);
            BLOCK_READ_WHT_FROM_SLM(W30, SLM_WEI_GR_OFFSET + 48 * WEI_IC_BLOCK);
            BLOCK_READ_WHT_FROM_SLM(W31, SLM_WEI_GR_OFFSET + 56 * WEI_IC_BLOCK);
#else
            BLOCK_READ_WHT(W00, 0);
            BLOCK_READ_WHT(W01, KERNEL_BLOCK_OFFSET);
            BLOCK_READ_WHT(W10, NEXT_KERNEL_OFFSET);
            BLOCK_READ_WHT(W11, NEXT_KERNEL_OFFSET + KERNEL_BLOCK_OFFSET);
            BLOCK_READ_WHT(W20, NEXT_KERNEL_OFFSET * 2);
            BLOCK_READ_WHT(W21, NEXT_KERNEL_OFFSET * 2 + KERNEL_BLOCK_OFFSET);
            BLOCK_READ_WHT(W30, NEXT_KERNEL_OFFSET * 3);
            BLOCK_READ_WHT(W31, NEXT_KERNEL_OFFSET * 3 + KERNEL_BLOCK_OFFSET);
#endif

            // MB 0-7, OC sg_lid
            C00 = MMAD8X8(S0, as_int8((ushort16)(W00, W01)), C00);
            C10 = MMAD8X8(S0, as_int8((ushort16)(W10, W11)), C10);
            C20 = MMAD8X8(S0, as_int8((ushort16)(W20, W21)), C20);
            C30 = MMAD8X8(S0, as_int8((ushort16)(W30, W31)), C30);
            // MB 8-15, OC sg_lid
            C01 = MMAD8X8(S1, as_int8((ushort16)(W00, W01)), C01);
            C11 = MMAD8X8(S1, as_int8((ushort16)(W10, W11)), C11);
            C21 = MMAD8X8(S1, as_int8((ushort16)(W20, W21)), C21);
            C31 = MMAD8X8(S1, as_int8((ushort16)(W30, W31)), C31);
            // MB 16-23, OC sg_lid
            C02 = MMAD8X8(S2, as_int8((ushort16)(W00, W01)), C02);
            C12 = MMAD8X8(S2, as_int8((ushort16)(W10, W11)), C12);
            C22 = MMAD8X8(S2, as_int8((ushort16)(W20, W21)), C22);
            C32 = MMAD8X8(S2, as_int8((ushort16)(W30, W31)), C32);
            // MB 24-31, OC sg_lid
            C03 = MMAD8X8(S3, as_int8((ushort16)(W00, W01)), C03);
            C13 = MMAD8X8(S3, as_int8((ushort16)(W10, W11)), C13);
            C23 = MMAD8X8(S3, as_int8((ushort16)(W20, W21)), C23);
            C33 = MMAD8X8(S3, as_int8((ushort16)(W30, W31)), C33);
        }

        src += INPUT_CHANNEL_BLOCK_OFFSET;
        wei += KERNEL_BLOCK_OFFSET * WEI_IC_BLOCK_NUMBER;
    }

    float8 dst_val[2];
    BIA_DATA2_T bias_val_1 = 0; // For 2 OC
    BIA_DATA2_T bias_val_2 = 0; // For next 2 OC

#if WITH_BIAS
    bias += oc_group_id * OC_BLOCK;
#if BIA_DT_F32
    bias_val_1
            = as_float2(intel_sub_group_block_read2((__global uint *)&bias[0]));
    bias_val_2 = as_float2(
            intel_sub_group_block_read2((__global uint *)&bias[16]));
#else
    bias_val_1 = AS_DATA2_T(
            intel_sub_group_block_read_us2((__global ushort *)&bias[0]));
    bias_val_2 = AS_DATA2_T(
            intel_sub_group_block_read_us2((__global ushort *)&bias[16]));
#endif
#endif

#define PACK_AND_WRITE(_acc0, _acc1, _bias_val, _dst_offset) \
    dst_val[0] = (float8)(_acc0[0], _acc1[0], _acc0[1], _acc1[1], _acc0[2], \
            _acc1[2], _acc0[3], _acc1[3]); \
    dst_val[1] = (float8)(_acc0[4], _acc1[4], _acc0[5], _acc1[5], _acc0[6], \
            _acc1[6], _acc0[7], _acc1[7]); \
\
    dst_val[0] = POST_OPS_PASS(dst_val[0], CONVERT_BIAS(_bias_val), sum_scale, \
            eltwise_alpha, eltwise_beta, dst + _dst_offset); \
    dst_val[1] = POST_OPS_PASS(dst_val[1], CONVERT_BIAS(_bias_val), sum_scale, \
            eltwise_alpha, eltwise_beta, dst + _dst_offset + 64); \
\
    DST_BLOCK_WRITE_8( \
            (__global DST_BLOCK_TYPE *)&dst[_dst_offset], TO_DST(dst_val[0])); \
    DST_BLOCK_WRITE_8((__global DST_BLOCK_TYPE *)&dst[_dst_offset + 64], \
            TO_DST(dst_val[1]));

// 2x intel_sub_group_block_write_us8 per pack
#define WRITE_OFFSET (2 * 8 * SUB_GROUP_SIZE)

    if (ow < OW) {
        // Write results from MB(0-7) of 2 Output Channels
        PACK_AND_WRITE(C00, C10, bias_val_1, 0)

        // Write results from MB(8-15) of 2 Output Channels
        PACK_AND_WRITE(C01, C11, bias_val_1, WRITE_OFFSET)

        // Write results from MB(16-23) of 2 Output Channels
        PACK_AND_WRITE(C02, C12, bias_val_1, 2 * WRITE_OFFSET)

        // Write results from MB(24-31) of 2 Output Channels
        PACK_AND_WRITE(C03, C13, bias_val_1, 3 * WRITE_OFFSET)

        dst += OUTPUT_CHANNEL_BLOCK_OFFSET;

        // Write results from MB(0-7) of 2 Output Channels
        PACK_AND_WRITE(C20, C30, bias_val_2, 0)

        // Write results from MB(8-15) of 2 Output Channels
        PACK_AND_WRITE(C21, C31, bias_val_2, WRITE_OFFSET)

        // Write results from MB(16-23) of 2 Output Channels
        PACK_AND_WRITE(C22, C32, bias_val_2, 2 * WRITE_OFFSET)

        // Write results from MB(24-31) of 2 Output Channels
        PACK_AND_WRITE(C23, C33, bias_val_2, 3 * WRITE_OFFSET)
    }
}

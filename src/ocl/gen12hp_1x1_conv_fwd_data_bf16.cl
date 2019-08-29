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

#include "ocl/ocl_math_utils.h"
#include "ocl/ocl_post_ops.h"
#include "ocl/ocl_types.h"

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

#define BLOCK_READ_WHT(data, idx) \
    data = intel_sub_group_block_read_us8((__global ushort *)&wei[idx]);

#define CHANNEL_OFFSET 1
#define MB_OFFSET IC_BLOCK

#define INPUT_PIXEL_WIDTH_OFFSET (MB_OFFSET * MB_BLOCK)
#define INPUT_PIXEL_HEIGHT_OFFSET (INPUT_PIXEL_WIDTH_OFFSET * IW)
#define INPUT_CHANNEL_BLOCK_OFFSET (INPUT_PIXEL_HEIGHT_OFFSET * IH) // For NChw

#define OUTPUT_PIXEL_WIDTH_OFFSET (MB_OFFSET * MB_BLOCK)
#define OUTPUT_PIXEL_HEIGHT_OFFSET (OUTPUT_PIXEL_WIDTH_OFFSET * OW)
#define OUTPUT_CHANNEL_BLOCK_OFFSET \
    (OUTPUT_PIXEL_HEIGHT_OFFSET * OH) // For NChw

// Weights offsets
#define WEIGHTS_WIDTH_OFFSET (8 * 8)
#define WEIGHTS_HEIGHT_OFFSET (WEIGHTS_WIDTH_OFFSET * 1)
#define KERNEL_BLOCK_OFFSET (WEIGHTS_HEIGHT_OFFSET * 1)
#define NEXT_KERNEL_OFFSET (KERNEL_BLOCK_OFFSET * IC_NCHUNK * 2)

inline float8 POST_OPS_PASS(float8 val, float2 bias, float scales, float alpha,
        float beta, float sum_scale, __global ushort *dst) {
#if WITH_BIAS
    val[0] += bias[0] * scales;
    val[2] += bias[0] * scales;
    val[4] += bias[0] * scales;
    val[6] += bias[0] * scales;
    val[1] += bias[1] * scales;
    val[3] += bias[1] * scales;
    val[5] += bias[1] * scales;
    val[7] += bias[1] * scales;
#else
    val *= scales;
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

#if WITH_SUM
    val += convert_bf16_to_f32_vec8(
                   intel_sub_group_block_read_us8((__global ushort *)&dst[0]))
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
gen12hp_1x1_conv_fwd_kernel_bf16(const __global ushort *src,
        const __global ushort *wei, const __global ushort *bias,
        __global ushort *dst, float eltwise_alpha, float eltwise_beta,
        float sum_scale, float scales) {

    // Groups:
    const uint oc_group_id = get_group_id(0);
    const uint sp_group_id = get_group_id(1);
    const uint mb_group_id = get_group_id(2);

    // SIMD
    const uint sg_local_id = get_sub_group_local_id();
    const uint sg_id = get_sub_group_id();

    // Spatial
    const uint sp = get_global_id(1);
    const uint ow = sp % OW;
    const uint oh = sp / OW;
    const uint iw = ow * SW;
    const uint ih = oh * SH;

    // Source (At ic = 0)
    src += mb_group_id * INPUT_CHANNEL_BLOCK_OFFSET * IC_NCHUNK; // MB off
    src += ih * INPUT_PIXEL_HEIGHT_OFFSET; // height offset
    src += iw * INPUT_PIXEL_WIDTH_OFFSET; // width offset

    // Destination
    dst += mb_group_id * OUTPUT_CHANNEL_BLOCK_OFFSET * OC_NCHUNK; // MB off
    dst += OUTPUT_CHANNEL_BLOCK_OFFSET * oc_group_id; //OC offset
    dst += oh * OUTPUT_PIXEL_HEIGHT_OFFSET; // height offset
    dst += ow * OUTPUT_PIXEL_WIDTH_OFFSET; // width offset

    // Weights
    wei += oc_group_id * KERNEL_BLOCK_OFFSET * IC_NCHUNK * 4;

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

    // Source:
    // S0 -> 8 MB(0-7), 2 IC(sg_lid, sg_lid+1)
    // S1 -> 8 MB(8-15), 2 IC(sg_lid, sg_lid+1)
    // S2 -> 8 MB(16-23), 2 IC(sg_lid, sg_lid+1)
    // S3 -> 8 MB(24-31), 2 IC(sg_lid, sg_lid+1)
    uint8 S0, S1, S2, S3;

    // Weights:
    // W00 -> 8 IC(0-7), 1 OC(sg_lid)
    // W01 -> 8 IC(8-15), 1 OC(sg_lid)
    ushort8 W00, W01;
    // W10 -> 8 IC(0-7), 1 OC(sg_lid + sg_size)
    // W11 -> 8 IC(8-15), 1 OC(sg_lid + sg_size)
    ushort8 W10, W11;

    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (uint ic_block_id = 0; ic_block_id < IC_NCHUNK; ++ic_block_id) {
        BLOCK_READ_SRC(S0, 0);
        BLOCK_READ_SRC(S1, 8 * IC_BLOCK);
        BLOCK_READ_SRC(S2, 16 * IC_BLOCK);
        BLOCK_READ_SRC(S3, 24 * IC_BLOCK);

        BLOCK_READ_WHT(W00, 0);
        BLOCK_READ_WHT(W01, KERNEL_BLOCK_OFFSET);

        BLOCK_READ_WHT(W10, NEXT_KERNEL_OFFSET);
        BLOCK_READ_WHT(W11, NEXT_KERNEL_OFFSET + KERNEL_BLOCK_OFFSET);

        // MB 0-7, OC sg_lid
        C00 = mmad8x8(S0, as_int8((ushort16)(W00, W01)), C00);
        // MB 0-7, OC sg_lid + sg_size
        C10 = mmad8x8(S0, as_int8((ushort16)(W10, W11)), C10);
        // MB 8-15, OC sg_lid
        C01 = mmad8x8(S1, as_int8((ushort16)(W00, W01)), C01);
        // MB 8-15, OC sg_lid + sg_size
        C11 = mmad8x8(S1, as_int8((ushort16)(W10, W11)), C11);
        // MB 16-23, OC sg_lid
        C02 = mmad8x8(S2, as_int8((ushort16)(W00, W01)), C02);
        // MB 16-23, OC sg_lid + sg_size
        C12 = mmad8x8(S2, as_int8((ushort16)(W10, W11)), C12);
        // MB 24-31, OC sg_lid
        C03 = mmad8x8(S3, as_int8((ushort16)(W00, W01)), C03);
        // MB 24-31, OC sg_lid + sg_size
        C13 = mmad8x8(S3, as_int8((ushort16)(W10, W11)), C13);

        src += INPUT_CHANNEL_BLOCK_OFFSET;
        wei += KERNEL_BLOCK_OFFSET * 2;
    }

    float8 dst_val[2];
    ushort2 bias_val = 0;

#if WITH_BIAS
    bias += oc_group_id * OC_BLOCK;
    bias_val = intel_sub_group_block_read_us2((__global ushort *)&bias[0]);
#endif

#define PACK_AND_WRITE(_acc0, _acc1) \
    dst_val[0] = (float8)(_acc0[0], _acc1[0], _acc0[1], _acc1[1], _acc0[2], \
            _acc1[2], _acc0[3], _acc1[3]); \
    dst_val[1] = (float8)(_acc0[4], _acc1[4], _acc0[5], _acc1[5], _acc0[6], \
            _acc1[6], _acc0[7], _acc1[7]); \
\
    dst_val[0] = POST_OPS_PASS(dst_val[0], convert_bf16_to_f32_vec2(bias_val), \
            scales, eltwise_alpha, eltwise_beta, sum_scale, dst); \
    dst_val[1] = POST_OPS_PASS(dst_val[1], convert_bf16_to_f32_vec2(bias_val), \
            scales, eltwise_alpha, eltwise_beta, sum_scale, dst + 64); \
\
    intel_sub_group_block_write_us8( \
            (__global ushort *)&dst[0], convert_f32_to_bf16_vec8(dst_val[0])); \
    dst += 64; \
    intel_sub_group_block_write_us8( \
            (__global ushort *)&dst[0], convert_f32_to_bf16_vec8(dst_val[1])); \
    dst += 64;

    // Write results from MB(0-7) of 2 Output Channels
    PACK_AND_WRITE(C00, C10)

    // Write results from MB(8-15) of 2 Output Channels
    PACK_AND_WRITE(C01, C11)

    // Write results from MB(16-23) of 2 Output Channels
    PACK_AND_WRITE(C02, C12)

    // Write results from MB(24-31) of 2 Output Channels
    PACK_AND_WRITE(C03, C13)
}

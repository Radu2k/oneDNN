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
#include "ocl/ocl_types.h"
#if WITH_ELTWISE == 1 || WITH_POST_SUM_ELTWISE == 1
#include "ocl/ocl_post_ops.h"
#endif

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));

#define BLOCK_READ_BIA(data, idx) \
    data = as_float4(intel_sub_group_block_read4((__global uint *)&bias[idx]));

#define CHANNEL_OFFSET 1
#define MB_OFFSET IC_BLOCK
#define PIXEL_WIDTH_OFFSET (MB_OFFSET * MB_BLOCK)
#define PIXEL_HEIGHT_OFFSET (PIXEL_WIDTH_OFFSET * IW)
#define CHANNEL_BLOCK_OFFSET (PIXEL_HEIGHT_OFFSET * IH) // For NChw

// Weights offsets
#define WEIGHTS_WIDTH_OFFSET (4 * 8 * 8 * 4)
#define WEIGHTS_HEIGHT_OFFSET (WEIGHTS_WIDTH_OFFSET * 1)
#define KERNEL_BLOCK_OFFSET (WEIGHTS_HEIGHT_OFFSET * 1)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
gen12lp_1x1_conv_fwd_u8s8s32u8_kernel(const __global uchar *src,
        const __global char *wei, const __global float *bias,
        __global uchar *dst, float alpha, float beta, float sum_scale,
        float scales) {

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
    const uint iw = ow;
    const uint ih = oh;

    // Source (At ic = 0)
    src += (mb_group_id % 2) * MB_BLOCK / 2 * MB_OFFSET; // MB block offset
    src += (mb_group_id / 2) * CHANNEL_BLOCK_OFFSET * IC_NCHUNK; // MB offset
    src += oh * PIXEL_HEIGHT_OFFSET; // height offset
    src += ow * PIXEL_WIDTH_OFFSET; // width offset

    // Destination
    dst += (mb_group_id % 2) * MB_BLOCK / 2 * MB_OFFSET; // MB block offset
    dst += (mb_group_id / 2) * CHANNEL_BLOCK_OFFSET * OC_NCHUNK; // MB offset
    dst += CHANNEL_BLOCK_OFFSET * oc_group_id; //OC offset
    dst += oh * PIXEL_HEIGHT_OFFSET;
    dst += ow * PIXEL_WIDTH_OFFSET;

    // Weights
    wei += oc_group_id * KERNEL_BLOCK_OFFSET * IC_NCHUNK;

    // Output accumulators:
    // 8 MB (0-7) x 4 Kernels  (32 8bit ints)
    int8 C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    // 8 MB (8-15) x 4 Kernels  (32 8bit ints)
    int8 C10 = 0, C11 = 0, C12 = 0, C13 = 0;

    for (uint ic_block_id = 0; ic_block_id < IC_NCHUNK; ++ic_block_id) {
        uint8 S0, S1;
        int8 W0, W1, W2, W3;

        BLOCK_READ_SRC(S0, 0 * IC_BLOCK);
        BLOCK_READ_SRC(S1, 8 * IC_BLOCK);

        BLOCK_READ_WHT(W0, 0);
        BLOCK_READ_WHT(W1, 8 * IC_BLOCK);
        BLOCK_READ_WHT(W2, 16 * IC_BLOCK);
        BLOCK_READ_WHT(W3, 24 * IC_BLOCK);

        C00 = mmad8x8(S0, W0, C00);
        C01 = mmad8x8(S0, W1, C01);
        C02 = mmad8x8(S0, W2, C02);
        C03 = mmad8x8(S0, W3, C03);
        C10 = mmad8x8(S1, W0, C10);
        C11 = mmad8x8(S1, W1, C11);
        C12 = mmad8x8(S1, W2, C12);
        C13 = mmad8x8(S1, W3, C13);

        src += CHANNEL_BLOCK_OFFSET;
        wei += KERNEL_BLOCK_OFFSET;
    }

    float4 tmp;
    uint8 dst_pack;
    uint8 D0, D1, D2, D3;

#if WITH_BIAS
    float4 bia;
    BLOCK_READ_BIA(bia, (oc_group_id + sg_id) * OC_BLOCK);
    bia *= scales;
#define QUANTIZE_ADD_BIAS() tmp = fma(tmp, (float4)scales, bia);
#else
#define QUANTIZE_ADD_BIAS() tmp *= scales;
#endif

#if WITH_SUM
    D0.s0123 = as_uint4(intel_sub_group_block_read_uc16((__global uchar *)dst));
    D0.s4567 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[4 * OC_BLOCK]));
    D1.s0123 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[8 * OC_BLOCK]));
    D1.s4567 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[12 * OC_BLOCK]));

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
        intel_sub_group_block_write_uc16( \
                &dst[mb_stride * OC_BLOCK], as_uchar16(dst_pack.s0123)); \
        intel_sub_group_block_write_uc16(&dst[mb_stride * OC_BLOCK + 16 * 8], \
                as_uchar16(dst_pack.s4567)); \
    } while (0)

    STORE_DST(C00, C01, C02, C03, D0, 0);
    STORE_DST(C10, C11, C12, C13, D1, 8);
}

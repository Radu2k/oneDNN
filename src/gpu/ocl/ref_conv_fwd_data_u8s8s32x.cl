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

#include "gpu/ocl/ocl_types.h"

#define ODHW_SIZE (OD * OH * OW)
#define IDHW_SIZE (ID * IH * IW)
#define KDHW_SIZE (KD * KH * KW)
#define OCG (OC / G)
#define ICG (IC / G)

__kernel void ref_conv_fwd(const __global uchar *src, const __global char *wei,
        const __global float *bias, __global DATA_T *dst,
        float relu_negative_slope, float sum_scale, float scales) {
    const int osp = get_global_id(0);
    const int od = osp / (OW * OH);
    const int ohw = osp % (OW * OH);
    const int ow = ohw % OW;
    const int oh = ohw / OW;
    const int oc = get_global_id(1);
    const int mbg = get_global_id(2);
    const int mb = mbg / G;
    const int g = mbg % G;

    ACC_DATA_T sum = WITH_BIAS ? bias[g * OCG + oc] : 0.0;

    for (int ic = 0; ic < ICG; ++ic)
        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {

                    const int id = od * SD - PD + kd * (1 + KDD);
                    const int ih = oh * SH - PH + kh * (1 + KDH);
                    const int iw = ow * SW - PW + kw * (1 + KDW);

                    if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0
                            || iw >= IW)
                        continue;

                    sum += src[SRC_OFF(mb, g * ICG + ic, id, ih, iw)]
                            * wei[g * OCG * ICG * KDHW_SIZE
                                    + oc * ICG * KDHW_SIZE + ic * KDHW_SIZE
                                    + kd * KH * KW + kh * KW + kw];
                }

    dst += DST_OFF(mb, g * OCG + oc, od, oh, ow);

    sum *= scales;

#if WITH_SUM_ELTWISE == 1
#if SUM_SCALE == 1
    sum += dst[0];
#else
    sum = fma(dst[0], sum_scale, sum);
#endif
    if (sum < 0) sum *= relu_negative_slope;
#else
#if WITH_RELU == 1
    if (sum < 0) sum *= relu_negative_slope;
#endif
#if WITH_SUM == 1
#if SUM_SCALE == 1
    sum += dst[0];
#else
    sum = fma(dst[0], sum_scale, sum);
#endif
#endif
#endif

    dst[0] = CONVERT_DATA_T(sum);
}

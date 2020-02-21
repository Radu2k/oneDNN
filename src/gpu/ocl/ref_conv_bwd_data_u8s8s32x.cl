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

__kernel void ref_conv_bwd_data(__global uchar *diff_src,
        const __global char *wei, const __global float *bias,
        const __global DATA_T *diff_dst) {
    const int isp = get_global_id(0);
    const int id = isp / (IW * IH);
    const int ihw = isp % (IW * IH);
    const int iw = ihw % IW;
    const int ih = ihw / IW;
    const int ic = get_global_id(1);
    const int mbg = get_global_id(2);
    const int mb = mbg / G;
    const int g = mbg % G;

    float d = WITH_BIAS ? bias[g * ICG + ic] : 0.0;

    for (int oc = 0; oc < OCG; ++oc)
        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {
                    if (iw + PW < kw * (1 + KDW) || ih + PH < kh * (1 + KDH)
                            || id + PD < kd * (1 + KDD))
                        continue;
                    int ow = iw - kw * (1 + KDW) + PW;
                    int oh = ih - kh * (1 + KDH) + PH;
                    int od = id - kd * (1 + KDD) + PD;
                    if (ow % SW != 0 || oh % SH != 0 || od % SD != 0) continue;

                    ow /= SW;
                    oh /= SH;
                    od /= SD;
                    if (oh < OH && ow < OW && od < OD) {
                        d += diff_dst[DST_OFF(mb, g * OCG + oc, od, oh, ow)]
                                * wei[g * OCG * ICG * KDHW_SIZE
                                        + oc * ICG * KDHW_SIZE + ic * KDHW_SIZE
                                        + kd * KH * KW + kh * KW + kw];
                    }
                }
    diff_src[SRC_OFF(mb, g * ICG + ic, id, ih, iw)] = convert_uchar_sat_rte(d);
}

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

#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#if !COPY_A && !COPY_B
#error Source matrix not defined.
#endif

inline ushort2 masked_block_read_us2(global ushort *p, int rem) {
    ushort2 v;
    int lid = get_sub_group_local_id();
    int sg = get_sub_group_size();

    v.s0 = (lid < rem) ? p[lid] : 0;
    v.s1 = (lid + sg < rem) ? p[lid + sg] : 0;

    return v;
}

inline ushort4 masked_block_read_us4(global ushort *p, int rem) {
    ushort4 v;
    int lid = get_sub_group_local_id();
    int sg = get_sub_group_size();

    v.s0 = (lid < rem) ? p[lid] : 0;
    v.s1 = (lid + sg < rem) ? p[lid + sg] : 0;
    v.s2 = (lid + 2 * sg < rem) ? p[lid + 2 * sg] : 0;
    v.s3 = (lid + 3 * sg < rem) ? p[lid + 3 * sg] : 0;

    return v;
}

#if COPY_A

#define UNROLL_M 32
#define UNROLL_K 16

#if !COPY_TRANS

#define REPACK_REG(rr, cc) \
    blk_r[rr].s##cc = (((uint)c[2 * cc + 1].s##rr) << 16) | c[2 * cc].s##rr

#define REPACK_CC(cc) \
    REPACK_REG(0, cc); \
    REPACK_REG(1, cc); \
    REPACK_REG(2, cc); \
    REPACK_REG(3, cc)

#define REPACK \
    REPACK_CC(0); \
    REPACK_CC(1); \
    REPACK_CC(2); \
    REPACK_CC(3); \
    REPACK_CC(4); \
    REPACK_CC(5); \
    REPACK_CC(6); \
    REPACK_CC(7)

// Each thread packs a 32x16 block of A.
// Nontranspose A copy.
__attribute__((intel_reqd_sub_group_size(8))) kernel void
gen12hp_systolic_gemm_copy(long m, long k, global ushort *a, long offseta,
        long lda, global ushort *a_packed, int offseta_packed, int lda_packed) {

    uint m0 = (sub_group_broadcast(get_global_id(0), 0) / 8) * UNROLL_M;
    uint k0 = get_global_id(1) * UNROLL_K;
    int mrem = m - m0;
    int krem = k - k0;
    bool aligned = ((as_long(a) | lda | offseta) & 1) == 0;

    a += offseta + m0 + k0 * lda;
    a_packed += offseta_packed + m0 * lda_packed + k0 * UNROLL_M;

    // Read all columns.
    ushort4 c[UNROLL_K];

    if (mrem >= UNROLL_M && krem >= UNROLL_K && aligned) {
        for (int h = 0; h < UNROLL_K; h++)
            c[h] = intel_sub_group_block_read_us4(a + h * lda);
    } else {
        for (int h = 0; h < UNROLL_K; h++)
            if (h < krem)
                c[h] = masked_block_read_us4(a + h * lda, mrem);
            else
                c[h] = 0;
    }

    // Rearrange.
    uint8 blk_r[UNROLL_M / 8];
    REPACK;

    // Write out.
    for (int rr = 0; rr < UNROLL_M / 8; rr++)
        intel_sub_group_block_write8(
                (global uint *)(a_packed + rr * UNROLL_K * 8), blk_r[rr]);
}

#else /* COPY_TRANS */

// Transpose A copy.
__attribute__((intel_reqd_sub_group_size(8))) kernel void
gen12hp_systolic_gemm_copy(long m, long k, global ushort *a, long offseta,
        long lda, global ushort *a_packed, int offseta_packed, int lda_packed) {

    int lid = get_sub_group_local_id();
    uint m0 = (sub_group_broadcast(get_global_id(0), 0) / 8) * UNROLL_M;
    uint k0 = get_global_id(1) * UNROLL_K;
    int mrem = m - m0;
    int krem = k - k0;

    a += offseta + m0 * lda + k0;
    a_packed += offseta_packed + m0 * lda_packed + k0 * UNROLL_M;

    for (int rr = 0; rr < UNROLL_M / 8; rr++, mrem -= 8) {
        ushort2 regs[8];

        if (mrem >= UNROLL_M && krem >= UNROLL_K) {
            for (int cc = 0; cc < UNROLL_K / 2; cc++)
                regs[cc] = vload2(0, a + ((rr * 8) + lid) * lda + (cc * 2));
        } else {
            for (int cc = 0; cc < UNROLL_K / 2; cc++) {
                regs[cc] = 0;
                if ((2 * cc + 1) < krem) {
                    if (lid < mrem)
                        regs[cc] = vload2(
                                0, a + ((rr * 8) + lid) * lda + (cc * 2));
                } else if (2 * cc < krem) {
                    if (lid < mrem)
                        regs[cc].s0 = a[((rr * 8) + lid) * lda + (cc * 2)];
                }
            }
        }

        uint8 blk_r;
        blk_r.s0 = as_uint(regs[0]);
        blk_r.s1 = as_uint(regs[1]);
        blk_r.s2 = as_uint(regs[2]);
        blk_r.s3 = as_uint(regs[3]);
        blk_r.s4 = as_uint(regs[4]);
        blk_r.s5 = as_uint(regs[5]);
        blk_r.s6 = as_uint(regs[6]);
        blk_r.s7 = as_uint(regs[7]);

        intel_sub_group_block_write8(
                (global uint *)(a_packed + rr * UNROLL_K * 8), blk_r);
    }
}

#endif /* !COPY_TRANS */
#endif /* COPY_A */

#if COPY_B

#define UNROLL_K 16
#define UNROLL_N 48

#define REPACK_CC(cc) \
    do { \
        colgroups[cc].s01 = cols[cc * 4]; \
        colgroups[cc].s23 = cols[cc * 4 + 1]; \
        colgroups[cc].s45 = cols[cc * 4 + 2]; \
        colgroups[cc].s67 = cols[cc * 4 + 3]; \
    } while (false)

#if !COPY_TRANS

// Nontranspose B copy.
__attribute__((intel_reqd_sub_group_size(8))) kernel void
gen12hp_systolic_gemm_copy(long k, long n, global ushort *b, long offsetb,
        long ldb, global ushort *b_packed, int offsetb_packed, int ldb_packed) {

    uint k0 = (sub_group_broadcast(get_global_id(0), 0) / 8) * UNROLL_K;
    uint n0 = get_global_id(1) * UNROLL_N;
    int krem = k - k0;
    int nrem = n - n0;
    bool aligned = ((as_long(b) | ldb | offsetb) & 1) == 0;

    b += offsetb + k0 + n0 * ldb;
    b_packed += offsetb_packed + n0 * ldb_packed + k0 * UNROLL_N;

    // Read all columns.
    ushort2 cols[UNROLL_N];
    if (krem >= UNROLL_K && nrem >= UNROLL_N && aligned) {
        for (int c = 0; c < UNROLL_N; c++)
            cols[c] = intel_sub_group_block_read_us2(b + c * ldb);
    } else {
        for (int c = 0; c < UNROLL_N; c++)
            if (c < nrem)
                cols[c] = masked_block_read_us2(b + c * ldb, krem);
            else
                cols[c] = 0;
    }

    // Repack.
    ushort8 colgroups[UNROLL_N / 4];
    for (int cc = 0; cc < UNROLL_N / 4; cc++)
        REPACK_CC(cc);

    // Write out.
    for (int cc = 0; cc < UNROLL_N / 4; cc++)
        intel_sub_group_block_write_us8(
                b_packed + cc * 4 * UNROLL_K, colgroups[cc]);
}

#else /* COPY_TRANS */

// Transpose B copy.
__attribute__((intel_reqd_sub_group_size(16))) kernel void
gen12hp_systolic_gemm_copy(long k, long n, global ushort *b, long offsetb,
        long ldb, global ushort *b_packed, int offsetb_packed, int ldb_packed) {

    int lid = get_sub_group_local_id();
    uint k0 = (sub_group_broadcast(get_global_id(0), 0) / 16) * UNROLL_K;
    uint n0 = get_global_id(1) * UNROLL_N;
    int krem = k - k0;
    int nrem = n - n0;

    b += offsetb + n0 + k0 * ldb;
    b_packed += offsetb_packed + n0 * ldb_packed + k0 * UNROLL_N;

    // Read all columns, in pairs.
    ushort2 cols[UNROLL_N / 2];
    if (krem >= UNROLL_K && nrem >= UNROLL_N) {
        for (int cc = 0; cc < UNROLL_N / 2; cc++)
            cols[cc] = vload2(0, b + cc * 2 + lid * ldb);
    } else {
        for (int cc = 0; cc < UNROLL_N / 2; cc++) {
            cols[cc] = 0;
            if ((2 * cc + 1) < nrem) {
                if (lid < krem) cols[cc] = vload2(0, b + cc * 2 + lid * ldb);
            } else if (2 * cc < nrem) {
                if (lid < krem) cols[cc].s0 = b[cc * 2 + lid * ldb];
            }
        }
    }

    // Repack.
    ushort8 colgroups[UNROLL_N / 8];
    for (int cc = 0; cc < UNROLL_N / 8; cc++)
        REPACK_CC(cc);

    // Write out.
    for (int cc = 0; cc < UNROLL_N / 8; cc++)
        intel_sub_group_block_write_us8(
                b_packed + cc * 8 * UNROLL_K, colgroups[cc]);
}

#endif /* !COPY_TRANS */
#endif /* COPY_B */

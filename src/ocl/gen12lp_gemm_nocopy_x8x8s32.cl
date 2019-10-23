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
//#define IMAD_SUPPORTED
#include "ocl/ocl_math_utils.h"

#include "ocl/ocl_types.h"
#if WITH_ELTWISE == 1
#include "ocl/ocl_post_ops.h"
#endif

#if defined(S8S8)
#define CHARA char
#define CHARA2 char2
#define CHARA4 char4
#define CHARB char
#define CHARB4 char4
#define AS_CHARA as_char
#define AS_CHARA2 as_char2
#define AS_CHARA4 as_char4
#define AS_CHARB as_char
#define AS_CHARB2 as_char2
#define AS_CHARB4 as_char4
#endif

#if defined(U8S8)
#define CHARA uchar
#define CHARA2 uchar2
#define CHARA4 uchar4
#define CHARB char
#define CHARB4 char4
#define AS_CHARA as_uchar
#define AS_CHARA2 as_uchar2
#define AS_CHARA4 as_uchar4
#define AS_CHARB as_char
#define AS_CHARB2 as_char2
#define AS_CHARB4 as_char4
#endif

#if defined(S8U8)
#define CHARA char
#define CHARA2 char2
#define CHARA4 char4
#define CHARB uchar
#define CHARB4 uchar4
#define AS_CHARA as_char
#define AS_CHARA2 as_char2
#define AS_CHARA4 as_char4
#define AS_CHARB as_uchar
#define AS_CHARB2 as_uchar2
#define AS_CHARB4 as_uchar4
#endif

#if defined(U8U8)
#define CHARA uchar
#define CHARA2 uchar2
#define CHARA4 uchar4
#define CHARB uchar
#define CHARB4 uchar4
#define AS_CHARA as_uchar
#define AS_CHARA2 as_uchar2
#define AS_CHARA4 as_uchar4
#define AS_CHARB as_uchar
#define AS_CHARB2 as_uchar2
#define AS_CHARB4 as_uchar4
#endif

#define INTC int
#define INTC4 int4

#if defined(TN)
#define DO_FMA DO_FMA_TN
#endif
#if defined(NN)
#define DO_FMA DO_FMA_NN
#endif
#if defined(NT)
#define DO_FMA DO_FMA_NT
#endif
#if defined(TT)
#define DO_FMA DO_FMA_TT
#endif

#define ADDAROW(z) \
    do { \
        sumRowA[z] = ai[z].s0 + ai[z].s1 + ai[z].s2 + ai[z].s3; \
    } while (0)

#define ADDAROWT() \
    do { \
        sumRowA[0] = ai[0].s0 + ai[1].s0 + ai[2].s0 + ai[3].s0; \
        sumRowA[1] = ai[0].s1 + ai[1].s1 + ai[2].s1 + ai[3].s1; \
    } while (0)

#define ADDBCOL() \
    do { \
        sumColB = bi.s0 + bi.s1 + bi.s2 + bi.s3; \
    } while (0)

#define ADDBCOLT() \
    do { \
        sumColB = bi[0] + bi[1] + bi[2] + bi[3]; \
    } while (0)

#ifdef ALIGNED
#define VLOADA4(z, p) \
    do { \
        ai[z] = *((global CHARA4 *)p); \
    } while (0)
#else
#define VLOADA4(z, p) \
    do { \
        ai[z].s0 = *(p + 0); \
        ai[z].s1 = *(p + 1); \
        ai[z].s2 = *(p + 2); \
        ai[z].s3 = *(p + 3); \
    } while (0)
#endif

#ifdef ALIGNED
#define BLOCKREADA(h, hh) \
    do { \
        ai[hh] = AS_CHARA2(intel_sub_group_block_read_uc2( \
                (global uchar *)(a_ptrs[hh] + h * lda))); \
    } while (0)
#else
#define BLOCKREADA(h, hh) \
    do { \
        ai[hh].s0 = *((a_ptrs[hh] + h * lda) + 0); \
        ai[hh].s1 = *((a_ptrs[hh] + h * lda) + 16); \
    } while (0)
#endif

#ifdef ALIGNED
#define BLOCKREADB(h, hh) \
    do { \
        bi[hh] = AS_CHARB(intel_sub_group_block_read_uc( \
                (global uchar *)(b_ptrs[hh] + h * ldb))); \
    } while (0)
#else
#define BLOCKREADB(h, hh) \
    do { \
        bi[hh] = *(b_ptrs[hh] + h * ldb); \
    } while (0)
#endif

#ifdef ALIGNED
#define VLOADB4(p) \
    do { \
        bi = *((global CHARB4 *)p); \
    } while (0)
#else
#define VLOADB4(p) \
    do { \
        bi.s0 = *(p + 0); \
        bi.s1 = *(p + 1); \
        bi.s2 = *(p + 2); \
        bi.s3 = *(p + 3); \
    } while (0)
#endif

#define LOADA_REM(z, p) \
    do { \
        if (krem == 3) { \
            ai[z].s0 = *(p + 0); \
            ai[z].s1 = *(p + 1); \
            ai[z].s2 = *(p + 2); \
        } \
        if (krem == 2) { \
            ai[z].s0 = *(p + 0); \
            ai[z].s1 = *(p + 1); \
        } \
        if (krem == 1) { ai[z].s0 = *(p + 0); } \
    } while (0)

#define LOADB_REM(p) \
    do { \
        if (krem == 3) { \
            bi.s0 = *(p + 0); \
            bi.s1 = *(p + 1); \
            bi.s2 = *(p + 2); \
        } \
        if (krem == 2) { \
            bi.s0 = *(p + 0); \
            bi.s1 = *(p + 1); \
        } \
        if (krem == 1) { bi.s0 = *(p + 0); } \
    } while (0)

#define COPYA() \
    do { \
        ait[0].s0 = ai[0].s0; \
        \ 
        ait[0] \
                .s1 \
                = ai[1].s0; \
        ait[0].s2 = ai[2].s0; \
        ait[0].s3 = ai[3].s0; \
        ait[1].s0 = ai[0].s1; \
        ait[1].s1 = ai[1].s1; \
        ait[1].s2 = ai[2].s1; \
        ait[1].s3 = ai[3].s1; \
    } while (0)

#define COPYB() \
    do { \
        biit.s0 = bi[0]; \
        \ 
        biit.s1 = bi[1]; \
        biit.s2 = bi[2]; \
        biit.s3 = bi[3]; \
    } while (0)

#define DO_FMA_TN(h, i) \
    do { \
        ci[0][i] = IMAD(AS_CHARB4(sub_group_broadcast(as_int(bi), i)), \
                AS_CHARA4(ai[0]), ci[0][i]); \
        ci[1][i] = IMAD(AS_CHARB4(sub_group_broadcast(as_int(bi), i)), \
                AS_CHARA4(ai[1]), ci[1][i]); \
    } while (0)

#define DO_FMA_NN(h, i) \
    do { \
        ci[0][i] = IMAD(AS_CHARB4(sub_group_broadcast(as_int(bi), i)), \
                AS_CHARA4(ait[0]), ci[0][i]); \
        ci[1][i] = IMAD(AS_CHARB4(sub_group_broadcast(as_int(bi), i)), \
                AS_CHARA4(ait[1]), ci[1][i]); \
    } while (0)

#define DO_FMA_NT(h, i) \
    do { \
        ci[0][i] = IMAD(AS_CHARB4(sub_group_broadcast(as_int(biit), i)), \
                AS_CHARA4(ait[0]), ci[0][i]); \
        ci[1][i] = IMAD(AS_CHARB4(sub_group_broadcast(as_int(biit), i)), \
                AS_CHARA4(ait[1]), ci[1][i]); \
    } while (0)

#define DO_FMA_TT(h, i) \
    do { \
        ci[0][i] = IMAD(AS_CHARB4(sub_group_broadcast(as_int(biit), i)), \
                AS_CHARA4(ai[0]), ci[0][i]); \
        ci[1][i] = IMAD(AS_CHARB4(sub_group_broadcast(as_int(biit), i)), \
                AS_CHARA4(ai[1]), ci[1][i]); \
    } while (0)

#if WITH_ELTWISE == 1
#define POST_OP(val) \
    do { \
        if (apply_eltwise) \
            val = fwd_eltwise(val, eltwise_alpha, eltwise_beta); \
    } while (0)
#else
#define POST_OP(val)
#endif

#define FMA_I_LOOP(h) \
    do { \
        DO_FMA(h, 0); \
        DO_FMA(h, 1); \
        DO_FMA(h, 2); \
        DO_FMA(h, 3); \
        DO_FMA(h, 4); \
        DO_FMA(h, 5); \
        DO_FMA(h, 6); \
        DO_FMA(h, 7); \
        DO_FMA(h, 8); \
        DO_FMA(h, 9); \
        DO_FMA(h, 10); \
        DO_FMA(h, 11); \
        DO_FMA(h, 12); \
        DO_FMA(h, 13); \
        DO_FMA(h, 14); \
        DO_FMA(h, 15); \
    } while (0)

#define ADD_BOFF(i) \
    do { \
        ci[0][i] -= (INTC)bo * sumRowA[0]; \
        ci[1][i] -= (INTC)bo * sumRowA[1]; \
    } while (0)

#define ADD_BOFF_LOOP() \
    do { \
        ADD_BOFF(0); \
        ADD_BOFF(1); \
        ADD_BOFF(2); \
        ADD_BOFF(3); \
        ADD_BOFF(4); \
        ADD_BOFF(5); \
        ADD_BOFF(6); \
        ADD_BOFF(7); \
        ADD_BOFF(8); \
        ADD_BOFF(9); \
        ADD_BOFF(10); \
        ADD_BOFF(11); \
        ADD_BOFF(12); \
        ADD_BOFF(13); \
        ADD_BOFF(14); \
        ADD_BOFF(15); \
    } while (0)

#define ADD_AOFF(h, i) \
    do { \
        ci[0][i] -= ((INTC)ao * sub_group_broadcast(as_int(sumColB), i)) \
                - (h * (INTC)ao * (INTC)bo); \
        ci[1][i] -= ((INTC)ao * sub_group_broadcast(as_int(sumColB), i)) \
                - (h * (INTC)ao * (INTC)bo); \
    } while (0)

#define ADD_AOFF_LOOP(h) \
    do { \
        ADD_AOFF(h, 0); \
        ADD_AOFF(h, 1); \
        ADD_AOFF(h, 2); \
        ADD_AOFF(h, 3); \
        ADD_AOFF(h, 4); \
        ADD_AOFF(h, 5); \
        ADD_AOFF(h, 6); \
        ADD_AOFF(h, 7); \
        ADD_AOFF(h, 8); \
        ADD_AOFF(h, 9); \
        ADD_AOFF(h, 10); \
        ADD_AOFF(h, 11); \
        ADD_AOFF(h, 12); \
        ADD_AOFF(h, 13); \
        ADD_AOFF(h, 14); \
        ADD_AOFF(h, 15); \
    } while (0)

#define UPDATE_C_COL(i, betaZero) \
    do { \
        if (jrem > i) { \
            if (irem > 0) { \
                if (c_offset_type == 0) { \
                    INTC val = ((betaZero) ? 0 : *c) \
                            + ((!apply_co) ? 0 : co[0]) + ci[0][i]; \
                    POST_OP(val); \
                    *c = val; \
                } \
                \  
                if (c_offset_type == 1) { \
                    INTC val = ((betaZero) ? 0 : *c) \
                            + ((!apply_co) ? 0 : co[0]) + ci[0][i]; \
                    POST_OP(val); \
                    *c = val; \
                } \
                \  
                if (c_offset_type == 2) { \
                    INTC val = ((betaZero) ? 0 : *c) \
                            + ((!apply_co) ? 0 : co[i]) + ci[0][i]; \
                    POST_OP(val); \
                    *c = val; \
                } \
                \  
             \
            } \
            if (irem > 16) { \
                if (c_offset_type == 0) { \
                    INTC val = ((betaZero) ? 0 : *c2) \
                            + ((!apply_co) ? 0 : co[0]) + ci[1][i]; \
                    POST_OP(val); \
                    *c2 = val; \
                } \
                if (c_offset_type == 1) { \
                    INTC val = ((betaZero) ? 0 : *c2) \
                            + ((!apply_co) ? 0 : co[16]) + ci[1][i]; \
                    POST_OP(val); \
                    *c2 = val; \
                } \
                if (c_offset_type == 2) { \
                    INTC val = ((betaZero) ? 0 : *c2) \
                            + ((!apply_co) ? 0 : co[i]) + ci[1][i]; \
                    POST_OP(val); \
                    *c2 = val; \
                } \
            } \
        } \
        c = c + ldc; \
        c2 = c2 + ldc; \
    } while (0)

#define UPDATE_C(betaZero) \
    do { \
        UPDATE_C_COL(0, betaZero); \
        UPDATE_C_COL(1, betaZero); \
        UPDATE_C_COL(2, betaZero); \
        UPDATE_C_COL(3, betaZero); \
        UPDATE_C_COL(4, betaZero); \
        UPDATE_C_COL(5, betaZero); \
        UPDATE_C_COL(6, betaZero); \
        UPDATE_C_COL(7, betaZero); \
        UPDATE_C_COL(8, betaZero); \
        UPDATE_C_COL(9, betaZero); \
        UPDATE_C_COL(10, betaZero); \
        UPDATE_C_COL(11, betaZero); \
        UPDATE_C_COL(12, betaZero); \
        UPDATE_C_COL(13, betaZero); \
        UPDATE_C_COL(14, betaZero); \
        UPDATE_C_COL(15, betaZero); \
    } while (0)

#ifdef TN
__attribute__((intel_reqd_sub_group_size(16))) kernel void
gen12lp_gemm_compute_x8x8s32_kernel(global CHARA *a, global CHARB *b,
        global INTC *c, int offsetA, int offsetB, int offsetC, int lda, int ldb,
        int ldc, int m, int n, int k, int beta, CHARA ao, CHARB bo,
        global INTC *co, int offsetCO, int apply_co, local CHARA *sa,
        local CHARB *sb, int apply_eltwise, float eltwise_alpha,
        float eltwise_beta) {

    // unroll_m = 32, unroll_n = 16, subgroupsize = 16
    CHARA4 ai[2]; // 32x4 block of A, 2x 16x4 scattered access
    CHARB4 bi; // 4x16 block of B, 1x 4x16 scattered access
    INTC ci[2][16]; // 32x16 block of C, 16x1 x 2x16 scattered access

    INTC sumRowA[2] = {0, 0};
    INTC sumColB = 0;

    int idM = get_group_id(0);
    int idN = get_group_id(1);
    int idlM = get_local_id(0);
    int idlN = get_local_id(1);
    int lid = get_sub_group_local_id();
    int lsm = 32; //get_local_size(0)
    int lsn = 8; //get_local_size(1)

    int i0 = (idM * lsm / 16) * 32 + (get_local_id(0) / 16) * 32;
    int j0 = idlN * 16 + (idN * lsn * 16);

    int irem = m - i0 - lid;
    int jrem = n - j0;

    if (irem < 0) irem = 0;
    if (jrem < 0) jrem = 0;

    a += offsetA + (i0 * lda) + (lid * lda);
    b += offsetB + (j0 * ldb) + (lid * ldb);
    c += offsetC + (i0) + (j0 * ldc) + lid;

    int c_offset_type = 0; //0:Fixed, 1:Column, 2:Row

#ifdef FF
    co += offsetCO;
    c_offset_type = 0;
#endif
#ifdef CC
    co += offsetCO + i0 + lid;
    c_offset_type = 1;
#endif
#ifdef RR
    co += offsetCO + (j0);
    c_offset_type = 2;
#endif

    global CHARA *a_ptrs[2] = {a, a + 16 * lda};
    global CHARB *b_ptrs = {b};

    for (int y = 0; y < 16; y++) {
        for (int z = 0; z < 2; z++) {
            ci[z][y] = 0;
        }
    }

    int k_align = k & ~3;
    for (int h = 0; h < k_align; h += 4) {

        // Load A
        for (int z = 0; z < 2; z++) {
            VLOADA4(z, (a_ptrs[z] + h));
#ifdef BOFFNONZERO
            ADDAROW(z);
#endif
        }
        // Load B
        VLOADB4((b_ptrs + h));
#ifdef AOFFNONZERO
        ADDBCOL();
#endif
        // Compute
        FMA_I_LOOP(0);

#ifdef BOFFNONZERO
        ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
        ADD_AOFF_LOOP(4);
#endif
    }

    // Remainder Loop
    int krem = k & 3;
    if (krem > 0) {
        ai[0] = 0;
        ai[1] = 0;
        bi = 0;
        // Load A
        for (int z = 0; z < 2; z++) {
            LOADA_REM(z, (a_ptrs[z] + k_align));
#ifdef BOFFNONZERO
            ADDAROW(z);
#endif
        }
        // Load B
        LOADB_REM((b_ptrs + k_align));
#ifdef AOFFNONZERO
        ADDBCOL();
#endif
        // Compute
        FMA_I_LOOP(0);
#ifdef BOFFNONZERO
        ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
        ADD_AOFF_LOOP(krem);
#endif
    }

    // Store C
    global INTC *c2 = c + 16;

    // Update C
    if (beta == 0)
        UPDATE_C(1);
    else
        UPDATE_C(0);
}
#endif //TN

#ifdef NN
__attribute__((intel_reqd_sub_group_size(16))) kernel void
gen12lp_gemm_compute_x8x8s32_kernel(global CHARA *a, global CHARB *b,
        global INTC *c, int offsetA, int offsetB, int offsetC, int lda, int ldb,
        int ldc, int m, int n, int k, int beta, CHARA ao, CHARB bo,
        global INTC *co, int offsetCO, int apply_co, local CHARA *sa,
        local CHARB *sb, int apply_eltwise, float eltwise_alpha,
        float eltwise_beta) {

    // unroll_m = 32, unroll_n = 16, subgroupsize = 16
    CHARA2 ai[4]; // 32x4 block of A, 4x 32x1 block access
    CHARB4 bi; // 4x16 block of B, 1x 4x16 scattered access
    INTC ci[2][16]; // 32x16 block of C, 16x1 x 2x16 scattered access

    INTC sumRowA[2] = {0, 0};
    INTC sumColB = 0;

    CHARA4 ait[2];

    int idM = get_group_id(0);
    int idN = get_group_id(1);
    int idlM = get_local_id(0);
    int idlN = get_local_id(1);
    int lid = get_sub_group_local_id();
    int lsm = 32; //get_local_size(0)
    int lsn = 8; //get_local_size(1)

    int i0 = (idM * lsm / 16) * 32 + (get_local_id(0) / 16) * 32;
    int j0 = idlN * 16 + (idN * lsn * 16);

    int irem = m - i0 - lid;
    int jrem = n - j0;

    if (irem < 0) irem = 0;
    if (jrem < 0) jrem = 0;
#ifdef ALIGNED
    a += offsetA + i0;
#else
    a += offsetA + i0 + lid;
#endif

    b += offsetB + (j0 * ldb) + (lid * ldb);
    c += offsetC + (i0) + (j0 * ldc) + lid;

    int c_offset_type = 0; //0:Fixed, 1:Column, 2:Row

#ifdef FF
    co += offsetCO;
    c_offset_type = 0;
#endif
#ifdef CC
    co += offsetCO + i0 + lid;
    c_offset_type = 1;
#endif
#ifdef RR
    co += offsetCO + (j0);
    c_offset_type = 2;
#endif

    global CHARA *a_ptrs[4] = {a, a + 1 * lda, a + 2 * lda, a + 3 * lda};
    global CHARB *b_ptrs = {b};

    for (int y = 0; y < 16; y++) {
        for (int z = 0; z < 2; z++) {
            ci[z][y] = 0;
        }
    }

    int k_align = k & ~3;
    for (int h = 0; h < k_align; h += 4) {

        // Load A
        for (int hh = 0; hh < 4; hh++) {
            BLOCKREADA(h, hh);
#ifdef BOFFNONZERO
            ADDAROWT();
#endif
        }
        // Copy A
        COPYA();
        // Load B
        VLOADB4((b_ptrs + h));
#ifdef AOFFNONZERO
        ADDBCOL();
#endif

        // Compute
        FMA_I_LOOP(0);
#ifdef BOFFNONZERO
        ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
        ADD_AOFF_LOOP(4);
#endif
    }

    // Remainder Loop
    int krem = k & 3;
    if (krem > 0) {
        ai[0] = 0;
        ai[1] = 0;
        ai[2] = 0;
        ai[3] = 0;
        bi = 0;
        // Load A
        for (int hh = 0; hh < krem; hh++) {
            BLOCKREADA(k_align, hh);
#ifdef BOFFNONZERO
            ADDAROWT();
#endif
        }

        // Copy A
        COPYA();

        // Load B
        LOADB_REM((b_ptrs + k_align));
#ifdef AOFFNONZERO
        ADDBCOL();
#endif
        // Compute
        FMA_I_LOOP(0);
#ifdef BOFFNONZERO
        ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
        ADD_AOFF_LOOP(krem);
#endif
    }

    // Store C
    global INTC *c2 = c + 16;

    // Update C
    if (beta == 0)
        UPDATE_C(1);
    else
        UPDATE_C(0);
}
#endif // NN

#ifdef NT
__attribute__((intel_reqd_sub_group_size(16))) kernel void
gen12lp_gemm_compute_x8x8s32_kernel(global CHARA *a, global CHARB *b,
        global INTC *c, int offsetA, int offsetB, int offsetC, int lda, int ldb,
        int ldc, int m, int n, int k, int beta, CHARA ao, CHARB bo,
        global INTC *co, int offsetCO, int apply_co, local CHARA *sa,
        local CHARB *sb, int apply_eltwise, float eltwise_alpha,
        float eltwise_beta) {

    // unroll_m = 32, unroll_n = 16, subgroupsize = 16
    CHARA2 ai[4]; // 32x4 block of A, 4x 32x1 block access
    CHARB bi[4]; // 4x16 block of B, 4x 1x16 block access
    INTC ci[2][16]; // 32x16 block of C, 16x1 x 2x16 scattered access

    INTC sumRowA[2] = {0, 0};
    INTC sumColB = 0;

    CHARA4 ait[2];
    CHARA4 biit;

    int idM = get_group_id(0);
    int idN = get_group_id(1);
    int idlM = get_local_id(0);
    int idlN = get_local_id(1);
    int lid = get_sub_group_local_id();
    int lsm = 32; //get_local_size(0)
    int lsn = 8; //get_local_size(1)

    int i0 = (idM * lsm / 16) * 32 + (get_local_id(0) / 16) * 32;
    int j0 = idlN * 16 + (idN * lsn * 16);

    int irem = m - i0 - lid;
    int jrem = n - j0;

    if (irem < 0) irem = 0;
    if (jrem < 0) jrem = 0;
#ifdef ALIGNED
    a += offsetA + i0;
#else
    a += offsetA + i0 + lid;
#endif

#ifdef ALIGNED
    b += offsetB + j0;
#else
    b += offsetB + j0 + lid;
#endif

    c += offsetC + (i0) + (j0 * ldc) + lid;

    int c_offset_type = 0; //0:Fixed, 1:Column, 2:Row

#ifdef FF
    co += offsetCO;
    c_offset_type = 0;
#endif
#ifdef CC
    co += offsetCO + i0 + lid;
    c_offset_type = 1;
#endif
#ifdef RR
    co += offsetCO + (j0);
    c_offset_type = 2;
#endif

    global CHARA *a_ptrs[4] = {a, a + 1 * lda, a + 2 * lda, a + 3 * lda};
    global CHARB *b_ptrs[4] = {b, b + 1 * ldb, b + 2 * ldb, b + 3 * ldb};

    for (int y = 0; y < 16; y++) {
        for (int z = 0; z < 2; z++) {
            ci[z][y] = 0;
        }
    }

    int k_align = k & ~3;
    for (int h = 0; h < k_align; h += 4) {

        // Load A
        for (int hh = 0; hh < 4; hh++) {
            BLOCKREADA(h, hh);
#ifdef BOFFNONZERO
            ADDAROWT();
#endif
        }
        // Copy A
        COPYA();
        // Load B
        for (int hh = 0; hh < 4; hh++) {
            BLOCKREADB(h, hh);
#ifdef AOFFNONZERO
            ADDBCOLT();
#endif
        }
        // Copy B
        COPYB();

        // Compute
        FMA_I_LOOP(0);
#ifdef BOFFNONZERO
        ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
        ADD_AOFF_LOOP(4);
#endif
    }

    // Remainder Loop
    int krem = k & 3;
    if (krem > 0) {
        ai[0] = 0;
        ai[1] = 0;
        ai[2] = 0;
        ai[3] = 0;

        bi[0] = 0;
        bi[1] = 0;
        bi[2] = 0;
        bi[3] = 0;
        // Load A
        for (int hh = 0; hh < krem; hh++) {
            BLOCKREADA(k_align, hh);
#ifdef BOFFNONZERO
            ADDAROWT();
#endif
        }
        // Copy A
        COPYA();

        // Load B
        for (int hh = 0; hh < krem; hh++) {
            BLOCKREADB(k_align, hh);
#ifdef AOFFNONZERO
            ADDBCOLT();
#endif
        }
        // Copy B
        COPYB();
        // Compute
        FMA_I_LOOP(0);
#ifdef BOFFNONZERO
        ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
        ADD_AOFF_LOOP(krem);
#endif
    }

    // Store C
    global INTC *c2 = c + 16;

    // Update C
    if (beta == 0)
        UPDATE_C(1);
    else
        UPDATE_C(0);
}
#endif // NT

#ifdef TT
__attribute__((intel_reqd_sub_group_size(16))) kernel void
gen12lp_gemm_compute_x8x8s32_kernel(global CHARA *a, global CHARB *b,
        global INTC *c, int offsetA, int offsetB, int offsetC, int lda, int ldb,
        int ldc, int m, int n, int k, int beta, CHARA ao, CHARB bo,
        global INTC *co, int offsetCO, int apply_co, local CHARA *sa,
        local CHARB *sb, int apply_eltwise, float eltwise_alpha,
        float eltwise_beta) {

    // unroll_m = 32, unroll_n = 16, subgroupsize = 16
    CHARA4 ai[2]; // 32x4 block of A, 2x 16x4 scattered access
    CHARB bi[4]; // 4x16 block of B, 4x 1x16 block access
    INTC ci[2][16]; // 32x16 block of C, 16x1 x 2x16 scattered access

    INTC sumRowA[2] = {0, 0};
    INTC sumColB = 0;

    CHARA4 biit;

    int idM = get_group_id(0);
    int idN = get_group_id(1);
    int idlM = get_local_id(0);
    int idlN = get_local_id(1);
    int lid = get_sub_group_local_id();
    int lsm = 32; //get_local_size(0)
    int lsn = 8; //get_local_size(1)

    int i0 = (idM * lsm / 16) * 32 + (get_local_id(0) / 16) * 32;
    int j0 = idlN * 16 + (idN * lsn * 16);

    int irem = m - i0 - lid;
    int jrem = n - j0;

    if (irem < 0) irem = 0;
    if (jrem < 0) jrem = 0;
    a += offsetA + (i0 * lda) + (lid * lda);

#ifdef ALIGNED
    b += offsetB + j0;
#else
    b += offsetB + j0 + lid;
#endif

    c += offsetC + (i0) + (j0 * ldc) + lid;

    int c_offset_type = 0; //0:Fixed, 1:Column, 2:Row

#ifdef FF
    co += offsetCO;
    c_offset_type = 0;
#endif
#ifdef CC
    co += offsetCO + i0 + lid;
    c_offset_type = 1;
#endif
#ifdef RR
    co += offsetCO + (j0);
    c_offset_type = 2;
#endif

    global CHARA *a_ptrs[2] = {a, a + 16 * lda};
    global CHARB *b_ptrs[4] = {b, b + 1 * ldb, b + 2 * ldb, b + 3 * ldb};

    for (int y = 0; y < 16; y++) {
        for (int z = 0; z < 2; z++) {
            ci[z][y] = 0;
        }
    }

    int k_align = k & ~3;
    for (int h = 0; h < k_align; h += 4) {

        // Load A
        for (int z = 0; z < 2; z++) {
            VLOADA4(z, ((a_ptrs[z]) + h));
#ifdef BOFFNONZERO
            ADDAROW(z);
#endif
        }
        // Load B
        for (int hh = 0; hh < 4; hh++) {
            BLOCKREADB(h, hh);
#ifdef AOFFNONZERO
            ADDBCOLT();
#endif
        }
        // Copy B
        COPYB();

        // Compute
        FMA_I_LOOP(0);
#ifdef BOFFNONZERO
        ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
        ADD_AOFF_LOOP(4);
#endif
    }

    // Remainder Loop
    int krem = k & 3;
    if (krem > 0) {
        ai[0] = 0;
        ai[1] = 0;

        bi[0] = 0;
        bi[1] = 0;
        bi[2] = 0;
        bi[3] = 0;

        // Load A
        for (int z = 0; z < 2; z++) {
            LOADA_REM(z, ((a_ptrs[z]) + k_align));
#ifdef BOFFNONZERO
            ADDAROW(z);
#endif
        }
        // Load B
        for (int hh = 0; hh < krem; hh++) {
            BLOCKREADB(k_align, hh);
#ifdef AOFFNONZERO
            ADDBCOLT();
#endif
        }
        // Copy B
        COPYB();
        // Compute
        FMA_I_LOOP(0);
#ifdef BOFFNONZERO
        ADD_BOFF_LOOP();
#endif
#ifdef AOFFNONZERO
        ADD_AOFF_LOOP(krem);
#endif
    }

    // Store C
    global INTC *c2 = c + 16;

    // Update C
    if (beta == 0)
        UPDATE_C(1);
    else
        UPDATE_C(0);
}
#endif // TT

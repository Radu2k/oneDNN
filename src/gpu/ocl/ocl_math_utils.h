/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef GPU_OCL_OCL_MATH_UTILS_H
#define GPU_OCL_OCL_MATH_UTILS_H

ulong8 __builtin_IB_simd_block_read_8_global_l(const __global ulong *);
void __builtin_IB_simd_block_write_8_global_l(__global ulong *, ulong8);
void __builtin_IB_simd_block_write_16_global_h(__global ushort *, ushort16);
ushort16 __builtin_IB_simd_block_read_16_global_h(const __global ushort *);

#ifdef cl_intel_subgroups_char
uchar2 __attribute__((overloadable))
intel_sub_group_block_read_uc2(const __global uchar *p);

uchar4 __attribute__((overloadable))
intel_sub_group_block_read_uc4(const __global uchar *p);

uchar16 __attribute__((overloadable))
intel_sub_group_block_read_uc16(const __global uchar *p);

void __attribute__((overloadable))
intel_sub_group_block_write_uc16(__global uchar *p, uchar16 data);
#endif

#ifdef cl_intel_dot_accumulate
inline int __imad(char4 a, char4 b, int c) __attribute__((overloadable)) {
    int __builtin_IB_dp4a_ss(int c, int a, int b) __attribute__((const));
    return __builtin_IB_dp4a_ss(c, as_int(a), as_int(b));
}
inline int __imad(uchar4 a, uchar4 b, int c) __attribute__((overloadable)) {
    int __builtin_IB_dp4a_uu(int c, int a, int b) __attribute__((const));
    return __builtin_IB_dp4a_uu(c, as_int(a), as_int(b));
}
inline int __imad(char4 a, uchar4 b, int c) __attribute__((overloadable)) {
    int __builtin_IB_dp4a_su(int c, int a, int b) __attribute__((const));
    return __builtin_IB_dp4a_su(c, as_int(a), as_int(b));
}
inline int __imad(uchar4 a, char4 b, int c) __attribute__((overloadable)) {
    int __builtin_IB_dp4a_us(int c, int a, int b) __attribute__((const));
    return __builtin_IB_dp4a_us(c, as_int(a), as_int(b));
}
#define IMAD(_O, _I, _W) __imad(_O, _I, _W)

#else // cl_intel_dot_accumulate

#define IMAD(_O, _I, _W) mmad_4(_O, _I, _W)

#endif

#ifdef cl_intel_subgroup_matrix_multiply_accumulate
inline int8 dpas_8_8(uint8 a, int8 b, int8 acc) __attribute__((overloadable)) {
    return intel_sub_group_u8_i8_matrix_mad_k32(a, b, acc);
}
inline int4 dpas_8_4(uint4 a, int8 b, int4 acc) __attribute__((overloadable)) {
    return intel_sub_group_u8_i8_matrix_mad_k32(a, b, acc);
}

// TODO: put conversion builtins under appropriate extension
// right now there doesn't seem to any such named extension
// float -> bf conversion builtins (rte rounding mode)
short __builtin_IB_ftobf_1(float a) __attribute__((const));
short2 __builtin_IB_ftobf_2(float2 a) __attribute__((const));
short4 __builtin_IB_ftobf_4(float4 a) __attribute__((const));
short8 __builtin_IB_ftobf_8(float8 a) __attribute__((const));
short16 __builtin_IB_ftobf_16(float16 a) __attribute__((const));

// bf -> float conversion builtins (precise conversion)
float __builtin_IB_bftof_1(short a) __attribute__((const));
float2 __builtin_IB_bftof_2(short2 a) __attribute__((const));
float4 __builtin_IB_bftof_4(short4 a) __attribute__((const));
float8 __builtin_IB_bftof_8(short8 a) __attribute__((const));
float16 __builtin_IB_bftof_16(short16 a) __attribute__((const));

inline ushort convert_f32_to_bf16(float f) {
    return as_ushort(__builtin_IB_ftobf_1(f));
}

inline ushort2 convert_f32_to_bf16_vec2(float2 f) {
    return as_ushort2(__builtin_IB_ftobf_2(f));
}

inline ushort4 convert_f32_to_bf16_vec4(float4 f) {
    return as_ushort4(__builtin_IB_ftobf_4(f));
}

inline ushort8 convert_f32_to_bf16_vec8(float8 f) {
    return as_ushort8(__builtin_IB_ftobf_8(f));
}

inline ushort16 convert_f32_to_bf16_vec16(float16 f) {
    return as_ushort16(__builtin_IB_ftobf_16(f));
}

inline float convert_bf16_to_f32(ushort b) {
    return __builtin_IB_bftof_1(as_short(b));
}

inline float2 convert_bf16_to_f32_vec2(ushort2 b) {
    return __builtin_IB_bftof_2(as_short2(b));
}

inline float4 convert_bf16_to_f32_vec4(ushort4 b) {
    return __builtin_IB_bftof_4(as_short4(b));
}

inline float8 convert_bf16_to_f32_vec8(ushort8 b) {
    return __builtin_IB_bftof_8(as_short8(b));
}

inline float16 convert_bf16_to_f32_vec16(ushort16 b) {
    return __builtin_IB_bftof_16(as_short16(b));
}

#if DT_F16
inline float8 dpas_8_8(uint8 a, int8 b, float8 acc)
        __attribute__((overloadable)) {
    return intel_sub_group_f16_f16_matrix_mad_k16(as_int8(a), b, acc);
}
#elif DT_BF16 == 1
inline float8 dpas_8_8(uint8 a, int8 b, float8 acc)
        __attribute__((overloadable)) {
    return intel_sub_group_bf16_bf16_matrix_mad_k16(as_int8(a), b, acc);
}

inline float8 __dpasw(uint4 a, int8 b, float8 acc)
        __attribute__((overloadable)) {
    float8 __builtin_IB_sub_group_fdpasw_bf_bf_8_8(float8 acc, int4 a, int8 b)
            __attribute__((const));
    return __builtin_IB_sub_group_fdpasw_bf_bf_8_8(acc, as_int4(a), b);
}
#endif

#define MMAD8X4(_O, _I, _W) dpas_8_4(_O, _I, _W)
#define MMAD8X8(_O, _I, _W) dpas_8_8(_O, _I, _W)

#else // cl_intel_subgroup_matrix_multiply_accumulate

inline int mmad_4(uchar4 input, char4 weight, int acc)
        __attribute__((overloadable)) {
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    acc += (input[2] * weight[2]);
    acc += (input[3] * weight[3]);
    return acc;
}

inline int mmad_4(char4 input, char4 weight, int acc)
        __attribute__((overloadable)) {
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    acc += (input[2] * weight[2]);
    acc += (input[3] * weight[3]);
    return acc;
}

inline int mmad_4(char4 input, uchar4 weight, int acc)
        __attribute__((overloadable)) {
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    acc += (input[2] * weight[2]);
    acc += (input[3] * weight[3]);
    return acc;
}

inline int mmad_4(uchar4 input, uchar4 weight, int acc)
        __attribute__((overloadable)) {
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    acc += (input[2] * weight[2]);
    acc += (input[3] * weight[3]);
    return acc;
}

inline int mmad8(int8 A_scalars, int8 B_vectors, int acc)
        __attribute__((overloadable)) {
    acc = IMAD(as_char4(A_scalars[0]), as_char4(B_vectors[0]), acc);
    acc = IMAD(as_char4(A_scalars[1]), as_char4(B_vectors[1]), acc);
    acc = IMAD(as_char4(A_scalars[2]), as_char4(B_vectors[2]), acc);
    acc = IMAD(as_char4(A_scalars[3]), as_char4(B_vectors[3]), acc);
    acc = IMAD(as_char4(A_scalars[4]), as_char4(B_vectors[4]), acc);
    acc = IMAD(as_char4(A_scalars[5]), as_char4(B_vectors[5]), acc);
    acc = IMAD(as_char4(A_scalars[6]), as_char4(B_vectors[6]), acc);
    acc = IMAD(as_char4(A_scalars[7]), as_char4(B_vectors[7]), acc);
    return acc;
}

inline int mmad8(uint8 A_scalars, int8 B_vectors, int acc)
        __attribute__((overloadable)) {
    acc = IMAD(as_uchar4(A_scalars[0]), as_char4(B_vectors[0]), acc);
    acc = IMAD(as_uchar4(A_scalars[1]), as_char4(B_vectors[1]), acc);
    acc = IMAD(as_uchar4(A_scalars[2]), as_char4(B_vectors[2]), acc);
    acc = IMAD(as_uchar4(A_scalars[3]), as_char4(B_vectors[3]), acc);
    acc = IMAD(as_uchar4(A_scalars[4]), as_char4(B_vectors[4]), acc);
    acc = IMAD(as_uchar4(A_scalars[5]), as_char4(B_vectors[5]), acc);
    acc = IMAD(as_uchar4(A_scalars[6]), as_char4(B_vectors[6]), acc);
    acc = IMAD(as_uchar4(A_scalars[7]), as_char4(B_vectors[7]), acc);
    return acc;
}

inline int4 mmad8x4(uint4 A_vectors, int8 B_vectors, int4 acc)
        __attribute__((overloadable)) {
    int4 ret;
    for (uint i = 0; i < 4; i++) {
        uint8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = mmad8(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int4 mmad8x4(int4 A_vectors, int8 B_vectors, int4 acc)
        __attribute__((overloadable)) {
    int4 ret;
    for (uint i = 0; i < 4; i++) {
        int8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = mmad8(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int8 mmad8x8(uint8 A_vectors, int8 B_vectors, int8 acc)
        __attribute__((overloadable)) {
    int8 ret;
    for (uint i = 0; i < 8; i++) {
        uint8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = mmad8(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int8 mmad8x8(int8 A_vectors, int8 B_vectors, int8 acc)
        __attribute__((overloadable)) {
    int8 ret;
    for (uint i = 0; i < 8; i++) {
        int8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = mmad8(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

#if DT_F16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline float mmad_2(half2 input, half2 weight, float acc)
        __attribute__((overloadable)) {
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    return acc;
}

inline float mmad8(uint8 A_scalars, int8 B_vectors, float acc)
        __attribute__((overloadable)) {
    acc = mmad_2(as_half2(A_scalars[0]), as_half2(B_vectors[0]), acc);
    acc = mmad_2(as_half2(A_scalars[1]), as_half2(B_vectors[1]), acc);
    acc = mmad_2(as_half2(A_scalars[2]), as_half2(B_vectors[2]), acc);
    acc = mmad_2(as_half2(A_scalars[3]), as_half2(B_vectors[3]), acc);
    acc = mmad_2(as_half2(A_scalars[4]), as_half2(B_vectors[4]), acc);
    acc = mmad_2(as_half2(A_scalars[5]), as_half2(B_vectors[5]), acc);
    acc = mmad_2(as_half2(A_scalars[6]), as_half2(B_vectors[6]), acc);
    acc = mmad_2(as_half2(A_scalars[7]), as_half2(B_vectors[7]), acc);
    return acc;
}

inline float8 mmad8x8(uint8 A_vectors, int8 B_vectors, float8 acc)
        __attribute__((overloadable)) {
    float8 ret;
    for (uint i = 0; i < 8; i++) {
        uint8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = mmad8(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}
#endif

inline ushort convert_f32_to_bf16(float f) {
    uint i = as_uint(f);
    i += 0x00007FFF + ((i & 0x10000) >> 16);
    ushort2 r = as_ushort2(i);
    return r[1];
}

inline ushort2 convert_f32_to_bf16_vec2(float2 f) {
    ushort2 r;
    for (int i = 0; i < 2; i++) {
        r[i] = convert_f32_to_bf16(f[i]);
    }
    return r;
}

inline ushort4 convert_f32_to_bf16_vec4(float4 f) {
    ushort4 r;
    for (int i = 0; i < 4; i++) {
        r[i] = convert_f32_to_bf16(f[i]);
    }
    return r;
}

inline ushort8 convert_f32_to_bf16_vec8(float8 f) {
    ushort8 r;
    for (int i = 0; i < 8; i++) {
        r[i] = convert_f32_to_bf16(f[i]);
    }
    return r;
}

inline ushort16 convert_f32_to_bf16_vec16(float16 f) {
    ushort16 r;
    for (int i = 0; i < 16; i++) {
        r[i] = convert_f32_to_bf16(f[i]);
    }
    return r;
}

inline float convert_bf16_to_f32(ushort b) {
    ushort2 r = {0, b};
    float f = as_float(r);
    return f;
}

inline float2 convert_bf16_to_f32_vec2(ushort2 b) {
    float2 f;
    for (int i = 0; i < 2; i++) {
        f[i] = convert_bf16_to_f32(b[i]);
    }
    return f;
}

inline float4 convert_bf16_to_f32_vec4(ushort4 b) {
    float4 f;
    for (int i = 0; i < 4; i++) {
        f[i] = convert_bf16_to_f32(b[i]);
    }
    return f;
}

inline float8 convert_bf16_to_f32_vec8(ushort8 b) {
    float8 f;
    for (int i = 0; i < 8; i++) {
        f[i] = convert_bf16_to_f32(b[i]);
    }
    return f;
}

inline float16 convert_bf16_to_f32_vec16(ushort16 b) {
    float16 f;
    for (int i = 0; i < 16; i++) {
        f[i] = convert_bf16_to_f32(b[i]);
    }
    return f;
}

#if DT_BF16 == 1
inline float mmad_2(ushort2 input, ushort2 weight, float acc)
        __attribute__((overloadable)) {
    acc += (convert_bf16_to_f32(input[0]) * convert_bf16_to_f32(weight[0]));
    acc += (convert_bf16_to_f32(input[1]) * convert_bf16_to_f32(weight[1]));
    return acc;
}

inline float mmad8(uint8 A_scalars, int8 B_vectors, float acc)
        __attribute__((overloadable)) {
    acc = mmad_2(as_ushort2(A_scalars[0]), as_ushort2(B_vectors[0]), acc);
    acc = mmad_2(as_ushort2(A_scalars[1]), as_ushort2(B_vectors[1]), acc);
    acc = mmad_2(as_ushort2(A_scalars[2]), as_ushort2(B_vectors[2]), acc);
    acc = mmad_2(as_ushort2(A_scalars[3]), as_ushort2(B_vectors[3]), acc);
    acc = mmad_2(as_ushort2(A_scalars[4]), as_ushort2(B_vectors[4]), acc);
    acc = mmad_2(as_ushort2(A_scalars[5]), as_ushort2(B_vectors[5]), acc);
    acc = mmad_2(as_ushort2(A_scalars[6]), as_ushort2(B_vectors[6]), acc);
    acc = mmad_2(as_ushort2(A_scalars[7]), as_ushort2(B_vectors[7]), acc);
    return acc;
}

inline float8 mmad8x8(uint8 A_vectors, int8 B_vectors, float8 acc)
        __attribute__((overloadable)) {
    float8 ret;
    for (uint i = 0; i < 8; i++) {
        uint8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = mmad8(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

#endif

#define MMAD8X4(_O, _I, _W) mmad8x4(_O, _I, _W)
#define MMAD8X8(_O, _I, _W) mmad8x8(_O, _I, _W)

#endif

#ifdef cl_intel_subgroup_local_block_io
inline uint8 sub_group_block_read_uint8(const __local uint *p)
        __attribute__((overloadable)) {
    uint8 __builtin_IB_simd_block_read_8_local(const __local uint *p)
            __attribute__((const));
    return __builtin_IB_simd_block_read_8_local(p);
}
inline uint4 sub_group_block_read_uint4(const __local uint *p)
        __attribute__((overloadable)) {
    uint4 __builtin_IB_simd_block_read_4_local(const __local uint *p)
            __attribute__((const));
    return __builtin_IB_simd_block_read_4_local(p);
}

inline uint sub_group_block_read_uint(const __local uint *p)
        __attribute__((overloadable)) {
    uint __builtin_IB_simd_block_read_1_local(const __local uint *p)
            __attribute__((const));
    return __builtin_IB_simd_block_read_1_local(p);
}

void sub_group_block_write_ushort(__local ushort *p, uint v)
        __attribute__((overloadable)) {
    void __builtin_IB_simd_block_write_1_local_h(__local ushort * p, ushort v);
    __builtin_IB_simd_block_write_1_local_h(p, v);
}

void sub_group_block_write_uint(__local uint *p, uint v)
        __attribute__((overloadable)) {
    void __builtin_IB_simd_block_write_1_local(__local uint * p, uint v);
    __builtin_IB_simd_block_write_1_local(p, v);
}

void sub_group_block_write_uint2(__local uint *p, uint2 v)
        __attribute__((overloadable)) {
    void __builtin_IB_simd_block_write_2_local(__local uint * p, uint2 v);
    __builtin_IB_simd_block_write_2_local(p, v);
}

void sub_group_block_write_uint4(__local uint *p, uint4 v)
        __attribute__((overloadable)) {
    void __builtin_IB_simd_block_write_4_local(__local uint * p, uint4 v);
    __builtin_IB_simd_block_write_4_local(p, v);
}

void sub_group_block_write_uint8(__local uint *p, uint8 v)
        __attribute__((overloadable)) {
    void __builtin_IB_simd_block_write_8_local(__local uint * p, uint8 v);
    __builtin_IB_simd_block_write_8_local(p, v);
}

inline ushort8 sub_group_block_read_ushort8(const __local ushort *p)
        __attribute__((overloadable)) {
    ushort8 __builtin_IB_simd_block_read_8_local_h(const __local ushort *p);
    return __builtin_IB_simd_block_read_8_local_h(p);
}

void sub_group_block_write_ushort8(__local ushort *p, ushort8 v)
        __attribute__((overloadable)) {
    void __builtin_IB_simd_block_write_8_local_h(__local ushort * p, ushort8);
    __builtin_IB_simd_block_write_8_local_h(p, v);
}

#define READ_LOCAL_US_8(_P) sub_group_block_read_ushort8(_P)
#define READ_LOCAL_8(_P) sub_group_block_read_uint8(_P)
#define READ_LOCAL_4(_P) sub_group_block_read_uint4(_P)
#define READ_LOCAL_1(_P) sub_group_block_read_uint(_P)
#define WRITE_LOCAL_US_8(_P, _V) sub_group_block_write_ushort8(_P, _V)
#define WRITE_LOCAL_8(_P, _V) sub_group_block_write_uint8(_P, _V)
#define WRITE_LOCAL_4(_P, _V) sub_group_block_write_uint4(_P, _V)
#define WRITE_LOCAL_2(_P, _V) sub_group_block_write_uint2(_P, _V)
#define WRITE_LOCAL_1(_P, _V) sub_group_block_write_uint(_P, _V)
#define WRITE_LOCAL_SHORT_1(_P, _V) sub_group_block_write_ushort(_P, _V)

#else // cl_intel_subgroup_local_block_io

#define READ_LOCAL_US_8(_P) sub_group_block_read_ushort8(_P)
#define READ_LOCAL_8(_P) sub_group_block_read_uint8(_P)
#define READ_LOCAL_4(_P) sub_group_block_read_uint4(_P)
#define READ_LOCAL_1(_P) sub_group_block_read_uint(_P)
#define WRITE_LOCAL_US_8(_P, _V) sub_group_block_write_ushort8(_P, _V)
#define WRITE_LOCAL_8(_P, _V) sub_group_block_write_uint8(_P, _V)
#define WRITE_LOCAL_4(_P, _V) sub_group_block_write_uint4(_P, _V)
#define WRITE_LOCAL_2(_P, _V) sub_group_block_write_uint2(_P, _V)
#define WRITE_LOCAL_1(_P, _V) sub_group_block_write_uint(_P, _V)
#define WRITE_LOCAL_SHORT_1(_P, _V) sub_group_block_write_ushort(_P, _V)

ushort8 sub_group_block_read_ushort8(const __local ushort *p) {
    ushort8 ret;
    uint idx = get_sub_group_local_id();
    ret.s0 = p[idx];
    idx += get_max_sub_group_size();
    ret.s1 = p[idx];
    idx += get_max_sub_group_size();
    ret.s2 = p[idx];
    idx += get_max_sub_group_size();
    ret.s3 = p[idx];
    idx += get_max_sub_group_size();
    ret.s4 = p[idx];
    idx += get_max_sub_group_size();
    ret.s5 = p[idx];
    idx += get_max_sub_group_size();
    ret.s6 = p[idx];
    idx += get_max_sub_group_size();
    ret.s7 = p[idx];
    return ret;
}

void sub_group_block_write_ushort8(__local ushort *p, ushort8 v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v.s0;
    p += get_max_sub_group_size();
    p[idx] = v.s1;
    p += get_max_sub_group_size();
    p[idx] = v.s2;
    p += get_max_sub_group_size();
    p[idx] = v.s3;
    p += get_max_sub_group_size();
    p[idx] = v.s4;
    p += get_max_sub_group_size();
    p[idx] = v.s5;
    p += get_max_sub_group_size();
    p[idx] = v.s6;
    p += get_max_sub_group_size();
    p[idx] = v.s7;
}

uint sub_group_block_read_uint(const __local uint *p) {
    uint ret;
    uint idx = get_sub_group_local_id();
    ret = p[idx];
    return ret;
}

uint8 sub_group_block_read_uint8(const __local uint *p) {
    uint8 ret;
    uint idx = get_sub_group_local_id();
    ret.s0 = p[idx];
    idx += get_max_sub_group_size();
    ret.s1 = p[idx];
    idx += get_max_sub_group_size();
    ret.s2 = p[idx];
    idx += get_max_sub_group_size();
    ret.s3 = p[idx];
    idx += get_max_sub_group_size();
    ret.s4 = p[idx];
    idx += get_max_sub_group_size();
    ret.s5 = p[idx];
    idx += get_max_sub_group_size();
    ret.s6 = p[idx];
    idx += get_max_sub_group_size();
    ret.s7 = p[idx];
    return ret;
}

uint4 sub_group_block_read_uint4(const __local uint *p) {
    uint4 ret;
    uint idx = get_sub_group_local_id();
    ret.s0 = p[idx];
    idx += get_max_sub_group_size();
    ret.s1 = p[idx];
    idx += get_max_sub_group_size();
    ret.s2 = p[idx];
    idx += get_max_sub_group_size();
    ret.s3 = p[idx];
    return ret;
}

void sub_group_block_write_ushort(__local ushort *p, ushort v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v;
}

void sub_group_block_write_uint(__local uint *p, uint v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v;
}

void sub_group_block_write_uint2(__local uint *p, uint2 v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v.s0;
    p += get_max_sub_group_size();
    p[idx] = v.s1;
}

void sub_group_block_write_uint4(__local uint *p, uint4 v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v.s0;
    p += get_max_sub_group_size();
    p[idx] = v.s1;
    p += get_max_sub_group_size();
    p[idx] = v.s2;
    p += get_max_sub_group_size();
    p[idx] = v.s3;
}

void sub_group_block_write_uint8(__local uint *p, uint8 v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v.s0;
    p += get_max_sub_group_size();
    p[idx] = v.s1;
    p += get_max_sub_group_size();
    p[idx] = v.s2;
    p += get_max_sub_group_size();
    p[idx] = v.s3;
    p += get_max_sub_group_size();
    p[idx] = v.s4;
    p += get_max_sub_group_size();
    p[idx] = v.s5;
    p += get_max_sub_group_size();
    p[idx] = v.s6;
    p += get_max_sub_group_size();
    p[idx] = v.s7;
}
#endif

inline int idot4(int src, int wei, int acc) __attribute__((overloadable)) {
    return IMAD(as_char4(src), as_char4(wei), acc);
}

inline int idot4(uint src, int wei, int acc) __attribute__((overloadable)) {
    return IMAD(as_uchar4(src), as_char4(wei), acc);
}

#define DECLARE_MMAD(_suffix, _src_type, _acc_type, _int_type, _owb, _icb) \
    inline int mmad_part##_suffix(_int_type a, int8 b, int acc) \
            __attribute__((overloadable)) { \
        for (uint i = 0; i < _icb; ++i) \
            acc = idot4(a[i], b[i], acc); \
        return acc; \
    } \
    inline _acc_type mmad##_suffix(_src_type a, int8 b, _acc_type acc) \
            __attribute__((overloadable)) { \
        _acc_type ret; \
        for (uint i = 0; i < _owb; ++i) { \
            _int_type c; \
            for (int j = 0; j < _icb; ++j) \
                c[j] = sub_group_broadcast(a[i], j); \
            ret[i] = mmad_part##_suffix(c, b, acc[i]); \
        } \
        return ret; \
    }

void subgroup_block_write_uint(__local uint *p, uint v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v;
}

void subgroup_block_write_ushort(__local ushort *p, ushort v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v;
}

#if __OPENCL_C_VERSION__ >= 200
#ifdef cl_intel_global_float_atomics
inline void atomic_add_global(
        volatile global atomic_float *source, float operand) {
    atomic_fetch_add(source, operand);
}

#else // float atomics
inline void atomic_add_global(
        volatile __global atomic_float *source, float operand) {
    float old_val = atomic_load_explicit(
            source, memory_order_relaxed, memory_scope_device);
    bool success = false;
    do {
        float new_val = old_val + operand;
        success = atomic_compare_exchange_strong_explicit(source, &old_val,
                new_val, memory_order_acq_rel, memory_order_relaxed,
                memory_scope_device);
    } while (!success);
}
#endif
#endif

#endif

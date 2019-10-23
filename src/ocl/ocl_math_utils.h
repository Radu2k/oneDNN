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

#ifndef OCL_MATH_UTILS_H
#define OCL_MATH_UTILS_H

ushort convert_f32_to_bf16(float f) {
    uint i = as_uint(f);
    i += 0x00007FFF + ((i & 0x10000) >> 16);
    ushort2 r = as_ushort2(i);
    return r[1];
}

float convert_bf16_to_f32(ushort b) {
    ushort2 r = {0, b};
    float f = as_float(r);
    return f;
}

ushort8 convert_f32_to_bf16_vec8(float8 f) {
    ushort8 r;
    for (int i = 0; i < 8; i++) {
        r[i] = convert_f32_to_bf16(f[i]);
    }
    return r;
}

float8 convert_bf16_to_f32_vec8(ushort8 b) {
    float8 f;
    for (int i = 0; i < 8; i++) {
        f[i] = convert_bf16_to_f32(b[i]);
    }
    return f;
}

float2 convert_bf16_to_f32_vec2(ushort2 b) {
    float2 f;
    for (int i = 0; i < 2; i++) {
        f[i] = convert_bf16_to_f32(b[i]);
    }
    return f;
}

#ifdef cl_intel_subgroups_char
void __attribute__((overloadable))
intel_sub_group_block_write_uc16(__global uchar *p, uchar16 data);

uchar16 __attribute__((overloadable))
intel_sub_group_block_read_uc16(__global uchar *p);
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
inline int8 __dpas(uint8 a, int8 b, int8 acc) __attribute__((overloadable)) {
    int8 __builtin_IB_sub_group_idpas_u8_s8_8_8(int8 acc, uint8 a, int8 b)
            __attribute__((const));
    return __builtin_IB_sub_group_idpas_u8_s8_8_8(acc, a, b);
}

#if DT_F16
inline float8 __dpas(uint8 a, int8 b, float8 acc)
        __attribute__((overloadable)) {
    float8 __builtin_IB_sub_group_fdpas_hf_hf_8_8(float8 acc, int8 a, int8 b)
            __attribute__((const));
    return __builtin_IB_sub_group_fdpas_hf_hf_8_8(acc, as_int8(a), b);
}
#elif DT_BF16 == 1
inline float8 __dpas(uint8 a, int8 b, float8 acc)
        __attribute__((overloadable)) {
    float8 __builtin_IB_sub_group_fdpas_bf_bf_8_8(float8 acc, int8 a, int8 b)
            __attribute__((const));
    return __builtin_IB_sub_group_fdpas_bf_bf_8_8(acc, as_int8(a), b);
}
#endif

#define MMAD8X8(_O, _I, _W) __dpas(_O, _I, _W)

#else // cl_intel_subgroup_matrix_multiply_accumulate

#define MMAD8X8(_O, _I, _W) mmad8x8(_O, _I, _W)

#endif

#ifdef cl_intel_subgroup_local_block_io
inline uint8 subgroup_block_read_uint8(const __local uint *p)
        __attribute__((overloadable)) {
    uint8 __builtin_IB_simd_block_read_8_local(const __local uint *p)
            __attribute__((const));
    return __builtin_IB_simd_block_read_8_local(p);
}

inline uint subgroup_block_read_uint(const __local uint *p)
        __attribute__((overloadable)) {
    uint __builtin_IB_simd_block_read_1_local(const __local uint *p)
            __attribute__((const));
    return __builtin_IB_simd_block_read_1_local(p);
}

void subgroup_block_write_ushort(__local ushort *p, uint v)
        __attribute__((overloadable)) {
    void __builtin_IB_simd_block_write_1_local_h(__local ushort * p, ushort v);
    __builtin_IB_simd_block_write_1_local_h(p, v);
}

void subgroup_block_write_uint(__local uint *p, uint v)
        __attribute__((overloadable)) {
    void __builtin_IB_simd_block_write_1_local(__local uint * p, uint v);
    __builtin_IB_simd_block_write_1_local(p, v);
}

void subgroup_block_write_uint2(__local uint *p, uint2 v)
        __attribute__((overloadable)) {
    void __builtin_IB_simd_block_write_2_local(__local uint * p, uint2 v);
    __builtin_IB_simd_block_write_2_local(p, v);
}

void subgroup_block_write_uint4(__local uint *p, uint4 v)
        __attribute__((overloadable)) {
    void __builtin_IB_simd_block_write_4_local(__local uint * p, uint4 v);
    __builtin_IB_simd_block_write_4_local(p, v);
}

void subgroup_block_write_uint8(__local uint *p, uint8 v)
        __attribute__((overloadable)) {
    void __builtin_IB_simd_block_write_8_local(__local uint * p, uint8 v);
    __builtin_IB_simd_block_write_8_local(p, v);
}

inline ushort8 subgroup_block_read_ushort8(const __local ushort *p)
        __attribute__((overloadable)) {
    ushort8 __builtin_IB_simd_block_read_8_local_h(const __local ushort *p);
    return __builtin_IB_simd_block_read_8_local_h(p);
}

void subgroup_block_write_ushort8(__local ushort *p, ushort8 v)
        __attribute__((overloadable)) {
    void __builtin_IB_simd_block_write_8_local_h(__local ushort * p, ushort8);
    __builtin_IB_simd_block_write_8_local_h(p, v);
}

#define READ_LOCAL_US_8(_P) subgroup_block_read_ushort8(_P)
#define READ_LOCAL_8(_P) subgroup_block_read_uint8(_P)
#define READ_LOCAL_1(_P) subgroup_block_read_uint(_P)
#define WRITE_LOCAL_US_8(_P, _V) subgroup_block_write_ushort8(_P, _V)
#define WRITE_LOCAL_8(_P, _V) subgroup_block_write_uint8(_P, _V)
#define WRITE_LOCAL_4(_P, _V) subgroup_block_write_uint4(_P, _V)
#define WRITE_LOCAL_2(_P, _V) subgroup_block_write_uint2(_P, _V)
#define WRITE_LOCAL_1(_P, _V) subgroup_block_write_uint(_P, _V)
#define WRITE_LOCAL_SHORT_1(_P, _V) subgroup_block_write_ushort(_P, _V)

#else // cl_intel_subgroup_local_block_io

#define READ_LOCAL_US_8(_P) subgroup_block_read_ushort8(_P)
#define READ_LOCAL_8(_P) subgroup_block_read_uint8(_P)
#define READ_LOCAL_1(_P) subgroup_block_read_uint(_P)
#define WRITE_LOCAL_US_8(_P, _V) subgroup_block_write_ushort8(_P, _V)
#define WRITE_LOCAL_8(_P, _V) subgroup_block_write_uint8(_P, _V)
#define WRITE_LOCAL_4(_P, _V) subgroup_block_write_uint4(_P, _V)
#define WRITE_LOCAL_2(_P, _V) subgroup_block_write_uint2(_P, _V)
#define WRITE_LOCAL_1(_P, _V) subgroup_block_write_uint(_P, _V)
#define WRITE_LOCAL_SHORT_1(_P, _V) subgroup_block_write_ushort(_P, _V)

ushort8 subgroup_block_read_ushort8(const __local ushort *p) {
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

void subgroup_block_write_ushort8(__local ushort *p, ushort8 v) {
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

uint subgroup_block_read_uint(const __local uint *p) {
    uint ret;
    uint idx = get_sub_group_local_id();
    ret = p[idx];
    return ret;
}

uint8 subgroup_block_read_uint8(const __local uint *p) {
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

void subgroup_block_write_ushort(__local ushort *p, ushort v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v;
}

void subgroup_block_write_uint(__local uint *p, uint v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v;
}

void subgroup_block_write_uint2(__local uint *p, uint2 v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v.s0;
    p += get_max_sub_group_size();
    p[idx] = v.s1;
}

void subgroup_block_write_uint4(__local uint *p, uint4 v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v.s0;
    p += get_max_sub_group_size();
    p[idx] = v.s1;
    p += get_max_sub_group_size();
    p[idx] = v.s2;
    p += get_max_sub_group_size();
    p[idx] = v.s3;
}

void subgroup_block_write_uint8(__local uint *p, uint8 v) {
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
#endif

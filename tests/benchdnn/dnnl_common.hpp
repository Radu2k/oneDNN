/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#ifndef DNNL_COMMON_HPP
#define DNNL_COMMON_HPP

#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "dnnl.h"
#include "src/common/bfloat16.hpp"
#include "src/common/float16.hpp"
#include "src/common/nstl.hpp"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_debug.hpp"

#define for_ for

#define DNN_SAFE(f, s) \
    do { \
        dnnl_status_t status = f; \
        if (status != dnnl_success) { \
            if (s == CRIT || s == WARN) { \
                BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                        __PRETTY_FUNCTION__, __LINE__, #f, status2str(status), \
                        (int)status); \
                fflush(0); \
                if (s == CRIT) exit(2); \
            } \
            return FAIL; \
        } \
    } while (0)

#define DNN_SAFE_V(f) \
    do { \
        dnnl_status_t status = f; \
        if (status != dnnl_success) { \
            BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                    __PRETTY_FUNCTION__, __LINE__, STRINGIFY(f), \
                    status2str(status), (int)status); \
            fflush(0); \
            exit(2); \
        } \
    } while (0)

#define DNN_SAFE_CLEAN(f, s, clean) \
    do { \
        dnnl_status_t status = f; \
        if (status != dnnl_success) { \
            if (s == CRIT || s == WARN) { \
                BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                        __PRETTY_FUNCTION__, __LINE__, #f, status2str(status), \
                        (int)status); \
                fflush(0); \
                if (s == CRIT) exit(2); \
            } \
            clean(); \
            return FAIL; \
        } \
    } while (0)

/* aux */
using bfloat16_t = dnnl::impl::bfloat16_t;
using float16_t = dnnl::impl::float16_t;
template <dnnl_data_type_t>
struct prec_traits;
template <>
struct prec_traits<dnnl_bf16> {
    typedef bfloat16_t type;
};
template <>
struct prec_traits<dnnl_f16> {
    typedef float16_t type;
};
template <>
struct prec_traits<dnnl_f32> {
    typedef float type;
};
template <>
struct prec_traits<dnnl_s32> {
    typedef int32_t type;
};
template <>
struct prec_traits<dnnl_s8> {
    typedef int8_t type;
};
template <>
struct prec_traits<dnnl_u8> {
    typedef uint8_t type;
};

#define CASE_ALL(dt) \
    switch (dt) { \
        CASE(dnnl_bf16); \
        CASE(dnnl_f16); \
        CASE(dnnl_f32); \
        CASE(dnnl_s32); \
        CASE(dnnl_s8); \
        CASE(dnnl_u8); \
        default: assert(!"bad data_type"); \
    }

inline size_t sizeof_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: return sizeof(typename prec_traits<dt>::type);

    CASE_ALL(dt);

#undef CASE
    return 0;
}

/* std::numeric_limits::digits functionality */
inline int digits_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::digits;

    CASE_ALL(dt);

#undef CASE
    return 0;
}

inline float epsilon_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::epsilon();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

#undef CASE_ALL

template <dnnl_data_type_t dt>
inline float saturate(float val) {
    auto res = MAX2((float)dnnl::impl::nstl::numeric_limits<
                            typename prec_traits<dt>::type>::lowest(),
            MIN2((float)dnnl::impl::nstl::numeric_limits<
                         typename prec_traits<dt>::type>::max(),
                    val));
    return mxcsr_round(res);
}

inline float maybe_saturate(dnnl_data_type_t dt, float value) {
    if (dt == dnnl_s32 || dt == dnnl_s8 || dt == dnnl_u8) {
        switch (dt) {
#define CASE(dt) \
    case dt: { \
        return saturate<dt>(value); \
    }
            CASE(dnnl_s32);
            CASE(dnnl_s8);
            CASE(dnnl_u8);
#undef CASE
            default: assert(!"bad data_type");
        }
        return 0;
    }
    return value;
}

float round_to_nearest_representable(dnnl_data_type_t dt, float value);

/* simplification */
extern dnnl_engine_kind_t engine_tgt_kind;
extern dnnl_scratchpad_mode_t scratchpad_mode;

extern dnnl_engine_t engine_tgt;
extern dnnl_stream_t stream_tgt;
struct dnn_mem_t;
/* for fast-ref-gpu support */
extern dnnl_engine_t engine_cpu;
extern dnnl_stream_t stream_cpu;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
extern "C" int dnnl_memory_get_sim_id(dnnl_memory_t mem);
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
bool is_gpu_sim();
bool is_gpu_perf_sim();

void register_dnn_mem_object(dnn_mem_t *mem);
void unregister_dnn_mem_object(dnn_mem_t *mem);
#else
inline bool is_gpu_sim() {
    return false;
}
inline bool is_gpu_perf_sim() {
    return false;
}

inline void register_dnn_mem_object(dnn_mem_t *mem) {}
inline void unregister_dnn_mem_object(dnn_mem_t *mem) {}
#endif
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "dnnl_threadpool_iface.hpp"
// XXX: cannot include dnnl_thread.hpp because of conflicting macro
// definitions
namespace dnnl {
namespace impl {
namespace threadpool_utils {
threadpool_iface *get_active_threadpool();
}
} // namespace impl
} // namespace dnnl
#endif

inline int create_dnnl_stream(
        dnnl_stream_t *stream, dnnl_engine_t engine, unsigned flags) {
    dnnl_engine_kind_t engine_kind;
    DNN_SAFE(dnnl_engine_get_kind(engine, &engine_kind), CRIT);

    dnnl_stream_attr_t stream_attr;
    DNN_SAFE(dnnl_stream_attr_create(&stream_attr, engine_kind), CRIT);
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    if (engine_kind == dnnl_cpu) {
        SAFE_V(dnnl_stream_attr_set_threadpool(stream_attr,
                dnnl::impl::threadpool_utils::get_active_threadpool()));
    }
#endif

    DNN_SAFE(dnnl_stream_create_v2(stream, engine, flags, stream_attr), CRIT);
    dnnl_stream_attr_destroy(stream_attr);
    return OK;
}

inline int init() {
    if (!engine_tgt) {
        DNN_SAFE(dnnl_engine_create(&engine_tgt, engine_tgt_kind, 0), CRIT);
        SAFE(create_dnnl_stream(
                     &stream_tgt, engine_tgt, dnnl_stream_default_flags),
                CRIT);
    }
    if (!engine_cpu) {
        DNN_SAFE(dnnl_engine_create(&engine_cpu, dnnl_cpu, 0), CRIT);
        SAFE(create_dnnl_stream(
                     &stream_cpu, engine_cpu, dnnl_stream_default_flags),
                CRIT);
    }
    if (!engine_cpu) {
        DNN_SAFE(dnnl_engine_create(&engine_cpu, dnnl_cpu, 0), CRIT);
        DNN_SAFE(dnnl_stream_create(
                         &stream_cpu, engine_cpu, dnnl_stream_default_flags),
                CRIT);
    }

    return OK;
}

inline int finalize() {
    DNN_SAFE(dnnl_stream_destroy(stream_tgt), CRIT);
    DNN_SAFE(dnnl_engine_destroy(engine_tgt), CRIT);
    DNN_SAFE(dnnl_engine_destroy(engine_cpu), CRIT);
    DNN_SAFE(dnnl_stream_destroy(stream_cpu), CRIT);
    return OK;
}

inline const char *query_impl_info(const_dnnl_primitive_desc_t pd) {
    const char *str;
    dnnl_primitive_desc_query(pd, dnnl_query_impl_info_str, 0, &str);
    return str;
}

struct dnn_mem_t;
struct attr_bundle_t;

struct args_t {
    args_t &set(int arg, const dnn_mem_t &mem);
    void clear() { args_.clear(); }

    int size() const { return (int)args_.size(); }

    int arg(int index) const { return args_[index].first; }
    const dnn_mem_t &dnn_mem(int index) const { return *args_[index].second; }

private:
    std::vector<std::pair<int, const dnn_mem_t *>> args_;
};

// Engine used to run oneDNN primitives for testing.
inline const engine_t &get_test_engine() {
    static const engine_t instance(engine_tgt_kind);
    return instance;
}

// Engine used to run reference implementations (fast-ref-gpu option).
inline const engine_t &get_cpu_engine() {
    static const engine_t instance(dnnl_cpu);
    return instance;
}

template <typename func_t, typename prb_t>
int init_prim(dnnl_primitive_t *prim, const func_t &init_pd_func, prb_t *p,
        res_t *r, dir_t dir = FLAG_FWD,
        const_dnnl_primitive_desc_t hint = nullptr) {

    dnnl_primitive_desc_t pd {};
    dnnl_primitive_t return_prim {};

    auto cleanup_pd = [&]() { dnnl_primitive_desc_destroy(pd); };
    auto cleanup_prim = [&]() { dnnl_primitive_destroy(return_prim); };
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    // The idea is to create the requested primitive twice using
    // different engines.
    // Rationale:
    // 1. Make sure that the primitive cache is robust for the cases when:
    //   - CPU engine is re-created
    //   - GPU engine is re-created for the same device but different context
    // These 2 cases are commonly used or expected to be used in the frameworks.
    // 2. (for GPU only) Identify context dependent parts in primitive
    // implementations, e.g. if a primitive implementation contains
    // a memory_storage_t (for scales, zero points or buffers), which depends
    // on a particular engine then it should fail at execution time.

    // The first primitive creation using a temporary engine.
    engine_t engine(engine_tgt_kind);
    int status = init_pd_func(engine, p, pd, r, dir, hint);
    if (status != OK) return status;
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;
    DNN_SAFE_CLEAN(dnnl_primitive_create(&return_prim, pd), WARN, cleanup_pd);
    DNN_SAFE_CLEAN(dnnl_primitive_desc_destroy(pd), WARN, cleanup_prim);
    DNN_SAFE(dnnl_primitive_destroy(return_prim), WARN);

#endif
    // The second (if the cache is enabled) primitive creation using
    // the global test engine.
    status = init_pd_func(get_test_engine(), p, pd, r, dir, hint);
    if (status != OK) return status;
    // This primitive is expected to come from the cache.
    DNN_SAFE_CLEAN(dnnl_primitive_create(&return_prim, pd), WARN, cleanup_pd);
    DNN_SAFE_CLEAN(dnnl_primitive_desc_destroy(pd), WARN, cleanup_prim);
    (*prim) = return_prim;
    return OK;
}

// TODO: Remove this as soon as it's no longer used.
template <typename func_t, typename prb_t>
int init_prim(dnnl_primitive_t *prim, const func_t &init_pd_func,
        engine_t &engine, prb_t *p, res_t *r, dir_t dir = FLAG_FWD,
        const_dnnl_primitive_desc_t hint = nullptr) {
    // create 1st engine
    engine.reset(engine_tgt_kind);

    dnnl_primitive_desc_t _pd {};
    dnnl_primitive_t _prim {};

    auto cleanup_pd = [&]() { dnnl_primitive_desc_destroy(_pd); };
    auto cleanup_prim = [&]() { dnnl_primitive_destroy(_prim); };

    int status = init_pd_func(engine, p, _pd, r, dir, hint);
    if (status != OK) return status;
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    DNN_SAFE_CLEAN(dnnl_primitive_create(&_prim, _pd), WARN, cleanup_pd);

#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    // The idea is to create the requested primitive twice for different engines.
    // Rationale:
    // 1. Make sure that the primitive cache is robust for the cases when:
    //   - CPU engine is re-created
    //   - GPU engine is re-created for the same device but different context
    // These 2 cases are commonly used or expected to be used in the frameworks.
    // 2. (for GPU only) Identify context dependent parts in primitive implementations, e.g.
    // if a primitive implementation contains memory_storage_t (for scales,
    // zero points or buffers), which depends on a particular engine,
    // then it should crash or fail at execution time

    // TODO: add an internal API to be able to get information about cache hit
    // e.g. bool from_cache = dnnl_primitive_get_cache_hit_state(prim) == true;
    // If from_cache == true then this step can be skipped
    DNN_SAFE_CLEAN(dnnl_primitive_desc_destroy(_pd), WARN, cleanup_prim);
    DNN_SAFE(dnnl_primitive_destroy(_prim), WARN);

    // create 2nd engine
    engine.reset(engine_tgt_kind);
    status = init_pd_func(engine, p, _pd, r, dir, hint);
    if (status != OK) return status;

    // this primitive comes from the cache
    DNN_SAFE_CLEAN(dnnl_primitive_create(&_prim, _pd), WARN, cleanup_pd);
    // XXX: maybe check if the primitive didn't come from the cache and
    // return FAIL in that case?
#endif
    DNN_SAFE_CLEAN(dnnl_primitive_desc_destroy(_pd), WARN, cleanup_prim);
    (*prim) = _prim;
    return OK;
}

dnnl_status_t execute_and_wait(
        dnnl_primitive_t prim, dnnl_engine_t engine, const args_t &args);

int measure_perf(benchdnn_timer_t &t, dnnl_engine_t engine,
        dnnl_primitive_t prim, args_t &args);

void maybe_prepare_runtime_scales(dnn_mem_t &scales_m, const attr_t &attr,
        int64_t scale_cnt, const float *scales);
void maybe_prepare_runtime_scales(
        dnn_mem_t &scales_m, const attr_bundle_t &attr_bundle);

void maybe_prepare_runtime_zero_points(
        dnn_mem_t &zero_points_m, const attr_t &attr, int arg);

bool check_md_consistency_with_tag(
        const dnnl_memory_desc_t &md, const std::string &tag);

#endif

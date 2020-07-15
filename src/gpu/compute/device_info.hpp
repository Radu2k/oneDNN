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

#ifndef GPU_COMPUTE_DEVICE_INFO_HPP
#define GPU_COMPUTE_DEVICE_INFO_HPP

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "common/z_magic.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

enum class gpu_arch_t {
    unknown,
    gen9,
    gen12lp,
    gen12hp,
};

inline gpu_arch_t str2gpu_arch(const char *str) {
#define CASE(_case) \
    if (!strcmp(STRINGIFY(_case), str)) return gpu_arch_t::_case

    CASE(gen9);
    CASE(gen12lp);
    CASE(gen12hp);
    return gpu_arch_t::unknown;
#undef CASE
}

inline const char *gpu_arch2str(gpu_arch_t arch) {
#define CASE(_case) \
    case gpu_arch_t::_case: return STRINGIFY(_case)

    switch (arch) {
        CASE(gen9);
        CASE(gen12lp);
        CASE(gen12hp);
        CASE(unknown);
    }
    return "unknown";
#undef CASE
}


enum class device_ext_t : int64_t {
    intel_subgroups = 1 << 0,
    intel_subgroups_short = 1 << 1,
    khr_fp16 = 1 << 2,
    khr_int64_base_atomics = 1 << 3,
    intel_dot_accumulate = 1 << 4,
    intel_subgroup_local_block_io = 1 << 5,
    intel_subgroup_matrix_multiply_accumulate = 1 << 6,
    intel_subgroup_split_matrix_multiply_accumulate = 1 << 7,
    intel_global_float_atomics = 1 << 8,
    future_bf16_cvt = 1 << 9,
    last
};


static bool has(uint64_t extensions, device_ext_t ext) {
    return extensions & (uint64_t)ext;
}

static inline const char *ext2cl_str(device_ext_t ext) {

#define CASE(x) \
    case device_ext_t::x: return STRINGIFY(CONCAT2(cl_, x));
    switch (ext) {
        CASE(intel_subgroups);
        CASE(intel_subgroups_short);
        CASE(intel_dot_accumulate);
        CASE(intel_subgroup_local_block_io);
        CASE(intel_subgroup_matrix_multiply_accumulate);
        CASE(intel_subgroup_split_matrix_multiply_accumulate);
        CASE(intel_global_float_atomics);
        CASE(khr_fp16);
        CASE(khr_int64_base_atomics);
        CASE(future_bf16_cvt);
        default: return nullptr;
    }
#undef CASE
}

static device_ext_t get_extensions(gpu_arch_t gpu_arch) {
    uint64_t extensions = 0;
    switch (gpu_arch) {
        case gpu_arch_t::gen12hp:
            extensions |= (uint64_t)
                    device_ext_t::intel_subgroup_matrix_multiply_accumulate;
            extensions |= (uint64_t)device_ext_t::
                    intel_subgroup_split_matrix_multiply_accumulate;
            extensions |= (uint64_t)device_ext_t::intel_global_float_atomics;
            extensions |= (uint64_t)device_ext_t::future_bf16_cvt;
        case gpu_arch_t::gen12lp:
            extensions |= (uint64_t)device_ext_t::intel_dot_accumulate;
            extensions |= (uint64_t)device_ext_t::intel_subgroup_local_block_io;
        case gpu_arch_t::gen9:
            extensions |= (uint64_t)device_ext_t::khr_fp16;
            extensions |= (uint64_t)device_ext_t::intel_subgroups;
            extensions |= (uint64_t)device_ext_t::intel_subgroups_short;
            break;
        case gpu_arch_t::unknown: break;
    }
    return (device_ext_t)extensions;
}

struct runtime_version_t {
    int major;
    int minor;
    int build;

    runtime_version_t(int major = 0, int minor = 0, int build = 0)
        : major {major}, minor {minor}, build {build} {}

    bool operator==(const runtime_version_t &other) const {
        return (major == other.major) && (minor == other.minor)
                && (build == other.build);
    }

    bool operator!=(const runtime_version_t &other) const {
        return !(*this == other);
    }

    bool operator<(const runtime_version_t &other) const {
        if (major < other.major) return true;
        if (major > other.major) return false;
        if (minor < other.minor) return true;
        if (minor > other.minor) return false;
        return (build < other.build);
    }

    bool operator>(const runtime_version_t &other) const {
        return (other < *this);
    }

    bool operator<=(const runtime_version_t &other) const {
        return !(*this > other);
    }

    bool operator>=(const runtime_version_t &other) const {
        return !(*this < other);
    }

    status_t set_from_string(const char *s) {
        int i_major = 0, i = 0;

        for (; s[i] != '.'; i++)
            if (!s[i]) return status::invalid_arguments;

        auto i_minor = ++i;

        for (; s[i] != '.'; i++)
            if (!s[i]) return status::invalid_arguments;

        auto i_build = ++i;

        major = atoi(&s[i_major]);
        minor = atoi(&s[i_minor]);
        build = atoi(&s[i_build]);

        return status::success;
    }

    std::string str() const {
        return utils::format("%d.%d.%d", major, minor, build);
    }
};

struct device_info_t {
public:
    virtual ~device_info_t() = default;

    virtual status_t init_runtime_version(runtime_version_t &ret) const = 0;
    virtual status_t init_name(std::string &ret) const = 0;
    virtual status_t init_eu_count(int &ret) const = 0;
    virtual status_t init_extension_string(std::string &ret) const = 0;

    virtual status_t init() {
        CHECK(init_runtime_version(runtime_version_));
        CHECK(init_name(name_));
        CHECK(init_eu_count(eu_count_));
        CHECK(init_extension_string(extension_string_));

        CHECK(init_arch());
        CHECK(init_extensions());
        CHECK(init_attributes());

        return status::success;
    }

    status_t init_arch() {
        if (name().find("Gen9") != std::string::npos)
            real_gpu_arch_ = gpu_arch_t::gen9;
        else if (name().find("Gen12LP") != std::string::npos)
            real_gpu_arch_ = gpu_arch_t::gen12lp;
        else if (name().find("Gen12HP") != std::string::npos)
            real_gpu_arch_ = gpu_arch_t::gen12hp;
        else
            real_gpu_arch_ = gpu_arch_t::unknown;

        gpu_arch_t env_gpu_arch = gpu_arch_t::unknown;
        char gpu_arch_str[32];
        if (getenv("DNNL_GPU_ARCH", gpu_arch_str, sizeof(gpu_arch_str)) > 0) {
            env_gpu_arch = str2gpu_arch(gpu_arch_str);
        }

        // GPU architecture is not overriden from environment, set and return.
        if (env_gpu_arch == gpu_arch_t::unknown) {
            gpu_arch_ = real_gpu_arch_;
            return status::success;
        }


        // Environment GPU architecture is different from the detected one, use
        // emulation.

        // Do not allow emulating older architectures
        if ((int)env_gpu_arch < (int)real_gpu_arch_) {
            assert(!"not expected");
            return status::runtime_error;
        }
        gpu_arch_ = env_gpu_arch;

        return status::success;
    }

    const runtime_version_t &runtime_version() const {
        return runtime_version_;
    }
    const std::string &name() const { return name_; }
    int eu_count() const { return eu_count_; }

    int hw_threads() const { return hw_threads_; }
    size_t llc_cache_size() const { return llc_cache_size_; }

    gpu_arch_t gpu_arch() const { return gpu_arch_; }
    gpu_arch_t real_gpu_arch() const { return real_gpu_arch_; }

    bool has(device_ext_t ext) const { return compute::has(extensions_, ext); }

protected:
    status_t init_extensions() {
        for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
                i_ext <<= 1) {
            const char *s_ext = ext2cl_str((device_ext_t)i_ext);
            if (s_ext && extension_string_.find(s_ext) != std::string::npos) {
                extensions_ |= i_ext;
                real_extensions_ |= i_ext;
            }
        }

        // This is to handle extensions that are not yet properly supported by
        // OpenCL/Level Zero/DPC++ runtimes.
        extensions_ |= (uint64_t)get_extensions(gpu_arch());
        real_extensions_ |= (uint64_t)get_extensions(real_gpu_arch());

        return status::success;
    }

    status_t init_attributes() {
        // Assume 7 threads by default
        int32_t threads_per_eu = 7;

        switch (gpu_arch()) {
            case gpu_arch_t::gen9: threads_per_eu = 7; break;
            case gpu_arch_t::gen12lp: threads_per_eu = 7; break;
            case gpu_arch_t::gen12hp:
                // Default is 8 threads, 128 GRF registers per thread. But we
                // set 4 threads configuration (with 256 registers) for better
                // performance.
                threads_per_eu = 4;
                break;
            default: break;
        }

        hw_threads_ = eu_count_ * threads_per_eu;

        // TODO: Fix for discrete GPUs. The code below is written for
        // integrated GPUs assuming that last-level cache for GPU is shared
        // with CPU.
        size_t cache_size = cpu::platform::get_per_core_cache_size(3)
                * cpu::platform::get_num_cores();
        llc_cache_size_ = (size_t)cache_size;
        return status::success;
    }

    runtime_version_t runtime_version_;
    std::string name_;
    int32_t eu_count_ = 0;
    std::string extension_string_;

    int32_t hw_threads_ = 0;
    size_t llc_cache_size_ = 0;

    // Effective extensions.
    uint64_t extensions_ = 0;
    // Effective GPU architecture.
    gpu_arch_t gpu_arch_ = gpu_arch_t::unknown;

    // Real extensions.
    uint64_t real_extensions_ = 0;
    // Real GPU architecture.
    gpu_arch_t real_gpu_arch_ = gpu_arch_t::unknown;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

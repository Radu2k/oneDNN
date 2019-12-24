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

#ifndef OCL_GPU_DEVICE_INFO_HPP
#define OCL_GPU_DEVICE_INFO_HPP

#include <string>
#include <vector>
#include <CL/cl.h>

#include "common/z_magic.hpp"
#include "compute/device_info.hpp"
#include "ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

enum class gpu_arch_t {
    unknown,
    gen9,
    gen12lp,
};

inline gpu_arch_t str2gpu_arch(const char *str) {
#define CASE(_case) \
    if (!strcmp(STRINGIFY(_case), str)) return gpu_arch_t::_case

    CASE(gen9);
    CASE(gen12lp);
    return gpu_arch_t::unknown;
#undef CASE
}

inline const char *gpu_arch2str(gpu_arch_t arch) {
#define CASE(_case) \
    case gpu_arch_t::_case: return STRINGIFY(_case)

    switch (arch) {
        CASE(gen9);
        CASE(gen12lp);
        CASE(unknown);
    }
    return "unknown";
#undef CASE
}

static compute::device_ext_t get_extensions(gpu_arch_t gpu_arch) {
    uint64_t extensions = 0;
    switch (gpu_arch) {
        case gpu_arch_t::gen12lp:
            extensions |= (uint64_t)compute::device_ext_t::intel_dot_accumulate;
            extensions |= (uint64_t)
                    compute::device_ext_t::intel_subgroup_local_block_io;
        case gpu_arch_t::gen9:
            extensions |= (uint64_t)compute::device_ext_t::khr_fp16;
            extensions |= (uint64_t)compute::device_ext_t::intel_subgroups;
            extensions
                    |= (uint64_t)compute::device_ext_t::intel_subgroups_short;
            break;
        case gpu_arch_t::unknown: break;
    }
    return (compute::device_ext_t)extensions;
}

class ocl_gpu_device_info_t : public compute::device_info_t {
public:
    ocl_gpu_device_info_t(cl_device_id device) : device_(device) {}

    virtual status_t init() override {
        CHECK(init_arch());
        CHECK(init_extensions());
        CHECK(init_attributes());

        // Device name
        size_t size_name {0};
        cl_int err = clGetDeviceInfo(
                device_, CL_DEVICE_NAME, 0, nullptr, &size_name);
        OCL_CHECK(err);

        std::string dev_name;
        dev_name.resize(size_name);
        err = clGetDeviceInfo(
                device_, CL_DEVICE_NAME, size_name, &dev_name[0], &size_name);
        OCL_CHECK(err);
        set_name(dev_name);

        // OpenCL runtime version
        size_t size_driver_version {0};
        err = clGetDeviceInfo(
                device_, CL_DRIVER_VERSION, 0, nullptr, &size_driver_version);
        OCL_CHECK(err);
        std::string driver_version;
        driver_version.resize(size_driver_version);
        err = clGetDeviceInfo(device_, CL_DRIVER_VERSION, size_driver_version,
                &driver_version[0], nullptr);
        OCL_CHECK(err);

        driver_version[size_driver_version - 1] = '\0';
        compute::runtime_version_t runtime_version;
        if (runtime_version.set_from_string(&driver_version[0])
                != status::success) {
            runtime_version.major = 0;
            runtime_version.minor = 0;
            runtime_version.build = 0;
        }
        set_runtime_version(runtime_version);

        return status::success;
    }

    virtual bool has(compute::device_ext_t ext) const override {
        return has(extensions_, ext);
    }

    std::string get_cl_ext_options() const {
        using namespace compute;

        std::string opts;
        for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
                i_ext <<= 1) {
            auto ext = (device_ext_t)i_ext;
            // Use real GPU extensions
            if (!has(real_extensions_, ext)) continue;

            // These extensions are not handled properly by the OpenCL runtime.
            // Pass macros for them manually.
            if (utils::one_of(ext, device_ext_t::intel_dot_accumulate,
                        device_ext_t::intel_subgroup_local_block_io))
                opts += std::string("-D") + ext2cl_str(ext) + " ";
        }
        if (!opts.empty()) { opts[opts.size() - 1] = '\0'; }
        return opts;
    }

    gpu_arch_t gpu_arch() const { return gpu_arch_; }

    virtual int eu_count() const override { return eu_count_; }
    virtual int hw_threads() const override { return hw_threads_; }

private:
    status_t init_arch() {
        if (name().find("Gen9") != std::string::npos)
            real_gpu_arch_ = gpu_arch_t::gen9;
        else if (name().find("Gen12LP") != std::string::npos)
            real_gpu_arch_ = gpu_arch_t::gen12lp;
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

    status_t init_extensions() {
        // Handle extensions provided by the OpenCL runtime
        size_t size_ext {0};
        cl_int err = clGetDeviceInfo(
                device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &size_ext);
        OCL_CHECK(err);

        std::string dev_ext;
        dev_ext.resize(size_ext);

        err = clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, size_ext,
                &dev_ext[0], &size_ext);
        OCL_CHECK(err);

        for (uint64_t i_ext = 1; i_ext < (uint64_t)compute::device_ext_t::last;
                i_ext <<= 1) {
            const char *s_ext = ext2cl_str((compute::device_ext_t)i_ext);
            if (s_ext != nullptr && dev_ext.find(s_ext) != std::string::npos) {
                extensions_ |= i_ext;
                real_extensions_ |= i_ext;
            }
        }

        // Handle future extensions, not yet supported by the OpenCL API
        extensions_ |= (uint64_t)get_extensions(gpu_arch_);
        real_extensions_ |= (uint64_t)get_extensions(real_gpu_arch_);

        return status::success;
    }

    status_t init_attributes() {
        cl_uint eu_count;
        cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(cl_uint), &eu_count, nullptr);
        eu_count_ = (err == CL_SUCCESS) ? eu_count : 0;

        // Assume 7 threads by default
        int32_t threads_per_eu = 7;

        switch (gpu_arch_) {
            case gpu_arch_t::gen9: threads_per_eu = 7; break;
            case gpu_arch_t::gen12lp: threads_per_eu = 7; break;
            default: break;
        }

        hw_threads_ = eu_count_ * threads_per_eu;
        return status::success;
    }

    bool has(uint64_t extensions, compute::device_ext_t ext) const {
        return extensions & (uint64_t)ext;
    }

    cl_device_id device_ = nullptr;

    int32_t eu_count_ = 0;
    int32_t hw_threads_ = 0;

    // extensions_ and gpu_arch_ describe effective extensions and GPU architecutre.
    // real_extensions_ and real_gpu_arch_ describe real extensions and GPU architecutre.
    uint64_t extensions_ = 0;
    uint64_t real_extensions_ = 0;

    gpu_arch_t gpu_arch_ = gpu_arch_t::unknown;
    gpu_arch_t real_gpu_arch_ = gpu_arch_t::unknown;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif // OCL_GPU_DEVICE_INFO_HPP

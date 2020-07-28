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

#ifndef GPU_OCL_OCL_GPU_DEVICE_INFO_HPP
#define GPU_OCL_OCL_GPU_DEVICE_INFO_HPP

#include <string>
#include <vector>
#include <CL/cl.h>

#include "common/z_magic.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

class ocl_gpu_device_info_t : public compute::device_info_t {
public:
    ocl_gpu_device_info_t(cl_device_id device) : device_(device) {}

    status_t init_name(std::string &ret) const override {
        size_t size_name {0};
        cl_int err = clGetDeviceInfo(
                device_, CL_DEVICE_NAME, 0, nullptr, &size_name);
        OCL_CHECK(err);

        ret.resize(size_name);
        err = clGetDeviceInfo(
                device_, CL_DEVICE_NAME, size_name, &ret[0], &size_name);
        OCL_CHECK(err);
        return status::success;
    }

    status_t init_runtime_version(
            compute::runtime_version_t &ret) const override {
        size_t size_driver_version {0};
        cl_int err = clGetDeviceInfo(
                device_, CL_DRIVER_VERSION, 0, nullptr, &size_driver_version);
        OCL_CHECK(err);
        std::string driver_version;
        driver_version.resize(size_driver_version);
        err = clGetDeviceInfo(device_, CL_DRIVER_VERSION, size_driver_version,
                &driver_version[0], nullptr);
        OCL_CHECK(err);

        driver_version[size_driver_version - 1] = '\0';
        if (ret.set_from_string(&driver_version[0]) != status::success) {
            ret.major = 0;
            ret.minor = 0;
            ret.build = 0;
        }
        return status::success;
    }

    status_t init_eu_count(int &ret) const override {
        cl_uint max_units;
        cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(cl_uint), &max_units, nullptr);
        OCL_CHECK(err);
        ret = (int)max_units;
        return status::success;
    }

    status_t init_extension_string(std::string &ret) const override {
        size_t size_ext {0};
        cl_int err = clGetDeviceInfo(
                device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &size_ext);
        OCL_CHECK(err);

        ret.resize(size_ext);
        err = clGetDeviceInfo(
                device_, CL_DEVICE_EXTENSIONS, size_ext, &ret[0], &size_ext);
        OCL_CHECK(err);
        return status::success;
    }

    std::string get_cl_ext_options() const {
        using namespace compute;

        std::string opts;
        for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
                i_ext <<= 1) {
            auto ext = (device_ext_t)i_ext;
            // Use real GPU extensions
            if (!compute::has(real_extensions_, ext)) continue;

            // These extensions are not handled properly by the OpenCL runtime.
            // Pass macros for them manually.
            if (utils::one_of(ext, device_ext_t::intel_dot_accumulate,
                        device_ext_t::intel_subgroup_local_block_io,
                        device_ext_t::intel_subgroup_matrix_multiply_accumulate,
                        device_ext_t::
                                intel_subgroup_split_matrix_multiply_accumulate,
                        device_ext_t::intel_global_float_atomics,
                        device_ext_t::future_bf16_cvt))
                opts += std::string("-D") + ext2cl_str(ext) + " ";
        }
        if (!opts.empty()) { opts[opts.size() - 1] = '\0'; }
        return opts;
    }

private:
    cl_device_id device_ = nullptr;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_OCL_OCL_GPU_DEVICE_INFO_HPP

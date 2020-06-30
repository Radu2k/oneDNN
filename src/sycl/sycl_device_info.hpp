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

#ifndef SYCL_DEVICE_INFO_HPP
#define SYCL_DEVICE_INFO_HPP

#include <vector>
#include <CL/sycl.hpp>

#include "gpu/compute/device_info.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_device_info_t : public gpu::compute::device_info_t {
public:
    sycl_device_info_t(const cl::sycl::device &device) : device_(device) {}

    status_t init_runtime_version(
            gpu::compute::runtime_version_t &ret) const override {
        auto driver_version
                = device_.get_info<cl::sycl::info::device::driver_version>();
        if (ret.set_from_string(driver_version.c_str()) != status::success) {
            ret.major = 0;
            ret.minor = 0;
            ret.build = 0;
        }
        return status::success;
    }

    status_t init_name(std::string &ret) const override {
        ret = device_.get_info<cl::sycl::info::device::name>();
        return status::success;
    }

    status_t init_eu_count(int &ret) const override {
        ret = device_.get_info<cl::sycl::info::device::max_compute_units>();
        return status::success;
    }

    status_t init_extension_string(std::string &ret) const override {
        using namespace gpu::compute;

        ret = "";
        for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
                i_ext <<= 1) {
            const char *s_ext = ext2cl_str((device_ext_t)i_ext);
            if (s_ext && device_.has_extension(s_ext)) {
                ret += std::string(s_ext) + " ";
            }
        }
        return status::success;
    }

private:
    cl::sycl::device device_;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // SYCL_DEVICE_INFO_HPP

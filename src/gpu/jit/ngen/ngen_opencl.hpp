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

#ifndef NGEN_OPENCL_HPP
#define NGEN_OPENCL_HPP

#include <CL/cl.h>

#include <sstream>

#ifndef NGEN_NEO_INTERFACE
#define NGEN_NEO_INTERFACE
#endif
#include "ngen.hpp"
#include "ngen_interface.hpp"

#include "npack/neo_packager.hpp"

namespace ngen {


// Exceptions.
class unsupported_opencl_runtime : public std::runtime_error {
public:
    unsupported_opencl_runtime() : std::runtime_error("Unsupported OpenCL runtime.") {}
};
class opencl_error : public std::runtime_error {
public:
    opencl_error(cl_int status_ = 0) : std::runtime_error("An OpenCL error occurred."), status(status_) {}
protected:
    cl_int status;
};

// OpenCL program generator class.
template <HW hw>
class OpenCLCodeGenerator : public BinaryCodeGenerator<hw>
{
public:
    inline cl_kernel getKernel(cl_context context, cl_device_id device, const std::string &options = "-cl-std=CL2.0", const std::vector<uint8_t> &patches = std::vector<uint8_t>{});
    static inline HW detectHW(cl_context context, cl_device_id device);

protected:
    NEOInterfaceHandler interface;
};

namespace detail {

static inline void handleCL(cl_int result)
{
    if (result != CL_SUCCESS)
        throw opencl_error{result};
}

static inline std::vector<uint8_t> getOpenCLCProgramBinary(cl_context context, cl_device_id device, const char *src, const char *options)
{
    cl_int status;

    auto program = clCreateProgramWithSource(context, 1, &src, nullptr, &status);

    detail::handleCL(status);
    if (program == nullptr)
        throw opencl_error();

    detail::handleCL(clBuildProgram(program, 1, &device, options, nullptr, nullptr));

    size_t binarySize;
    detail::handleCL(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(binarySize), &binarySize, nullptr));

    std::vector<uint8_t> binary(binarySize);
    const auto *binaryPtr = binary.data();
    detail::handleCL(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binaryPtr), &binaryPtr, nullptr));

    detail::handleCL(clReleaseProgram(program));

    return binary;
}

}; /* namespace detail */

template <HW hw>
cl_kernel OpenCLCodeGenerator<hw>::getKernel(cl_context context, cl_device_id device, const std::string &options, const std::vector<uint8_t> &patches)
{
    cl_int status;
    std::ostringstream dummyCL;

    interface.generateDummyCL(dummyCL);
    auto dummyCLString = dummyCL.str();

    auto binary = detail::getOpenCLCProgramBinary(context, device, dummyCLString.c_str(), options.c_str());

    npack::replaceKernel(binary, this->getCode(), patches);

    const auto *binaryPtr = binary.data();
    size_t binarySize = binary.size();
    auto program = clCreateProgramWithBinary(context, 1, &device, &binarySize, &binaryPtr, nullptr, &status);
    detail::handleCL(status);
    if (program == nullptr)
        throw opencl_error();

    detail::handleCL(clBuildProgram(program, 1, &device, options.c_str(), nullptr, nullptr));

    auto kernel = clCreateKernel(program, interface.getExternalName().c_str(), &status);
    detail::handleCL(status);
    if (kernel == nullptr)
        throw opencl_error();

    detail::handleCL(clReleaseProgram(program));

    return kernel;
}

template <HW hw>
HW OpenCLCodeGenerator<hw>::detectHW(cl_context context, cl_device_id device)
{
    const char *dummyCL = "kernel void _(){}";
    const char *dummyOptions = "";

    auto binary = detail::getOpenCLCProgramBinary(context, device, dummyCL, dummyOptions);
    return npack::getBinaryArch(binary);
}

} /* namespace ngen */

#endif

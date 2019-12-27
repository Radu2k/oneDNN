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

#include "ocl/ngen_binary_format.hpp"
#include "ocl/ocl_gpu_engine.hpp"
#include "ocl/ocl_stream.hpp"

#include "ocl/ngen/ngen_opencl.hpp"

#define MAGIC0 0xBEEFCAFEu
#define MAGIC1 0x3141592653589793ull
#define MAGIC2 0xBEAD
#define MAGIC3 0xFACE
#define MAGIC4 0x0123456789ABCDEFull
#define MAGIC5 0xFEDCBA9876543210ull
#define MAGICPTR 0xABADFEEDu
#define MAGICSIZEX 8
#define MAGICSIZEY 3
#define MAGICSIZEZ 2

namespace dnnl {
namespace impl {
namespace ocl {

using namespace ngen;

class ngen_gen9_binary_format_kernel_t
    : public ngen::OpenCLCodeGenerator<ngen::HW::Gen9> {
public:
    ngen_gen9_binary_format_kernel_t()
        : ngen::OpenCLCodeGenerator<ngen::HW::Gen9>() {
        interface.newArgument("src0", DataType::ud); // r5.4:ud
        interface.newArgument("src1", DataType::uq); // r5.3:uq
        interface.newArgument("src2", DataType::uw); // r6.0:uw
        interface.newArgument("src3", DataType::uw); // r6.2:uw
        interface.newArgument("src4", DataType::uq); // r6.1:uq
        interface.newArgument("src5", DataType::uq); // r6.2:uq
        interface.newArgument("src_ptr", ExternalArgumentType::GlobalPtr);
        interface.newArgument("ok", ExternalArgumentType::GlobalPtr);

        interface.requireSIMD(8);
        interface.requireLocalID(3); // r1-r3
        interface.requireLocalSize(); // r7.0-2:ud
        interface.finalize();

        Label doWrite;

        auto data = r30;
        auto data2 = r31;
        auto qtemp = r90.uq(0);
        auto ok = data.ud(0);
        auto header = r64;

        mov(1, ok, uint16_t(0));

        cmp(1 | eq | f0[0], null.ud(), interface.getArgument("src0"),
                uint32_t(MAGIC0));
        jmpi(1 | ~f0[0], doWrite);
        mov(1, qtemp, uint64_t(MAGIC1));
        cmp(1 | eq | f0[0], null.uq(), interface.getArgument("src1"), qtemp);
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.uw(), interface.getArgument("src2"),
                uint16_t(MAGIC2));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.uw(), interface.getArgument("src3"),
                uint16_t(MAGIC3));
        jmpi(1 | ~f0[0], doWrite);
        mov(1, qtemp, uint64_t(MAGIC4));
        cmp(1 | eq | f0[0], null.uq(), interface.getArgument("src4"), qtemp);
        jmpi(1 | ~f0[0], doWrite);
        mov(1, qtemp, uint64_t(MAGIC5));
        cmp(1 | eq | f0[0], null.uq(), interface.getArgument("src5"), qtemp);
        jmpi(1 | ~f0[0], doWrite);

        mov<uint64_t>(1, header[0], interface.getArgument("src_ptr"));
        load(1, data2, scattered_dword(), A64, header);
        cmp(1 | eq | f0[0], null.ud(), data2.ud(0), uint32_t(MAGICPTR));
        jmpi(1 | ~f0[0], doWrite);

        cmp(1 | eq | f0[0], null.ud(), interface.getLocalSize(0),
                uint32_t(MAGICSIZEX));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.ud(), interface.getLocalSize(1),
                uint32_t(MAGICSIZEY));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.ud(), interface.getLocalSize(2),
                uint32_t(MAGICSIZEZ));
        jmpi(1 | ~f0[0], doWrite);

        mov(1, ok, uint16_t(1));

        mark(doWrite);

        mov<uint32_t>(1, header, uint16_t(0));
        store(1, scattered_dword(), Surface(interface.getArgumentSurface("ok")),
                header, data);

        mov<uint32_t>(8, r127, r0);
        threadend(r127);
    }
};

status_t gpu_supports_binary_format(bool *ok, engine_t *engine) {
    *ok = false;

    // TODO: binary format kernel supports gen9, extend it for gen12
#if 0
    if (!gpu_is_gen9) return status::runtime_error;
#endif

#if defined(_MSC_VER) && (_MSC_VER < 1910)
    // MSVC 2015 and earlier are not supported due to compiler bug
    return status::success;
#endif

    auto *gpu_engine = utils::downcast<ocl_gpu_engine_t *>(engine);
    if (!gpu_engine) return status::runtime_error;

    auto binary_format_kernel
            = utils::make_unique<ngen_gen9_binary_format_kernel_t>();
    auto kernel = binary_format_kernel->getKernel(
            gpu_engine->context(), gpu_engine->device());
    if (!kernel) return status::runtime_error;
    auto compute_kernel = compute::kernel_t(new ocl_gpu_kernel_t(kernel));

    status_t status = status::success;

    // Binary kernel check.
    uint32_t magic0 = MAGIC0;
    uint64_t magic1 = MAGIC1;
    uint16_t magic2 = MAGIC2;
    uint16_t magic3 = MAGIC3;
    uint64_t magic4 = MAGIC4;
    uint64_t magic5 = MAGIC5;
    uint32_t magic_ptr = MAGICPTR;

    size_t gws[3] = {1, 1, 1};
    size_t lws[3] = {MAGICSIZEX, MAGICSIZEY, MAGICSIZEZ};

    memory_storage_t *storage = nullptr;
    std::unique_ptr<memory_storage_t> magic_buf, result_buf;

    status = engine->create_memory_storage(&storage, sizeof(int32_t));
    if (status != status::success) return status::runtime_error;
    magic_buf.reset(storage);

    status = engine->create_memory_storage(&storage, sizeof(int32_t));
    if (status != status::success) return status::runtime_error;
    result_buf.reset(storage);

    auto stream = utils::downcast<ocl_stream_t *>(gpu_engine->service_stream());
    auto queue = stream->queue();

    OCL_CHECK(clEnqueueWriteBuffer(queue, (cl_mem)magic_buf->data_handle(),
            CL_TRUE, 0, sizeof(magic_ptr), &magic_ptr, 0, nullptr, nullptr));

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, magic0);
    arg_list.set(1, magic1);
    arg_list.set(2, magic2);
    arg_list.set(3, magic3);
    arg_list.set(4, magic4);
    arg_list.set(5, magic5);
    arg_list.set(6, *magic_buf.get());
    arg_list.set(7, *result_buf.get());

    auto nd_range = compute::nd_range_t(gws, lws);
    status = stream->parallel_for(nd_range, compute_kernel, arg_list);
    if (status != status::success) return status::runtime_error;

    status = stream->wait();
    if (status != status::success) return status::runtime_error;

    int result = 0;
    OCL_CHECK(clEnqueueReadBuffer(queue, (cl_mem)result_buf->data_handle(),
            CL_TRUE, 0, sizeof(int32_t), &result, 0, nullptr, nullptr));
    *ok = (result != 0);

    return status::success;
}

} // namespace ocl
} // namespace impl
} // namespace dnnl

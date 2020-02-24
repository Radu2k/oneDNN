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

#ifndef CPU_STREAM_HPP
#define CPU_STREAM_HPP

#include "common/c_types_map.hpp"
#include "common/stream.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct cpu_stream_t : public stream_t {
    cpu_stream_t(engine_t *engine, unsigned flags) : stream_t(engine, flags) {}
    virtual ~cpu_stream_t() = default;

    virtual dnnl::impl::status_t wait() override {
        // CPU execution is synchronous so return immediately
        return dnnl::impl::status::success;
    }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <cm/cm.h>

#if DT_F16
#define DATA_T half
#elif DT_F32
#define DATA_T float
#elif DT_U8
#define DATA_T uchar
#elif DT_S8
#define DATA_T char
#else
#error
#endif

template <typename T, int N, int Block = 128>
_GENX_ inline void read_x(SurfaceIndex buf, int off, vector_ref<T, N> v) {
    if constexpr ((N & Block) != 0) read(buf, off, v.select<Block, 1>());
    constexpr int rest = N - (N & Block);
    if constexpr (rest > 0 && Block >= 8) {
        read_x<T, rest, Block / 2>(buf, off + (N & Block) * sizeof(T),
                v.select<rest, 1>(N & Block));
    }
}

template <typename T, int N, int Block = 128>
_GENX_ inline void write_x(SurfaceIndex buf, int off, vector_ref<T, N> v) {
    if constexpr ((N & Block) != 0) write(buf, off, v.select<Block, 1>());
    constexpr int rest = N - (N & Block);
    if constexpr (rest > 0 && Block >= 8) {
        write_x<T, rest, Block / 2>(buf, off + (N & Block) * sizeof(T),
                v.select<rest, 1>(N & Block));
    }
}

extern "C" _GENX_MAIN_ void simple_sum(SurfaceIndex input [[type("buffer_t")]],
        SurfaceIndex output [[type("buffer_t")]], float scale, int a) {
    int ithr = cm_linear_global_id();

    vector<DATA_T, BLOCK_SIZE> in;
    vector<DATA_T, BLOCK_SIZE> ou;

    read_x(input, BLOCK_SIZE * ithr * sizeof(DATA_T), in);
    if (a == 0) {
        ou = (scale * in);
    } else {
        read_x(output, BLOCK_SIZE * ithr * sizeof(DATA_T), ou);
        ou += (scale * in);
    }
    write_x(output, BLOCK_SIZE * ithr * sizeof(DATA_T), ou);
}

// CudaArray: header-only library for interfacing with CUDA array-type objects
// Author: True Price <jtprice at cs.unc.edu>
//
// BSD License
// Copyright (C) 2017  The University of North Carolina at Chapel Hill
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the original author nor the names of contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
// THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
// CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
// NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef CUDA_RANDOM_STATE_ARRAY3D_H_
#define CUDA_RANDOM_STATE_ARRAY3D_H_

#include "cudaArray3D.h"

#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

namespace cua {

class CudaRandomStateArray3D;  // forward declaration

namespace kernel {

__global__ void CudaRandomStateArray3DInit(CudaRandomStateArray3D array,
                                           size_t seed);

}  // namespace kernel

//------------------------------------------------------------------------------

/**
 * @class CudaRandomStateArray3D
 */
class CudaRandomStateArray3D : public CudaArray3D<curandState_t> {
 public:
  CudaRandomStateArray3D(size_t width, size_t height, size_t depth);

  CudaRandomStateArray3D(size_t width, size_t height, size_t depth,
                         size_t seed);

 private:
  static inline size_t GetSeed() {
    auto span = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(span).count();
  }
};

//------------------------------------------------------------------------------
//
// class method implementations
//
//------------------------------------------------------------------------------

CudaRandomStateArray3D::CudaRandomStateArray3D(size_t width, size_t height,
                                               size_t depth)
    : CudaRandomStateArray3D(width, height, depth, GetSeed()) {}

CudaRandomStateArray3D::CudaRandomStateArray3D(size_t width, size_t height,
                                               size_t depth, size_t seed)
    : CudaArray3D<curandState_t>::CudaArray3D(width, height, depth) {
  kernel::CudaRandomStateArray3DInit<<<grid_dim_, block_dim_>>>(*this, seed);
}

//------------------------------------------------------------------------------
//
// kernel functions
//
//------------------------------------------------------------------------------

namespace kernel {

//
// initialize an array of random generators
//
__global__ void CudaRandomStateArray3DInit(CudaRandomStateArray3D array,
                                                   size_t seed) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  // the curand documentation says it should be faster (and probably ok) to use
  // different seeds with sequence number 0
  if (x < array.Width() && y < array.Height() && z < array.Depth()) {
    curandState_t rand_state;
    curand_init(seed + (z * array.Height() + y) * array.Width() + x, 0, 0,
                &rand_state);
    array.set(x, y, z, rand_state);
  }
}

}  // namespace kernel

}  // namespace cua

#endif  // CUDA_RANDOM_STATE_ARRAY3D_H_

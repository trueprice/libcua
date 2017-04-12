// CudaArray: header-only library for interfacing with CUDA array-type objects
// Copyright (C) 2017  True Price <jtprice at cs.unc.edu>
//
// MIT License
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef CUDARANDOMSTATEARRAY2D_KERNELS_H_
#define CUDARANDOMSTATEARRAY2D_KERNELS_H_

#include "cudaArray2D.h"

#include <curand.h>
#include <curand_kernel.h>

namespace cua {

//
// class-specific kernel functions for the CudaRandonStateArray2D class
//

//
// CudaArray2D_init_rand: initialize a matrix of random generators
//
template <typename cudaRandomStateArrayClass>
__global__ void
CudaRandomStateArray2D_init_kernel(cudaRandomStateArrayClass mat,
                                   const size_t seed) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // the curand documentation says it should be faster (and probably ok) to use
  // different seeds with sequence number 0
  if (x < mat.get_width() && y < mat.get_height()) {
    curandState_t rand_state;
    curand_init(seed + y * mat.get_width() + x, 0, 0, &rand_state);
    mat.set(x, y, rand_state);
  }
}

} // namespace cua

#endif // CUDARANDOMSTATEARRAY2D_KERNELS_H_

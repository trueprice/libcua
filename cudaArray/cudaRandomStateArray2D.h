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

// TODO: In the future, have the class be independent of CudaArray2D; instead,
// a call to this CudaRandomStateArray2D(x, y) would create a new curandState_t
// instance if one didn't already exist (we could still initialize width x
// height random states). This would keep us from having to ensure the object
// had enough entries
// TODO: another option would be to have this class fill matrices, rather than
// the other way around. That might work better.
// TODO: also want to add an Options class that would allow for different random
// generators, etc.

#ifndef CUDARANDOMSTATEARRAY2D_H_
#define CUDARANDOMSTATEARRAY2D_H_

#include "cudaArray2D.h"
#include "cudaRandomStateArray2D_kernels.h"

#include <chrono>
#include <curand.h>
#include <curand_kernel.h>

namespace cua {

//
// CudaRandomStateArray2D
//
class CudaRandomStateArray2D : public CudaArray2D<curandState_t> {
 public:
  CudaRandomStateArray2D(const size_t width, const size_t height);
};

//
// class method implementations
//

CudaRandomStateArray2D::CudaRandomStateArray2D(const size_t width,
                                               const size_t height)
    : CudaArray2D<curandState_t>::CudaArray2D(width, height) {
  const auto duration =
      std::chrono::high_resolution_clock::now().time_since_epoch();
  const size_t seed =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  CudaRandomStateArray2D_init_kernel << <grid_dim_, block_dim_>>> (*this, seed);
}

} // namespace cua

#endif  // CUDARANDOMSTATEARRAY2D_H_

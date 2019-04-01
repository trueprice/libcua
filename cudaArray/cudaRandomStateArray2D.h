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

// TODO: In the future, have the class be independent of CudaArray2D; instead,
// a call to this CudaRandomStateArray2D(x, y) would create a new curandState_t
// instance if one didn't already exist (we could still initialize width x
// height random states). This would keep us from having to ensure the object
// had enough entries
// TODO: another option would be to have this class fill matrices, rather than
// the other way around. That might work better.
// TODO: also want to add an Options class that would allow for different random
// generators, etc.

#ifndef CUDA_RANDOM_STATE_ARRAY2D_H_
#define CUDA_RANDOM_STATE_ARRAY2D_H_

#include "cudaArray2D.h"
#include "cudaRandomStateArray2D_kernels.h"

#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

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

  CudaRandomStateArray2D_init_kernel<<<grid_dim_, block_dim_>>>(*this, seed);
}

}  // namespace cua

#endif  // CUDA_RANDOM_STATE_ARRAY2D_H_

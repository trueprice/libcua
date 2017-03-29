// Author: True Price <jtprice at cs.unc.edu>

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

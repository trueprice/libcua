// Author: True Price <jtprice at cs.unc.edu>

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

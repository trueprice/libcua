#ifndef CUDA_ARRAY_FWD_H_
#define CUDA_ARRAY_FWD_H_

/**
 * @namespace cua
 * @brief cudaArray namespace
 */
namespace cua {

template <typename T>
class CudaArray2D;

template <typename T>
class CudaArray3D;

class CudaRandomStateArray2D;

template <typename T>
class CudaSurface2D;

template <typename T>
class CudaTexture2D;

}  // namespace cua

#endif  // CUDA_ARRAY_FWD_H_

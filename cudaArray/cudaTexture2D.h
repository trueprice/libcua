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

#ifndef CUDA_TEXTURE2D_H_
#define CUDA_TEXTURE2D_H_

#include "cudaArray2DBase.h"
#include "cudaSharedArrayObject.h"

#include <memory>  // for shared_ptr

namespace cua {

/**
 * @class CudaTexture2D
 * @brief Texture-memory 2D array.
 *
 * This class implements an interface for 2D texture-memory arrays on the GPU.
 * These arrays are read-only, and copy for CudaTexture2D objects is a shallow
 * operation.
 *
 * The arrays can be directly passed into device-level code, i.e., you can write
 * kernels that have CudaTexture2D objects in their parameter lists:
 *
 *     __global__ void device_kernel(const CudaTexture2D<float> in,
 *                                   CudaSurface2D<float> out) {
 *       const int x = blockIdx.x * blockDim.x + threadIdx.x;
 *       const int y = blockIdx.y * blockDim.y + threadIdx.y;
 *       out.set(x, y, in.get(x, y));
 *     }
 *
 * TODO (True): mipmaps, etc. would be useful, as well as texture-coordinate
 *   lookups, etc.
 */
template <typename T>
class CudaTexture2D : public CudaArray2DBase<CudaTexture2D<T>> {
 public:
  friend class CudaArray2DBase<CudaTexture2D<T>>;

  /// datatype of the array
  typedef T Scalar;

  typedef CudaArray2DBase<CudaTexture2D<T>> Base;

 protected:
  // for convenience, reference base class members directly (they are otherwise
  // not in the current scope because CudaArray2DBase is templated)
  using Base::width_;
  using Base::height_;
  using Base::block_dim_;
  using Base::grid_dim_;
  using Base::stream_;

 public:
  //----------------------------------------------------------------------------
  // constructors and destructor

  /**
   * Constructor.
   * @param width number of columns in the array, assuming a row-major array
   * @param height number of rows in the array, assuming a row-major array
   * @param filter_mode use cudaFilterModeLinear to allow for interpolation
   * @param address_mode specifies how to read values outside of 2D extent of
   *   the texture
   * @param read_mode can also optionally specify this as
   *   cudaReadModeNormalizedFloat
   * @param block_dim default block size for CUDA kernel calls involving this
   *   object, i.e., the values for blockDim.x/y/z; note that the default grid
   *   dimension is computed automatically based on the array size
   * @param stream CUDA stream for this array object
   *   extents of the array
   */
  CudaTexture2D(
      const size_t width, const size_t height,
      const cudaTextureFilterMode filter_mode = cudaFilterModePoint,
      const cudaTextureAddressMode address_mode = cudaAddressModeBorder,
      const cudaTextureReadMode read_mode = cudaReadModeElementType,
      const dim3 block_dim = CudaTexture2D<T>::BLOCK_DIM,
      const cudaStream_t stream = 0);  // default stream

  /**
   * Host and device-level copy constructor. This is a shallow-copy operation,
   * meaning that the underlying CUDA memory is the same for both arrays.
   */
  __host__ __device__ CudaTexture2D(const CudaTexture2D<T> &other);

  ~CudaTexture2D() {}

  //----------------------------------------------------------------------------
  // array operations

  /**
   * Create an empty array of the same size as the current array.
   */
  CudaTexture2D<T> EmptyCopy() const;

  /**
   * Create a new empty array with transposed dimensions (flipped height/width).
   */
  CudaTexture2D<T> EmptyFlippedCopy() const;

  /**
   * Shallow re-assignment of the given array to share the contents of another.
   * @param other a separate array whose contents will now also be referenced by
   *   the current array
   * @return *this
   */
  CudaTexture2D<T> &operator=(const CudaTexture2D<T> &other);

  /**
   * Copy the contents of a CPU-bound memory array to the current array. This
   * function assumes that the CPU array has the correct size!
   * @param host_array the CPU-bound array
   * @return *this
   */
  CudaTexture2D<T> &operator=(const T *host_array);

  /**
   * Copy the contents of the current array to a CPU-bound memory array. This
   * function assumes that the CPU array has the correct size!
   * @param host_array the CPU-bound array
   */
  void CopyTo(T *host_array) const;

  //----------------------------------------------------------------------------
  // getters

  /**
   * Device-level function for getting a texture pixel value. Note, if you use
   * cudaReadModeNormalizedFloat as the texture read mode, you'll need to
   * specify the appropriate return type (i.e., float) in the template argument.
   * @param x first coordinate, i.e., the column index in a row-major array
   * @param y second coordinate, i.e., the row index in a row-major array
   * @return the value at array(x, y)
   */
  template <typename ReturnType = T>
  __device__ inline ReturnType get(const int x, const int y) const {
    return tex2D<ReturnType>(shared_texture_.get_cuda_api_object(), x + 0.5f,
                             y + 0.5f);
  }

  /**
   * Device-level function for getting an interpolated texture pixel value,
   * which is enabled by specifying filter_mode as cudaFilterModeLinear in the
   * constructor. Note, if you use cudaReadModeNormalizedFloat as the texture
   * read mode, you'll need to specify the appropriate return type (i.e., float)
   * in the template argument.
   * @param x first coordinate, i.e., the column index in a row-major array
   * @param y second coordinate, i.e., the row index in a row-major array
   * @return the interpolated value at array(x, y)
   */
  // to properly use cudaReadModeNormalizedFloat, you'll need to specify the
  // appropriate return type (e.g., float) in the template argument
  template <typename ReturnType = T>
  __device__ inline ReturnType interp(const float x, const float y) const {
    return tex2D<ReturnType>(shared_texture_.get_cuda_api_object(), x, y);
  }

 private:
  CudaSharedTextureObject<T> shared_texture_;
};

//------------------------------------------------------------------------------
// template typedef for CRTP model, a la Eigen

template <typename T>
struct CudaArrayTraits<CudaTexture2D<T>> {
  typedef T Scalar;
};

//------------------------------------------------------------------------------
//
// public method implementations
//
//------------------------------------------------------------------------------

template <typename T>
CudaTexture2D<T>::CudaTexture2D<T>(const size_t width, const size_t height,
                                   const cudaTextureFilterMode filter_mode,
                                   const cudaTextureAddressMode address_mode,
                                   const cudaTextureReadMode read_mode,
                                   const dim3 block_dim,
                                   const cudaStream_t stream)
    : Base(width, height, block_dim, stream),
      shared_texture_(width, height, filter_mode, address_mode, read_mode) {}

//------------------------------------------------------------------------------

// host- and device-level copy constructor
template <typename T>
__host__ __device__
CudaTexture2D<T>::CudaTexture2D<T>(const CudaTexture2D<T> &other)
    : Base(other), shared_texture_(other.shared_texture_) {}

//------------------------------------------------------------------------------

template <typename T>
CudaTexture2D<T> CudaTexture2D<T>::EmptyCopy() const {
  return CudaTexture2D<T>(width_, height_, block_dim_, stream_);
}

//------------------------------------------------------------------------------

template <typename T>
CudaTexture2D<T> CudaTexture2D<T>::EmptyFlippedCopy() const {
  return CudaTexture2D<T>(height_, width_, dim3(block_dim_.y, block_dim_.x),
                          stream_);
}

//------------------------------------------------------------------------------

template <typename T>
CudaTexture2D<T> &CudaTexture2D<T>::operator=(const CudaTexture2D<T> &other) {
  if (this == &other) {
    return *this;
  }

  Base::operator=(other);

  shared_texture_ = other.shared_texture_;

  return *this;
}

//------------------------------------------------------------------------------

template <typename T>
CudaTexture2D<T> &CudaTexture2D<T>::operator=(const T *host_array) {
  cudaMemcpyToArray(shared_texture_.get_dev_array(), 0, 0, host_array,
                    sizeof(T) * width_ * height_, cudaMemcpyHostToDevice);

  return *this;
}

//------------------------------------------------------------------------------

template <typename T>
void CudaTexture2D<T>::CopyTo(T *host_array) const {
  cudaMemcpyFromArray(host_array, shared_texture_.get_dev_array(), 0, 0,
                      sizeof(T) * width_ * height_, cudaMemcpyDeviceToHost);
}

}  // namespace cua

#endif  // CUDA_TEXTURE2D_H_

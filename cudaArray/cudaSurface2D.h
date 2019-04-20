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

#ifndef CUDA_SURFACE2D_H_
#define CUDA_SURFACE2D_H_

#include "cudaArray2DBase.h"
#include "cudaSharedArrayObject.h"

#include <memory>  // for shared_ptr

namespace cua {

/**
 * @class CudaSurface2D
 * @brief Surface-memory 2D array.
 *
 * This class implements an interface for 2D surface-memory arrays on the GPU.
 * These arrays are read-able and write-able, and compared to linear-memory
 * arrays, they have better cache coherence properties for memory accesses in a
 * 2D neighborhood. Copy/assignment for CudaSurface2D objects is a shallow
 * operation; use Copy(other) to perform a deep copy.
 *
 * The arrays can be directly passed into device-level code, i.e., you can write
 * kernels that have CudaSurface2D objects in their parameter lists:
 *
 *     __global__ void device_kernel(CudaSurface2D<float> arr) {
 *       const int x = blockIdx.x * blockDim.x + threadIdx.x;
 *       const int y = blockIdx.y * blockDim.y + threadIdx.y;
 *       arr.set(x, y, 0.0);
 *     }
 */
template <typename T>
class CudaSurface2D : public CudaArray2DBase<CudaSurface2D<T>> {
 public:
  friend class CudaArray2DBase<CudaSurface2D<T>>;

  /// datatype of the array
  typedef T Scalar;

  typedef CudaArray2DBase<CudaSurface2D<T>> Base;

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
   * @param block_dim default block size for CUDA kernel calls involving this
   *   object, i.e., the values for blockDim.x/y/z; note that the default grid
   *   dimension is computed automatically based on the array size
   * @param stream CUDA stream for this array object
   * @param boundary_mode boundary mode to use for reads that go outside the 2D
   *   extents of the array
   */
  CudaSurface2D(
      const size_t width, const size_t height,
      const dim3 block_dim = CudaSurface2D<T>::BLOCK_DIM,
      const cudaStream_t stream = 0,  // default stream
      const cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeZero);

  /**
   * Host and device-level copy constructor. This is a shallow-copy operation,
   * meaning that the underlying CUDA memory is the same for both arrays.
   */
  __host__ __device__ CudaSurface2D(const CudaSurface2D<T> &other);

  ~CudaSurface2D() {}

  /**
   * Create a view onto the underlying CUDA memory. This function assumes that
   * the cropped view region is valid!
   * @param x x-coordinate for the top left of the view
   * @param y y-coordinate for the top left of the view
   * @param width width of the view
   * @param height height of the view
   * @return new CudaSurface2D view whose underlying device pointer and size is
   * aligned with the view
   */
  CudaSurface2D<T> inline View(const size_t x, const size_t y,
                               const size_t width, const size_t height) const {
    return CudaSurface2D<T>(x, y, width, height, *this);
  }

  //----------------------------------------------------------------------------
  // array operations

  /**
   * Create an empty array of the same size as the current array.
   */
  CudaSurface2D<T> EmptyCopy() const;

  /**
   * Create a new empty array with transposed dimensions (flipped height/width).
   */
  CudaSurface2D<T> EmptyFlippedCopy() const;

  /**
   * Shallow re-assignment of the given array to share the contents of another.
   * @param other a separate array whose contents will now also be referenced by
   *   the current array
   * @return *this
   */
  CudaSurface2D<T> &operator=(const CudaSurface2D<T> &other);

  /**
   * Copy the contents of a CPU-bound memory array to the current array. This
   * function assumes that the CPU array has the correct size!
   * @param host_array the CPU-bound array
   * @return *this
   */
  CudaSurface2D<T> &operator=(const T *host_array);

  /**
   * Copy the contents of the current array to a CPU-bound memory array. This
   * function assumes that the CPU array has the correct size!
   * @param host_array the CPU-bound array
   */
  void CopyTo(T *host_array) const;

  //----------------------------------------------------------------------------
  // getters/setters

  /**
   * Device-level function for setting an element in an array
   * @param x first coordinate, i.e., the column index in a row-major array
   * @param y second coordinate, i.e., the row index in a row-major array
   * @param v the new value to assign to array(x, y)
   */
  __device__ inline void set(const int x, const int y, const T v) {
    surf2Dwrite(v, shared_surface_.get_cuda_api_object(),
                sizeof(T) * (x + x_offset_), y + y_offset_, boundary_mode_);
  }

  /**
   * Device-level function for getting an element in an array
   * @param x first coordinate, i.e., the column index in a row-major array
   * @param y second coordinate, i.e., the row index in a row-major array
   * @return the value at array(x, y)
   */
  __device__ inline T get(const int x, const int y) const {
    return surf2Dread<T>(shared_surface_.get_cuda_api_object(),
                         sizeof(T) * (x + x_offset_), y + y_offset_,
                         boundary_mode_);
  }

  /**
   * @return the boundary mode for the underlying CUDA Surface object
   */
  __host__ __device__ inline cudaSurfaceBoundaryMode BoundaryMode() const {
    return boundary_mode_;
  }

  /**
   * set the boundary mode for the underlying CUDA Surface object
   */
  __host__ __device__ inline void SetBoundaryMode(
      const cudaSurfaceBoundaryMode boundary_mode) {
    boundary_mode_ = boundary_mode;
  }

  //----------------------------------------------------------------------------
  // private class methods and fields

 private:
  /**
   * Internal constructor used for creating views.
   * @param x x-coordinate for the top left of the view
   * @param y y-coordinate for the top left of the view
   * @param width width of the view
   * @param height height of the view
   */
  CudaSurface2D(const size_t x, const size_t y, const size_t width,
                const size_t height, const CudaSurface2D<T> &other);

  CudaSharedSurfaceObject<T> shared_surface_;

  cudaSurfaceBoundaryMode boundary_mode_;

  size_t x_offset_, y_offset_;  // = 0 if not using a view
};

//------------------------------------------------------------------------------
// template typedef for CRTP model, a la Eigen

template <typename T>
struct CudaArrayTraits<CudaSurface2D<T>> {
  typedef T Scalar;
  typedef bool Mutable;
};

//------------------------------------------------------------------------------
//
// public method implementations
//
//------------------------------------------------------------------------------

template <typename T>
CudaSurface2D<T>::CudaSurface2D<T>(const size_t width, const size_t height,
                                   const dim3 block_dim,
                                   const cudaStream_t stream,
                                   const cudaSurfaceBoundaryMode boundary_mode)
    : Base(width, height, block_dim, stream),
      boundary_mode_(boundary_mode),
      shared_surface_(width, height),
      x_offset_(0),
      y_offset_(0) {}

//------------------------------------------------------------------------------

// host- and device-level copy constructor
template <typename T>
__host__ __device__
CudaSurface2D<T>::CudaSurface2D<T>(const CudaSurface2D<T> &other)
    : Base(other),
      boundary_mode_(other.boundary_mode_),
      shared_surface_(other.shared_surface_),
      x_offset_(other.x_offset_),
      y_offset_(other.y_offset_) {}

//------------------------------------------------------------------------------

// host-level private constructor for creating views
template <typename T>
CudaSurface2D<T>::CudaSurface2D<T>(const size_t x, const size_t y,
                                   const size_t width, const size_t height,
                                   const CudaSurface2D<T> &other)
    : Base(width, height, other.block_dim_, other.stream_),
      boundary_mode_(other.boundary_mode_),
      shared_surface_(other.shared_surface_),
      x_offset_(x + other.x_offset_),
      y_offset_(y + other.y_offset_) {}

//------------------------------------------------------------------------------

template <typename T>
inline CudaSurface2D<T> CudaSurface2D<T>::EmptyCopy() const {
  return CudaSurface2D<T>(width_, height_, block_dim_, stream_, boundary_mode_);
}

//------------------------------------------------------------------------------

// create a transposed version (flipped height/width) of the given matrix
template <typename T>
inline CudaSurface2D<T> CudaSurface2D<T>::EmptyFlippedCopy() const {
  return CudaSurface2D<T>(height_, width_, dim3(block_dim_.y, block_dim_.x),
                          stream_, boundary_mode_);
}

//------------------------------------------------------------------------------

template <typename T>
inline CudaSurface2D<T> &CudaSurface2D<T>::operator=(const T *host_array) {
  const size_t width_in_bytes = width_ * sizeof(T);
  cudaMemcpy2DToArray(shared_surface_.get_dev_array(), x_offset_ * sizeof(T),
                      y_offset_, host_array, width_in_bytes, width_in_bytes,
                      height_, cudaMemcpyHostToDevice);

  return *this;
}

//------------------------------------------------------------------------------

template <typename T>
inline CudaSurface2D<T> &CudaSurface2D<T>::operator=(
    const CudaSurface2D<T> &other) {
  if (this == &other) {
    return *this;
  }

  Base::operator=(other);

  shared_surface_ = other.shared_surface_;

  boundary_mode_ = other.boundary_mode_;

  x_offset_ = other.x_offset_;
  y_offset_ = other.y_offset_;

  return *this;
}

//------------------------------------------------------------------------------

template <typename T>
inline void CudaSurface2D<T>::CopyTo(T *host_array) const {
  const size_t width_in_bytes = width_ * sizeof(T);
  cudaMemcpy2DFromArray(host_array, width_in_bytes,
                        shared_surface_.get_dev_array(), x_offset_ * sizeof(T),
                        y_offset_, width_in_bytes, height_,
                        cudaMemcpyDeviceToHost);
}

//-------------------------------------------------------------------------------

}  // namespace cua

#endif  // CUDA_SURFACE2D_H_

// libcua: header-only library for interfacing with CUDA array-type objects
// Author: True Price <jtprice at cs.unc.edu>
//
// BSD License
// Copyright (C) 2017-2019  The University of North Carolina at Chapel Hill
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

#ifndef LIBCUA_CUDA_ARRAY2D_H_
#define LIBCUA_CUDA_ARRAY2D_H_

#include "cudaArray2DBase.h"

#include <memory>  // for shared_ptr

#include "cudaArray_fwd.h"

namespace cua {

/**
 * @class CudaArray2D
 * @brief Linear-memory 2D array.
 *
 * This class implements a straightforward interface for linear 2D arrays on the
 * GPU. These arrays are read-able and write-able, and copy/assignment for these
 * arrays is a shallow operation. Use Copy(other) to perform a deep copy.
 *
 * The arrays can be directly passed into device-level code, i.e., you can write
 * kernels that have CudaArray2D objects in their parameter lists:
 *
 *     __global__ void device_kernel(CudaArray2D<float> arr) {
 *       const int x = blockIdx.x * blockDim.x + threadIdx.x;
 *       const int y = blockIdx.y * blockDim.y + threadIdx.y;
 *       arr.set(x, y, 0.0);
 *     }
 */
template <typename T>
class CudaArray2D : public CudaArray2DBase<CudaArray2D<T>> {
 public:
  friend class CudaArray2DBase<CudaArray2D<T>>;

  /// datatype of the array
  typedef T Scalar;

  typedef CudaArray2DBase<CudaArray2D<T>> Base;
  typedef typename Base::SizeType SizeType;
  typedef typename Base::IndexType IndexType;

 protected:
  // for convenience, reference protected base class members directly (they are
  // otherwise not in the current scope because CudaArray2DBase is templated)
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
   */
  CudaArray2D(SizeType width, SizeType height,
              const dim3 block_dim = CudaArray2D<T>::kBlockDim,
              const cudaStream_t stream = 0);  // default stream

  /**
   * Host and device-level copy constructor. This is a shallow-copy operation,
   * meaning that the underlying CUDA memory is the same for both arrays.
   */
  __host__ __device__ CudaArray2D(const CudaArray2D<T> &other);

  ~CudaArray2D();

  //----------------------------------------------------------------------------
  // array operations

  /**
   * Create an empty array of the same size as the current array.
   */
  CudaArray2D<T> EmptyCopy() const;

  /**
   * Create a new empty array with transposed dimensions (flipped height/width).
   */
  CudaArray2D<T> EmptyFlippedCopy() const;

  /**
   * Shallow re-assignment of the given array to share the contents of another.
   * @param other a separate array whose contents will now also be referenced by
   *   the current array
   * @return *this
   */
  CudaArray2D<T> &operator=(const CudaArray2D<T> &other);

  /**
   * Copy the contents of a CPU-bound memory array to the current array. This
   * function assumes that the CPU array has the correct size!
   * @param host_array the CPU-bound array
   * @return *this
   */
  CudaArray2D<T> &operator=(const T *host_array);

  /**
   * Copy the contents of the current array to a CPU-bound memory array. This
   * function assumes that the CPU array has the correct size!
   * @param host_array the CPU-bound array
   */
  void CopyTo(T *host_array) const;

  /**
   * Copy to an array.
   * @param other destination array
   */
  void CopyTo(CudaArray2D<T> *other) const;

  /**
   * Copy to a surface.
   * @param other destination surface
   */
  void CopyTo(CudaSurface2D<T> *other) const;

  /**
   * Copy to a texture.
   * @param other destination texture
   */
  void CopyTo(CudaTexture2D<T> *other) const;

  //----------------------------------------------------------------------------

  /**
   * Create a view onto the underlying CUDA memory. This function assumes that
   * the cropped view region is valid!
   * @param x x-coordinate for the top left of the view
   * @param y y-coordinate for the top left of the view
   * @param width width of the view
   * @param height height of the view
   * @return new CudaArray2D object whose underlying device pointer and size is
   * aligned with the view
   */
  inline CudaArray2D<T> View(IndexType x, IndexType y, SizeType width,
                             SizeType height) const {
    return CudaArray2D<T>(x, y, width, height, *this);
  }

  //----------------------------------------------------------------------------
  // getters/setters

  /**
   * Device-level function for getting the address of an element in an array
   * @param x first coordinate, i.e., the column index in a row-major array
   * @param y second coordinate, i.e., the row index in a row-major array
   * @return pointer to the value at array(x, y)
   */
  __host__ __device__ inline T *ptr(IndexType x = 0, IndexType y = 0) {
    return reinterpret_cast<T *>(reinterpret_cast<char *>(dev_array_ref_) +
                                 y * pitch_ + x * sizeof(T));
  }

  __host__ __device__ inline const T *ptr(IndexType x = 0,
                                          IndexType y = 0) const {
    return reinterpret_cast<const T *>(
        reinterpret_cast<const char *>(dev_array_ref_) + y * pitch_ +
        x * sizeof(T));
  }

  /**
   * Device-level function for setting an element in an array
   * @param x first coordinate, i.e., the column index in a row-major array
   * @param y second coordinate, i.e., the row index in a row-major array
   * @param v the new value to assign to array(x, y)
   */
  __device__ inline void set(IndexType x, IndexType y, const T v) {
    *ptr(x, y) = v;
  }

  /**
   * Device-level function for getting an element in an array
   * @param x first coordinate, i.e., the column index in a row-major array
   * @param y second coordinate, i.e., the row index in a row-major array
   * @return the value at array(x, y)
   */
  __device__ inline T get(IndexType x, IndexType y) const { return *ptr(x, y); }

  /**
   * Get the pitch of the array (the number of bytes in a row for a row-major
   * array).
   */
  __host__ __device__ inline size_t Pitch() const { return pitch_; }

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
  CudaArray2D(IndexType x, IndexType y, SizeType width, SizeType height,
              const CudaArray2D<T> &other);

  size_t pitch_;
  std::shared_ptr<T> dev_array_;
  T *dev_array_ref_;  // equivalent to dev_array_.get(); necessary because that
                      // function is not available on the device
};

//------------------------------------------------------------------------------
// template typedef for CRTP model, a la Eigen

template <typename T>
struct CudaArrayTraits<CudaArray2D<T>> {
  typedef T Scalar;
  typedef bool Mutable;
};

//------------------------------------------------------------------------------
//
// public method implementations
//
//------------------------------------------------------------------------------

template <typename T>
CudaArray2D<T>::CudaArray2D<T>(SizeType width, SizeType height,
                               const dim3 block_dim, const cudaStream_t stream)
    : Base(width, height, block_dim, stream), dev_array_(nullptr) {
  cudaMallocPitch(&dev_array_ref_, &pitch_, sizeof(T) * width_, height_);
#ifdef __CUDA_ARCH__
#else
  dev_array_ = std::shared_ptr<T>(dev_array_ref_, cudaFree);
#endif
}

//------------------------------------------------------------------------------

// host- and device-level copy constructor
template <typename T>
__host__ __device__ CudaArray2D<T>::CudaArray2D<T>(const CudaArray2D<T> &other)
    : Base(other),
      pitch_(other.pitch_),
#ifdef __CUDA_ARCH__
      dev_array_(nullptr),
#else
      dev_array_(other.dev_array_),
#endif
      dev_array_ref_(other.dev_array_ref_) {
}

//------------------------------------------------------------------------------

// host-level private constructor for creating views
template <typename T>
CudaArray2D<T>::CudaArray2D<T>(IndexType x, IndexType y, SizeType width,
                               SizeType height, const CudaArray2D<T> &other)
    : Base(width, height, other.block_dim_, other.stream_),
      pitch_(other.pitch_),
#ifdef __CUDA_ARCH__
      dev_array_(nullptr),
#else
      dev_array_(other.dev_array_),
#endif
      dev_array_ref_(const_cast<T *>(other.ptr(x, y))) {
}

//------------------------------------------------------------------------------

template <typename T>
CudaArray2D<T>::~CudaArray2D<T>() {
  dev_array_.reset();
  dev_array_ref_ = nullptr;

  width_ = 0;
  height_ = 0;
  pitch_ = 0;
}

//------------------------------------------------------------------------------

template <typename T>
inline CudaArray2D<T> CudaArray2D<T>::EmptyCopy() const {
  return CudaArray2D<T>(width_, height_, block_dim_, stream_);
}

//------------------------------------------------------------------------------

template <typename T>
inline CudaArray2D<T> CudaArray2D<T>::EmptyFlippedCopy() const {
  return CudaArray2D<T>(height_, width_, dim3(block_dim_.y, block_dim_.x),
                        stream_);
}

//------------------------------------------------------------------------------

template <typename T>
inline CudaArray2D<T> &CudaArray2D<T>::operator=(const CudaArray2D<T> &other) {
  if (this == &other) {
    return *this;
  }

  Base::operator=(other);

  pitch_ = other.pitch_;
#ifdef __CUDA_ARCH__
#else
  dev_array_ = other.dev_array_;
#endif
  dev_array_ref_ = other.dev_array_ref_;

  return *this;
}

//------------------------------------------------------------------------------

template <typename T>
inline CudaArray2D<T> &CudaArray2D<T>::operator=(const T *host_array) {
  const SizeType width_in_bytes = width_ * sizeof(T);
  cudaMemcpy2D(dev_array_ref_, pitch_, host_array, width_in_bytes,
               width_in_bytes, height_, cudaMemcpyHostToDevice);

  return *this;
}

//------------------------------------------------------------------------------

template <typename T>
inline void CudaArray2D<T>::CopyTo(T *host_array) const {
  const SizeType width_in_bytes = width_ * sizeof(T);
  cudaMemcpy2D(host_array, width_in_bytes, dev_array_ref_, pitch_,
               width_in_bytes, height_, cudaMemcpyDeviceToHost);
}

//------------------------------------------------------------------------------

template <typename T>
inline void CudaArray2D<T>::CopyTo(CudaArray2D<T> *other) const {
  // TODO (True): size checking
  const SizeType width_in_bytes = width_ * sizeof(T);
  cudaMemcpy2D(other->dev_array_ref_, other->pitch_, dev_array_ref_, pitch_,
               width_in_bytes, height_, cudaMemcpyDeviceToDevice);
}

//------------------------------------------------------------------------------

template <typename T>
inline void CudaArray2D<T>::CopyTo(CudaSurface2D<T> *other) const {
  // TODO (True): size checking
  const SizeType width_in_bytes = width_ * sizeof(T);
  cudaMemcpy2DToArray(other->DeviceArray(), other->XOffset() * sizeof(T),
                      other->YOffset(), dev_array_ref_, pitch_, width_in_bytes,
                      height_, cudaMemcpyDeviceToDevice);
}

//------------------------------------------------------------------------------

template <typename T>
inline void CudaArray2D<T>::CopyTo(CudaTexture2D<T> *other) const {
  // TODO (True): size checking
  const SizeType width_in_bytes = width_ * sizeof(T);
  cudaMemcpy2DToArray(other->DeviceArray(), 0, 0, dev_array_ref_, pitch_,
                      width_in_bytes, height_, cudaMemcpyDeviceToDevice);
}

//------------------------------------------------------------------------------

}  // namespace cua

#endif  // LIBCUA_CUDA_ARRAY2D_H_

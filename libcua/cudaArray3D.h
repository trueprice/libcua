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

#ifndef LIBCUA_CUDA_ARRAY3D_H_
#define LIBCUA_CUDA_ARRAY3D_H_

#include "cudaArray3DBase.h"

#include <memory>  // for shared_ptr

namespace cua {

/**
 * @class CudaArray3D
 * @brief Linear-memory 3D array.
 *
 * This class implements a straightforward interface for linear 3D arrays on the
 * GPU. These arrays are read-able and write-able, and copy/assignment for these
 * arrays is a shallow operation. Use Copy(other) to perform a deep copy.
 *
 * The arrays can be directly passed into device-level code, i.e., you can write
 * kernels that have CudaArray3D objects in their parameter lists:
 *
 *     __global__ void device_kernel(CudaArray3D<float> arr) {
 *       const int x = blockIdx.x * blockDim.x + threadIdx.x;
 *       const int y = blockIdx.y * blockDim.y + threadIdx.y;
 *       const int z = blockIdx.z * blockDim.z + threadIdx.z;
 *       arr.set(x, y, z, 0.0);
 *     }
 */
template <typename T>
class CudaArray3D : public CudaArray3DBase<CudaArray3D<T>> {
 public:
  friend class CudaArray3DBase<CudaArray3D<T>>;

  /// datatype of the array
  typedef T Scalar;

  typedef CudaArray3DBase<CudaArray3D<T>> Base;
  typedef typename Base::SizeType SizeType;
  typedef typename Base::IndexType IndexType;

 protected:
  // for convenience, reference protected base class members directly (they are
  // otherwise not in the current scope because CudaArray3DBase is templated)
  using Base::width_;
  using Base::height_;
  using Base::depth_;
  using Base::block_dim_;
  using Base::grid_dim_;
  using Base::stream_;

 public:
  //----------------------------------------------------------------------------
  // constructors and destructor

  /**
   * Constructor.
   * @param width number of elements in the first dimension of the array
   * @param height number of elements in the second dimension of the array
   * @param height number of elements in the third dimension of the array
   * @param block_dim default block size for CUDA kernel calls involving this
   *   object, i.e., the values for blockDim.x/y/z; note that the default grid
   *   dimension is computed automatically based on the array size
   * @param stream CUDA stream for this array object
   */
  CudaArray3D(SizeType width, SizeType height, SizeType depth,
              const dim3 block_dim = CudaArray3D<T>::kBlockDim,
              const cudaStream_t stream = 0);  // default stream

  /**
   * Host and device-level copy constructor. This is a shallow-copy operation,
   * meaning that the underlying CUDA memory is the same for both arrays.
   */
  __host__ __device__ CudaArray3D(const CudaArray3D<T> &other);

  ~CudaArray3D();

  //----------------------------------------------------------------------------
  // array operations

  /**
   * Create an empty array of the same size as the current array.
   */
  CudaArray3D<T> EmptyCopy() const;

  /**
   * Shallow re-assignment of the given array to share the contents of another.
   * @param other a separate array whose contents will now also be referenced by
   *   the current array
   * @return *this
   */
  CudaArray3D<T> &operator=(const CudaArray3D<T> &other);

  /**
   * Copy the contents of a CPU-bound memory array to the current array. This
   * function assumes that the CPU array has the correct size!
   * @param host_array the CPU-bound array
   * @return *this
   */
  CudaArray3D<T> &operator=(const T *host_array);

  /**
   * Copy the contents of the current array to a CPU-bound memory array. This
   * function assumes that the CPU array has the correct size!
   * @param host_array the CPU-bound array
   */
  void CopyTo(T *host_array) const;

  //----------------------------------------------------------------------------

  /**
   * Create a view onto the underlying CUDA memory. This function assumes that
   * the cropped view region is valid!
   * @param x x-coordinate for the top left of the view
   * @param y y-coordinate for the top left of the view
   * @param z z-coordinate for the top left of the view
   * @param width width of the view
   * @param height height of the view
   * @param depth depth of the view
   * @return new CudaArray2D object whose underlying device pointer and size is
   * aligned with the view
   */
  inline CudaArray3D<T> View(IndexType x, IndexType y, IndexType z,
                             SizeType width, SizeType height,
                             SizeType depth) const {
    return CudaArray3D<T>(x, y, z, width, height, depth, *this);
  }

  //----------------------------------------------------------------------------
  // getters/setters

  /**
   * Device-level function for getting the address of element in an array
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @return pointer to the value at array(x, y, z)
   */
  __host__ __device__ inline T *ptr(IndexType x = 0, IndexType y = 0,
                                    IndexType z = 0) {
    return reinterpret_cast<T *>(reinterpret_cast<char *>(dev_array_ref_) +
                                 (z * y_pitch_ + y) * pitch_ + x * sizeof(T));
  }

  __host__ __device__ inline const T *ptr(IndexType x = 0, IndexType y = 0,
                                          IndexType z = 0) const {
    return reinterpret_cast<const T *>(
        reinterpret_cast<const char *>(dev_array_ref_) +
        (z * y_pitch_ + y) * pitch_ + x * sizeof(T));
  }

  /**
   * Device-level function for setting an element in an array
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @param v the new value to assign to array(x, y, z)
   */
  __device__ inline void set(IndexType x, IndexType y, IndexType z, const T v) {
    *ptr(x, y, z) = v;
  }

  /**
   * Device-level function for getting an element in an array
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @return the value at array(x, y, z)
   */
  __device__ inline T get(IndexType x, IndexType y, IndexType z) const {
    return *ptr(x, y, z);
  }

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
   * @param z z-coordinate for the top left of the view
   * @param width width of the view
   * @param height height of the view
   * @param depth height of the view
   */
  CudaArray3D(IndexType x, IndexType y, IndexType z, SizeType width,
              SizeType height, SizeType depth, const CudaArray3D<T> &other);

  size_t pitch_;
  size_t y_pitch_;  // offset when using a view (always equals original height)
  std::shared_ptr<T> dev_array_;
  T *dev_array_ref_;
};

//------------------------------------------------------------------------------
//
// public method implementations
//
//------------------------------------------------------------------------------

template <typename T>
CudaArray3D<T>::CudaArray3D<T>(SizeType width, SizeType height,
                               SizeType depth, const dim3 block_dim,
                               const cudaStream_t stream)
    : Base(width, height, depth, block_dim, stream),
      dev_array_(nullptr),
      y_pitch_(height) {
  cudaPitchedPtr dev_pitched_ptr;
  cudaMalloc3D(&dev_pitched_ptr,
               make_cudaExtent(sizeof(T) * width_, height_, depth_));

  pitch_ = dev_pitched_ptr.pitch;
  dev_array_ref_ = reinterpret_cast<T *>(dev_pitched_ptr.ptr);
#ifdef __CUDA_ARCH__
#else
  dev_array_ = std::shared_ptr<T>(dev_array_ref_, cudaFree);
#endif
}

//------------------------------------------------------------------------------

// host- and device-level copy constructor
template <typename T>
__host__ __device__ CudaArray3D<T>::CudaArray3D<T>(const CudaArray3D<T> &other)
    : Base(other),
      pitch_(other.pitch_),
      y_pitch_(other.y_pitch_),
#ifdef __CUDA_ARCH__
      dev_array_(nullptr),
#else
      dev_array_(other.dev_array_),
#endif
      dev_array_ref_(other.dev_array_ref_) {
}

//------------------------------------------------------------------------------

// host- and device-level private constructor for creating views
template <typename T>
CudaArray3D<T>::CudaArray3D<T>(IndexType x, IndexType y, IndexType z,
                               SizeType width, SizeType height, SizeType depth,
                               const CudaArray3D<T> &other)
    : Base(width, height, depth, other.block_dim_, other.stream_),
      pitch_(other.pitch_),
      y_pitch_(other.y_pitch_),
#ifdef __CUDA_ARCH__
      dev_array_(nullptr),
#else
      dev_array_(other.dev_array_),
#endif
      dev_array_ref_(const_cast<T *>(other.ptr(x, y, z))) {
}

//------------------------------------------------------------------------------

template <typename T>
CudaArray3D<T>::~CudaArray3D<T>() {
  dev_array_.reset();
  dev_array_ref_ = nullptr;
  pitch_ = 0;
  y_pitch_ = 0;

  width_ = 0;
  height_ = 0;
  depth_ = 0;
}

//------------------------------------------------------------------------------

template <typename T>
inline CudaArray3D<T> CudaArray3D<T>::EmptyCopy() const {
  return CudaArray3D<T>(width_, height_, depth_, block_dim_, stream_);
}

//------------------------------------------------------------------------------

template <typename T>
inline CudaArray3D<T> &CudaArray3D<T>::operator=(const T *host_array) {
  size_t width_in_bytes = width_ * sizeof(T);
  cudaMemcpy3DParms params = {0};
  params.srcPtr = make_cudaPitchedPtr(const_cast<T *>(host_array),
                                      width_in_bytes, width_in_bytes, height_);
  params.dstPtr =
      make_cudaPitchedPtr(dev_array_ref_, pitch_, width_in_bytes, y_pitch_);
  params.extent = make_cudaExtent(width_in_bytes, height_, depth_);
  params.kind = cudaMemcpyHostToDevice;

  cudaMemcpy3D(&params);  // last copy is synchronous

  return *this;
}

//------------------------------------------------------------------------------

template <typename T>
inline CudaArray3D<T> &CudaArray3D<T>::operator=(const CudaArray3D<T> &other) {
  if (this == &other) {
    return *this;
  }

  Base::operator=(other);

  pitch_ = other.pitch_;
  y_pitch_ = other.y_pitch_;

  dev_array_ = other.dev_array_;
  dev_array_ref_ = other.dev_array_ref_;

  return *this;
}

//------------------------------------------------------------------------------

template <typename T>
inline void CudaArray3D<T>::CopyTo(T *host_array) const {
  size_t width_in_bytes = width_ * sizeof(T);
  cudaMemcpy3DParms params = {0};
  params.srcPtr =
      make_cudaPitchedPtr(dev_array_ref_, pitch_, width_in_bytes, y_pitch_);
  params.dstPtr =
      make_cudaPitchedPtr(host_array, width_in_bytes, width_in_bytes, height_);
  params.extent = make_cudaExtent(width_in_bytes, height_, depth_);
  params.kind = cudaMemcpyDeviceToHost;

  cudaMemcpy3D(&params);
}

//------------------------------------------------------------------------------

//
// template typedef for CRTP model, a la Eigen
//
template <typename T>
struct CudaArrayTraits<CudaArray3D<T>> {
  typedef T Scalar;
  typedef bool Mutable;
};

}  // namespace cua

#endif  // LIBCUA_CUDA_ARRAY3D_H_

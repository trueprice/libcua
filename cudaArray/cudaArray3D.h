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

#ifndef CUDAMATRIX3D_H_
#define CUDAMATRIX3D_H_

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

  // for convenience, reference protected base class members directly (they are
  // otherwise not in the current scope because CudaArray3DBase is templated)
  using Base::width_;
  using Base::height_;
  using Base::depth_;
  using Base::block_dim_;
  using Base::grid_dim_;
  using Base::stream_;

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
  CudaArray3D(const size_t width, const size_t height, const size_t depth,
              const dim3 block_dim = CudaArray3D<T>::BLOCK_DIM,
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
  CudaArray3D<T> emptyCopy() const;

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
  void copyTo(T *host_array) const;

  //----------------------------------------------------------------------------
  // getters/setters

  /**
   * Device-level function for setting an element in an array
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @param v the new value to assign to array(x, y, z)
   */
  __device__ inline void set(const size_t x, const size_t y, const size_t z,
                             const T v) {
    *((T *)((char *)dev_array_ref_ + (z * height_ + y) * pitch_ +
            x * sizeof(T))) = v;
  }

  /**
   * Device-level function for getting an element in an array
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @return the value at array(x, y, z)
   */
  __device__ inline T get(const size_t x, const size_t y,
                          const size_t z) const {
    return *((T *)((char *)dev_array_ref_ + (z * height_ + y) * pitch_ +
                   x * sizeof(T)));
  }

  //----------------------------------------------------------------------------
  // private class methods and fields

 private:
  size_t pitch_;

  std::shared_ptr<T> dev_array_;
  T *dev_array_ref_;
};

//------------------------------------------------------------------------------
//
// public method implementations
//
//------------------------------------------------------------------------------

template <typename T>
CudaArray3D<T>::CudaArray3D<T>(const size_t width, const size_t height,
                               const size_t depth, const dim3 block_dim,
                               const cudaStream_t stream)
    : Base(width, height, depth, block_dim, stream) {
  cudaPitchedPtr dev_pitched_ptr;
  cudaMalloc3D(&dev_pitched_ptr,
               make_cudaExtent(sizeof(T) * width_, height_, depth_));

  pitch_ = dev_pitched_ptr.pitch;
  dev_array_ref_ = (T *)dev_pitched_ptr.ptr;
  dev_array_ = std::shared_ptr<T>(dev_array_ref_, cudaFree);
}

//------------------------------------------------------------------------------

// host- and device-level copy constructor
template <typename T>
__host__ __device__ CudaArray3D<T>::CudaArray3D<T>(const CudaArray3D<T> &other)
    : Base(other),
      pitch_(other.pitch_),
      dev_array_(nullptr),
      dev_array_ref_(other.dev_array_ref_) {
#ifdef __CUDA_ARCH__
#else
  dev_array_ = other.dev_array_;  // allow this only on the host device
#endif
}

//------------------------------------------------------------------------------

template <typename T>
CudaArray3D<T>::~CudaArray3D<T>() {
  dev_array_.reset();
  dev_array_ref_ = nullptr;
  pitch_ = 0;

  width_ = 0;
  height_ = 0;
  depth_ = 0;
}

//------------------------------------------------------------------------------

template <typename T>
CudaArray3D<T> CudaArray3D<T>::emptyCopy() const {
  return CudaArray3D<T>(width_, height_, depth_, block_dim_, stream_);
}

//------------------------------------------------------------------------------

template <typename T>
CudaArray3D<T> &CudaArray3D<T>::operator=(const T *host_array) {
  size_t width_in_bytes = width_ * sizeof(T);
  cudaMemcpy3DParms params = {0};
  params.srcPtr =
      make_cudaPitchedPtr(host_array, width_in_bytes, width_in_bytes, height_);
  params.dstPtr =
      make_cudaPitchedPtr(dev_array_ref_, pitch_, width_in_bytes, height_);
  params.kind = cudaMemcpyHostToDevice;
  params.extent = make_cudaExtent(width_in_bytes, height_, depth_);

  cudaMemcpy3D(&params);

  return *this;
}

//------------------------------------------------------------------------------

template <typename T>
CudaArray3D<T> &CudaArray3D<T>::operator=(const CudaArray3D<T> &other) {
  if (this == &other) {
    return *this;
  }

  Base::operator=(other);

  pitch_ = other.pitch_;

  dev_array_ = other.dev_array_;
  dev_array_ref_ = other.dev_array_ref_;

  return *this;
}

//------------------------------------------------------------------------------

template <typename T>
void CudaArray3D<T>::copyTo(T *host_array) const {
  size_t width_in_bytes = width_ * sizeof(T);
  cudaMemcpy3DParms params = {0};
  params.srcPtr =
      make_cudaPitchedPtr(dev_array_ref_, pitch_, width_in_bytes, height_);
  params.dstPtr =
      make_cudaPitchedPtr(host_array, width_in_bytes, width_in_bytes, height_);
  params.kind = cudaMemcpyDeviceToHost;
  params.extent = make_cudaExtent(width_in_bytes, height_, depth_);

  cudaMemcpy3D(&params);
}

//------------------------------------------------------------------------------

//
// template typedef for CRTP model, a la Eigen
//
template <typename T>
struct CudaArrayTraits<CudaArray3D<T>> {
  typedef T Scalar;
};

}  // namespace cua

#endif  // CUDAMATRIX3D_H_

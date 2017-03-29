// Author: True Price <jtprice at cs.unc.edu>

#ifndef CUDAMATRIX3D_H_
#define CUDAMATRIX3D_H_

#include "cudaArray3DBase.h"

#include <memory>  // for shared_ptr

namespace cua {

//------------------------------------------------------------------------------
//
// CudaArray3D class definition
//
//------------------------------------------------------------------------------

template <typename T>
class CudaArray3D : public CudaArray3DBase<CudaArray3D<T>> {
 public:
  friend class CudaArray3DBase<CudaArray3D<T>>;

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

  CudaArray3D(const size_t width, const size_t height, const size_t depth,
              const dim3 block_dim = CudaArray3D<T>::BLOCK_DIM,
              const cudaStream_t stream = 0);  // default stream

  __host__ __device__ CudaArray3D(const CudaArray3D<T> &other);

  ~CudaArray3D();

  //
  CudaArray3D<T> emptyCopy() const;

  //
  CudaArray3D<T> &operator=(const CudaArray3D<T> &other);

  //
  CudaArray3D<T> &operator=(const T *host_array);

  //
  void copyTo(T *host_array) const;

  //
  __device__ inline void set(const size_t x, const size_t y, const size_t z,
                             const T v) {
    *((T *)((char *)dev_array_ref_ + (z * height_ + y) * pitch_ +
            x * sizeof(T))) = v;
  }

  //
  __device__ inline T get(const size_t x, const size_t y,
                          const size_t z) const {
    return *((T *)((char *)dev_array_ref_ + (z * height_ + y) * pitch_ +
                   x * sizeof(T)));
  }

 private:
  //
  // private class fields
  //

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

// Author: True Price <jtprice at cs.unc.edu>

#ifndef CUDA_SURFACE2D_ARRAY_H_
#define CUDA_SURFACE2D_ARRAY_H_

#include "cudaArray3DBase.h"

#include <memory>  // for shared_ptr

namespace cua {

//------------------------------------------------------------------------------
//
// CudaSurface3DBase class definition
//
//------------------------------------------------------------------------------

template <typename Derived>
class CudaSurface3DBase : public CudaArray3DBase<Derived> {
  // inherited classes will need to declare get(), set(), typedef Scalar, and
  // static bool IS_LAYERED

 public:
  friend class CudaArray3DBase<Derived>;

  typedef CudaArray3DBase<Derived> Base;
  typedef typename CudaArrayTraits<Derived>::Scalar Scalar;

  // for convenience, reference base class members directly (they are otherwise
  // not in the current scope because CudaArray2DBase is templated)
  using Base::width_;
  using Base::height_;
  using Base::depth_;
  using Base::block_dim_;
  using Base::grid_dim_;
  using Base::stream_;

  CudaSurface3DBase(
      const size_t width, const size_t height, const size_t depth,
      const dim3 block_dim = CudaSurface3DBase<Derived>::BLOCK_DIM,
      const cudaStream_t stream = 0,  // default stream
      const cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeZero);

  __host__ __device__
  CudaSurface3DBase(const CudaSurface3DBase<Derived> &other);

  ~CudaSurface3DBase() {}

  //
  CudaSurface3DBase<Derived> emptyCopy() const;

  //
  CudaSurface3DBase<Derived> &operator=(
      const CudaSurface3DBase<Derived> &other);

  //
  CudaSurface3DBase<Derived> &operator=(const Scalar *host_array);

  //
  void copyTo(Scalar *host_array) const;

  //
  // getters/setters
  //

  __host__ __device__ inline cudaSurfaceBoundaryMode get_boundary_mode() const {
    return boundary_mode_;
  }

 protected:
  //
  // protected class fields
  //

  CudaSharedSurfaceObject<Scalar> surface;

  cudaSurfaceBoundaryMode boundary_mode_;
};

//------------------------------------------------------------------------------
//
// public method implementations
//
//------------------------------------------------------------------------------

template <typename Derived>
CudaSurface3DBase<Derived>::CudaSurface3DBase<Derived>(
    const size_t width, const size_t height, const size_t depth,
    const dim3 block_dim, const cudaStream_t stream,
    const cudaSurfaceBoundaryMode boundary_mode)
    : Base(width, height, depth, block_dim, stream),
      boundary_mode_(boundary_mode),
      surface(width, height, depth, CudaArrayTraits<Derived>::IS_LAYERED) {}

//------------------------------------------------------------------------------

// host- and device-level copy constructor
template <typename Derived>
__host__ __device__ CudaSurface3DBase<Derived>::CudaSurface3DBase<Derived>(
    const CudaSurface3DBase<Derived> &other)
    : Base(other),
      boundary_mode_(other.boundary_mode_),
      surface(other.surface) {}

//------------------------------------------------------------------------------

template <typename Derived>
CudaSurface3DBase<Derived> CudaSurface3DBase<Derived>::emptyCopy() const {
  return CudaSurface3DBase<Derived>(width_, height_, block_dim_, stream_,
                                    boundary_mode_);
}

//------------------------------------------------------------------------------

template <typename Derived>
CudaSurface3DBase<Derived> &CudaSurface3DBase<Derived>::operator=(
    const CudaSurface3DBase<Derived>::Scalar *host_array) {
  cudaMemcpyToArray(surface.dev_array, 0, 0, host_array,
                    sizeof(Scalar) * width_ * height_ * depth_,
                    cudaMemcpyHostToDevice);

  return *this;
}

//------------------------------------------------------------------------------

template <typename Derived>
CudaSurface3DBase<Derived> &CudaSurface3DBase<Derived>::operator=(
    const CudaSurface3DBase<Derived> &other) {
  if (this == &other) {
    return *this;
  }

  Base::operator=(other);

  surface = other.surface;

  boundary_mode_ = other.boundary_mode_;

  return *this;
}

//------------------------------------------------------------------------------

template <typename Derived>
void CudaSurface3DBase<Derived>::copyTo(
    CudaSurface3DBase<Derived>::Scalar *host_array) const {
  cudaMemcpyFromArray(host_array, surface.dev_array, 0, 0,
                      sizeof(Scalar) * width_ * height_ * depth_,
                      cudaMemcpyDeviceToHost);
}

//------------------------------------------------------------------------------
//
// Implementations (get and set, mainly)
//
//------------------------------------------------------------------------------

template <typename T>
class CudaSurface2DArray : public CudaSurface3DBase<CudaSurface2DArray<T>> {
 public:
  using CudaSurface3DBase<CudaSurface2DArray<T>>::CudaSurface3DBase;

  //
  __device__ inline void set(const int x, const int y, const int z, const T v) {
    surf2DLayeredwrite(v, this->surface.get_cuda_api_object(), sizeof(T) * x, y,
                       z, this->boundary_mode_);
  }

  //
  __device__ inline T get(const int x, const int y, const int z) const {
    return surf2DLayeredread<T>(this->surface.get_cuda_api_object(),
                                sizeof(T) * x, y, z, this->boundary_mode_);
  }
};

//------------------------------------------------------------------------------

template <typename T>
class CudaSurface3D : public CudaSurface3DBase<CudaSurface3D<T>> {
 public:
  using CudaSurface3DBase<CudaSurface3D<T>>::CudaSurface3DBase;

  //
  __device__ inline void set(const int x, const int y, const int z, const T v) {
    surf3Dwrite(v, this->surface.get_cuda_api_object(), sizeof(T) * x, y, z,
                this->boundary_mode_);
  }

  //
  __device__ inline T get(const int x, const int y, const int z) const {
    return surf3Dread<T>(this->surface.get_cuda_api_object(), sizeof(T) * x, y,
                         z, this->boundary_mode_);
  }
};

//------------------------------------------------------------------------------
//
// template typedefs for CRTP model, a la Eigen
//
//------------------------------------------------------------------------------

template <typename T>
struct CudaArrayTraits<CudaSurface2DArray<T>> {
  typedef T Scalar;
  static const bool IS_LAYERED = true;
};

template <typename T>
struct CudaArrayTraits<CudaSurface3D<T>> {
  typedef T Scalar;
  static const bool IS_LAYERED = false;
};

}  // namespace cua

#endif  // CUDA_SURFACE2D_ARRAY_H_

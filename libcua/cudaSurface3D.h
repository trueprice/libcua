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

#ifndef LIBCUA_CUDA_SURFACE3D_H_
#define LIBCUA_CUDA_SURFACE3D_H_

#include "cudaArray3DBase.h"
#include "cudaSharedArrayObject.h"

#include "cudaArray_fwd.h"
#include "util.h"

namespace cua {

/**
 * @class CudaSurface3DBase
 * @brief Base class for a surface-memory 3D array.
 *
 * This class implements an interface for 3D surface-memory arrays on the GPU.
 * These arrays are read-able and write-able, and compared to linear-memory
 * array they have better cache coherence properties for memory accesses in a 3D
 * neighborhood. Copy/assignment for CudaSurface3D objects is a shallow
 * operation; use `Copy()` or `CopyTo(other)` to perform a deep copy.
 *
 * Derived classes implement array access for both layered 2D (that is, an array
 * of 2D arrays) and 3D surface-memory arrays.
 *
 * The arrays can be directly passed into device-level code, i.e., you can write
 * kernels that have CudaSurface3D objects in their parameter lists:
 *
 *     __global__ void device_kernel(CudaSurface3D<float> arr) {
 *       const int x = blockIdx.x * blockDim.x + threadIdx.x;
 *       const int y = blockIdx.y * blockDim.y + threadIdx.y;
 *       const int z = blockIdx.z * blockDim.z + threadIdx.z;
 *       arr.set(x, y, z, 0.0f);
 *     }
 */
template <typename Derived>
class CudaSurface3DBase : public CudaArray3DBase<Derived> {
  // inherited classes will need to declare get(), set(), typedef Scalar, and
  // typedef std::<true/false>_type IsLayered

 public:
  friend class CudaArray3DBase<Derived>;

  /// datatype of the array
  typedef typename CudaArrayTraits<Derived>::Scalar Scalar;

  typedef CudaArray3DBase<Derived> Base;
  typedef typename Base::SizeType SizeType;
  typedef typename Base::IndexType IndexType;

 protected:
  // for convenience, reference base class members directly (they are otherwise
  // not in the current scope because CudaArray2DBase is templated)
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
   * @param boundary_mode boundary mode to use for reads that go outside the 3D
   *   extents of the array
   */
  CudaSurface3DBase(
      SizeType width, SizeType height, SizeType depth,
      const dim3 block_dim = CudaSurface3DBase<Derived>::kBlockDim,
      const cudaStream_t stream = 0,  // default stream
      const cudaSurfaceBoundaryMode boundary_mode = cudaBoundaryModeZero);

  /**
   * Host and device-level copy constructor. This is a shallow-copy operation,
   * meaning that the underlying CUDA memory is the same for both arrays.
   */
  __host__ __device__
  CudaSurface3DBase(const CudaSurface3DBase<Derived> &other);

  /**
   * Create a view onto the underlying CUDA memory. This function assumes that
   * the cropped view region is valid!
   * @param x x-coordinate for the top left of the view
   * @param y y-coordinate for the top left of the view
   * @param z z-coordinate for the top left of the view
   * @param width width of the view
   * @param height height of the view
   * @param depth depth of the view
   * @return new CudaSurface3D view whose underlying device pointer and size is
   * aligned with the view
   */
  inline Derived View(IndexType x, IndexType y, IndexType z, SizeType width,
                      SizeType height, SizeType depth) const {
    return CudaSurface3DBase<Derived>(x, y, z, width, height, depth, *this)
        .derived();
  }

  //----------------------------------------------------------------------------
  // array operations

  /**
   * Create an empty array of the same size as the current array.
   */
  CudaSurface3DBase<Derived> EmptyCopy() const;

  /**
   * Shallow re-assignment of the given array to share the contents of another.
   * @param other a separate array whose contents will now also be referenced by
   *   the current array
   * @return *this
   */
  CudaSurface3DBase<Derived> &operator=(
      const CudaSurface3DBase<Derived> &other);

  /**
   * Copy the contents of a CPU-bound memory array to the current array. This
   * function assumes that the CPU array has the correct size!
   * @param host_array the CPU-bound array
   * @return *this
   */
  CudaSurface3DBase<Derived> &operator=(const Scalar *host_array);

  /**
   * Copy the contents of the current array to a CPU-bound memory array. This
   * function assumes that the CPU array has the correct size!
   * @param host_array the CPU-bound array
   */
  void CopyTo(Scalar *host_array) const;

  /**
   * Copy to an array.
   * @param other destination array
   */
  void CopyTo(CudaArray3D<Scalar> *other) const;

  /**
   * Copy to a surface.
   * @param other destination surface
   */
  template <typename OtherDerived>
  void CopyTo(CudaSurface3DBase<OtherDerived> *other) const;

  /**
   * Copy to a texture.
   * @param other destination texture
   */
  template <typename OtherDerived>
  void CopyTo(CudaTexture3DBase<OtherDerived> *other) const;

  //----------------------------------------------------------------------------
  // getters/setters

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

 protected:
  //
  // protected class fields
  //

  CudaSharedSurfaceObject<Scalar> shared_surface_;

  cudaSurfaceBoundaryMode boundary_mode_;

  IndexType x_offset_, y_offset_, z_offset_;

 private:
  /**
   * Internal constructor used for creating views.
   * @param x x-coordinate for the top left of the view
   * @param y y-coordinate for the top left of the view
   * @param z z-coordinate for the top left of the view
   * @param width width of the view
   * @param height height of the view
   * @param depth depth of the view
   */
  CudaSurface3DBase(IndexType x, IndexType y, IndexType z, SizeType width,
                    SizeType height, SizeType depth,
                    const CudaSurface3DBase<Derived> &other);
};

//------------------------------------------------------------------------------
//
// public method implementations
//
//------------------------------------------------------------------------------

template <typename Derived>
CudaSurface3DBase<Derived>::CudaSurface3DBase<Derived>(
    SizeType width, SizeType height, SizeType depth,
    const dim3 block_dim, const cudaStream_t stream,
    const cudaSurfaceBoundaryMode boundary_mode)
    : Base(width, height, depth, block_dim, stream),
      boundary_mode_(boundary_mode),
      shared_surface_(width, height, depth,
                      CudaArrayTraits<Derived>::IsLayered::value),
      x_offset_(0),
      y_offset_(0),
      z_offset_(0) {}

//------------------------------------------------------------------------------

// host- and device-level copy constructor
template <typename Derived>
__host__ __device__ CudaSurface3DBase<Derived>::CudaSurface3DBase<Derived>(
    const CudaSurface3DBase<Derived> &other)
    : Base(other),
      boundary_mode_(other.boundary_mode_),
      shared_surface_(other.shared_surface_),
      x_offset_(other.x_offset_),
      y_offset_(other.y_offset_),
      z_offset_(other.z_offset_) {}

//------------------------------------------------------------------------------

// host-level private constructor for creating views
template <typename Derived>
CudaSurface3DBase<Derived>::CudaSurface3DBase<Derived>(
    IndexType x, IndexType y, IndexType z, SizeType width, SizeType height,
    SizeType depth, const CudaSurface3DBase<Derived> &other)
    : Base(width, height, depth, other.block_dim_, other.stream_),
      boundary_mode_(other.boundary_mode_),
      shared_surface_(other.shared_surface_),
      x_offset_(x + other.x_offset_),
      y_offset_(y + other.y_offset_),
      z_offset_(z + other.z_offset_) {}

//------------------------------------------------------------------------------

template <typename Derived>
inline CudaSurface3DBase<Derived> CudaSurface3DBase<Derived>::EmptyCopy()
    const {
  return CudaSurface3DBase<Derived>(width_, height_, depth_, block_dim_,
                                    stream_, boundary_mode_);
}

//------------------------------------------------------------------------------

template <typename Derived>
inline CudaSurface3DBase<Derived> &CudaSurface3DBase<Derived>::operator=(
    const Scalar *host_array) {
  internal::CheckNotNull(host_array);

  cudaMemcpy3DParms params = {0};
  params.srcPtr = make_cudaPitchedPtr(const_cast<Scalar *>(host_array),
                                      width_ * sizeof(Scalar), width_, height_);
  params.dstArray = shared_surface_.DeviceArray();
  params.dstPos = make_cudaPos(x_offset_, y_offset_, z_offset_);
  params.extent = make_cudaExtent(width_, height_, depth_);
  params.kind = cudaMemcpyHostToDevice;

  cudaMemcpy3D(&params);  // last copy is synchronous

  return *this;
}

//------------------------------------------------------------------------------

template <typename Derived>
inline CudaSurface3DBase<Derived> &CudaSurface3DBase<Derived>::operator=(
    const CudaSurface3DBase<Derived> &other) {
  if (this == &other) {
    return *this;
  }

  Base::operator=(other);

  shared_surface_ = other.shared_surface_;

  boundary_mode_ = other.boundary_mode_;

  x_offset_ = other.x_offset_;
  y_offset_ = other.y_offset_;
  z_offset_ = other.z_offset_;

  return *this;
}

//------------------------------------------------------------------------------

template <typename Derived>
inline void CudaSurface3DBase<Derived>::CopyTo(
    CudaSurface3DBase<Derived>::Scalar *host_array) const {
  internal::CheckNotNull(host_array);

  cudaMemcpy3DParms params = {0};
  params.srcArray = shared_surface_.DeviceArray();
  params.srcPos = make_cudaPos(x_offset_, y_offset_, z_offset_);
  params.dstPtr = make_cudaPitchedPtr(const_cast<Scalar *>(host_array),
                                      width_ * sizeof(Scalar), width_, height_);
  params.extent = make_cudaExtent(width_, height_, depth_);
  params.kind = cudaMemcpyDeviceToHost;

  cudaMemcpy3D(&params);
}

//------------------------------------------------------------------------------

template <typename Derived>
inline void CudaSurface3DBase<Derived>::CopyTo(
    CudaArray3D<Scalar> *other) const {
  internal::CheckNotNull(other);
  internal::CheckSizeEqual3D(*this, *other);

  cudaMemcpy3DParms params = {0};
  params.srcArray = shared_surface_.DeviceArray();
  params.srcPos = make_cudaPos(x_offset_, y_offset_, z_offset_);
  params.dstPtr = other->GetPitchedPtr();
  params.extent = make_cudaExtent(width_, height_, depth_);
  params.kind = cudaMemcpyDeviceToDevice;

  cudaMemcpy3D(&params);
}

//------------------------------------------------------------------------------

template <typename Derived>
template <typename OtherDerived>
inline void CudaSurface3DBase<Derived>::CopyTo(
    CudaSurface3DBase<OtherDerived> *other) const {
  internal::CheckNotNull(other);
  internal::CheckSizeEqual3D(*this, *other);

  cudaMemcpy3DParms params = {0};
  params.srcArray = shared_surface_.DeviceArray();
  params.srcPos = make_cudaPos(x_offset_, y_offset_, z_offset_);
  params.dstArray = other->DeviceArray();
  params.dstPos =
      make_cudaPos(other->x_offset_, other->y_offset_, other->z_offset_);
  params.extent = make_cudaExtent(width_, height_, depth_);
  params.kind = cudaMemcpyDeviceToDevice;

  cudaMemcpy3D(&params);
}

//------------------------------------------------------------------------------

template <typename Derived>
template <typename OtherDerived>
inline void CudaSurface3DBase<Derived>::CopyTo(
    CudaTexture3DBase<OtherDerived> *other) const {
  internal::CheckNotNull(other);
  internal::CheckSizeEqual3D(*this, *other);

  cudaMemcpy3DParms params = {0};
  params.srcArray = shared_surface_.DeviceArray();
  params.srcPos = make_cudaPos(x_offset_, y_offset_, z_offset_);
  params.dstArray = other->DeviceArray();
  params.dstPos = make_cudaPos(0, 0, 0);
  params.extent = make_cudaExtent(width_, height_, depth_);
  params.kind = cudaMemcpyDeviceToDevice;

  cudaMemcpy3D(&params);
}

//------------------------------------------------------------------------------
//
// Sub-class implementations (layered 2D arrays and pure 3D arrays)
//
//------------------------------------------------------------------------------

/**
 * @class CudaSurface2DArray
 * @brief Array of surface-memory 2D arrays.
 *
 * See CudaSurface3DBase for more details.
 */
template <typename T>
class CudaSurface2DArray : public CudaSurface3DBase<CudaSurface2DArray<T>> {
 public:
  using CudaSurface3DBase<CudaSurface2DArray<T>>::CudaSurface3DBase;
  using CudaSurface3DBase<CudaSurface2DArray<T>>::operator=;

  ~CudaSurface2DArray() {}

  /**
   * Device-level function for setting an element in an array
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @param v the new value to assign to array(x, y, z)
   */
  __device__ inline void set(const int x, const int y, const int z, const T v) {
    surf2DLayeredwrite(v, this->shared_surface_.CudaApiObject(),
                       sizeof(T) * (x + this->x_offset_), y + this->y_offset_,
                       z + this->z_offset_, this->boundary_mode_);
  }

  /**
   * Device-level function for getting an element in an array
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @return the value at array(x, y, z)
   */
  __device__ inline T get(const int x, const int y, const int z) const {
    return surf2DLayeredread<T>(this->shared_surface_.CudaApiObject(),
                                sizeof(T) * (x + this->x_offset_),
                                y + this->y_offset_, z + this->z_offset_,
                                this->boundary_mode_);
  }
};

//------------------------------------------------------------------------------

/**
 * @class CudaSurface3D
 * @brief Surface-memory 3D array.
 *
 * See CudaSurface3DBase for more details.
 */
template <typename T>
class CudaSurface3D : public CudaSurface3DBase<CudaSurface3D<T>> {
 public:
  using CudaSurface3DBase<CudaSurface3D<T>>::CudaSurface3DBase;
  using CudaSurface3DBase<CudaSurface3D<T>>::operator=;

  ~CudaSurface3D() {}

  /**
   * Device-level function for setting an element in an array
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @param v the new value to assign to array(x, y, z)
   */
  __device__ inline void set(const int x, const int y, const int z, const T v) {
    surf3Dwrite(v, this->shared_surface_.CudaApiObject(),
                sizeof(T) * (x + this->x_offset_), y + this->y_offset_,
                z + this->z_offset_, this->boundary_mode_);
  }

  /**
   * Device-level function for getting an element in an array
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @return the value at array(x, y, z)
   */
  __device__ inline T get(const int x, const int y, const int z) const {
    return surf3Dread<T>(this->shared_surface_.CudaApiObject(),
                         sizeof(T) * (x + this->x_offset_), y + this->y_offset_,
                         z + this->z_offset_, this->boundary_mode_);
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
  typedef bool Mutable;
  typedef std::true_type IsLayered;
};

template <typename T>
struct CudaArrayTraits<CudaSurface3D<T>> {
  typedef T Scalar;
  typedef bool Mutable;
  typedef std::false_type IsLayered;
};

}  // namespace cua

#endif  // LIBCUA_CUDA_SURFACE3D_H_

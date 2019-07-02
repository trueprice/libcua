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

#ifndef LIBCUA_CUDA_TEXTURE3D_H_
#define LIBCUA_CUDA_TEXTURE3D_H_

#include "cudaArray3DBase.h"
#include "cudaSharedArrayObject.h"

namespace cua {

/**
 * @class CudaTexture3DBase
 * @brief Base class for a surface-memory 3D array.
 *
 * This class implements an interface for 3D texture-memory arrays on the GPU.
 * These arrays are read-only, and copy for CudaTexture3D objects is a shallow
 * operation.
 *
 * Derived classes implement array access for both layered 2D (that is, an array
 * of 2D arrays) and 3D texture-memory arrays.
 *
 * The arrays can be directly passed into device-level code, i.e., you can write
 * kernels that have CudaTexture2D objects in their parameter lists:
 *
 *     __global__ void device_kernel(const CudaTexture3D<float> in,
 *                                   CudaSurface3D<float> out) {
 *       const int x = blockIdx.x * blockDim.x + threadIdx.x;
 *       const int y = blockIdx.y * blockDim.y + threadIdx.y;
 *       const int z = blockIdx.z * blockDim.z + threadIdx.z;
 *       out.set(x, y, z, in.get(x, y, z));
 *     }
 *
 * TODO (True): mipmaps, etc. would be useful, as well as texture-coordinate
 *   lookups, etc.
 */
template <typename Derived>
class CudaTexture3DBase : public CudaArray3DBase<Derived> {
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
  CudaTexture3DBase(
      SizeType width, SizeType height, SizeType depth,
      const cudaTextureFilterMode filter_mode = cudaFilterModePoint,
      const cudaTextureAddressMode address_mode = cudaAddressModeBorder,
      const cudaTextureReadMode read_mode = cudaReadModeElementType,
      const dim3 block_dim = CudaTexture3DBase<Derived>::kBlockDim,
      const cudaStream_t stream = 0);  // default stream

  /**
   * Host and device-level copy constructor. This is a shallow-copy operation,
   * meaning that the underlying CUDA memory is the same for both arrays.
   */
  __host__ __device__
  CudaTexture3DBase(const CudaTexture3DBase<Derived> &other);

  //----------------------------------------------------------------------------
  // array operations

  /**
   * Shallow re-assignment of the given array to share the contents of another.
   * @param other a separate array whose contents will now also be referenced by
   *   the current array
   * @return *this
   */
  CudaTexture3DBase<Derived> &operator=(
      const CudaTexture3DBase<Derived> &other);

  /**
   * Copy the contents of a CPU-bound memory array to the current array. This
   * function assumes that the CPU array has the correct size!
   * @param host_array the CPU-bound array
   * @return *this
   */
  CudaTexture3DBase<Derived> &operator=(const Scalar *host_array);

  /**
   * Copy the contents of the current array to a CPU-bound memory array. This
   * function assumes that the CPU array has the correct size!
   * @param host_array the CPU-bound array
   */
  void CopyTo(Scalar *host_array) const;

 protected:
  //
  // protected class fields
  //

  CudaSharedTextureObject<Scalar> shared_texture_;
};

//------------------------------------------------------------------------------
//
// public method implementations
//
//------------------------------------------------------------------------------

template <typename Derived>
CudaTexture3DBase<Derived>::CudaTexture3DBase<Derived>(
    SizeType width, SizeType height, SizeType depth,
    const cudaTextureFilterMode filter_mode,
    const cudaTextureAddressMode address_mode,
    const cudaTextureReadMode read_mode, const dim3 block_dim,
    const cudaStream_t stream)
    : Base(width, height, depth, block_dim, stream),
      shared_texture_(width, height, depth, filter_mode, address_mode,
                      read_mode, CudaArrayTraits<Derived>::IsLayered::value) {}

//------------------------------------------------------------------------------

// host- and device-level copy constructor
template <typename Derived>
__host__ __device__ CudaTexture3DBase<Derived>::CudaTexture3DBase<Derived>(
    const CudaTexture3DBase<Derived> &other)
    : Base(other), shared_texture_(other.shared_texture_) {}

//------------------------------------------------------------------------------

template <typename Derived>
inline CudaTexture3DBase<Derived> &CudaTexture3DBase<Derived>::operator=(
    const CudaTexture3DBase<Derived> &other) {
  if (this == &other) {
    return *this;
  }

  Base::operator=(other);

  shared_texture_ = other.shared_texture_;

  return *this;
}

//------------------------------------------------------------------------------

template <typename Derived>
inline CudaTexture3DBase<Derived> &CudaTexture3DBase<Derived>::operator=(
    const Scalar *host_array) {
  cudaMemcpy3DParms params = {0};
  params.srcPtr = make_cudaPitchedPtr(const_cast<Scalar *>(host_array),
                                      width_ * sizeof(Scalar), width_, height_);
  params.dstArray = shared_texture_.DeviceArray();
  params.extent = make_cudaExtent(width_, height_, depth_);
  params.kind = cudaMemcpyHostToDevice;

  cudaMemcpy3D(&params);

  return *this;
}

//------------------------------------------------------------------------------

template <typename Derived>
inline void CudaTexture3DBase<Derived>::CopyTo(
    CudaTexture3DBase<Derived>::Scalar *host_array) const {
  cudaMemcpy3DParms params = {0};
  params.srcArray = shared_texture_.DeviceArray();
  params.dstPtr = make_cudaPitchedPtr(const_cast<Scalar *>(host_array),
                                      width_ * sizeof(Scalar), width_, height_);
  params.extent = make_cudaExtent(width_, height_, depth_);
  params.kind = cudaMemcpyDeviceToHost;

  cudaMemcpy3D(&params);
}

//------------------------------------------------------------------------------
//
// Sub-class implementations (layered 2D arrays and pure 3D arrays)
//
//------------------------------------------------------------------------------

/**
 * @class CudaTexture2DArray
 * @brief Array of surface-memory 2D arrays.
 *
 * See CudaTexture3DBase for more details.
 */
template <typename T>
class CudaTexture2DArray : public CudaTexture3DBase<CudaTexture2DArray<T>> {
 public:
  using CudaTexture3DBase<CudaTexture2DArray<T>>::CudaTexture3DBase;
  using CudaTexture3DBase<CudaTexture2DArray<T>>::operator=;

  ~CudaTexture2DArray() {}

  /**
   * Device-level function for getting a texture pixel value. Note, if you use
   * cudaReadModeNormalizedFloat as the texture read mode, you'll need to
   * specify the appropriate return type (i.e., float) in the template argument.
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @return the value at array(x, y, z)
   */
  template <typename ReturnType = T>
  __device__ inline ReturnType get(const int x, const int y,
                                   const int z) const {
    return tex2DLayered<ReturnType>(this->shared_texture_.CudaApiObject(),
                                    x + 0.5f, y + 0.5f, z + 0.5f);
  }

  /**
   * Device-level function for getting an interpolated texture pixel value,
   * which is enabled by specifying filter_mode as cudaFilterModeLinear in the
   * constructor. Note, if you use cudaReadModeNormalizedFloat as the texture
   * read mode, you'll need to specify the appropriate return type (i.e., float)
   * in the template argument.
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @return the interpolated value at array(x, y, z)
   */
  // to properly use cudaReadModeNormalizedFloat, you'll need to specify the
  // appropriate return type (e.g., float) in the template argument
  template <typename ReturnType = T>
  __device__ inline ReturnType interp(const float x, const float y,
                                      const float z) const {
    return tex2DLayered<ReturnType>(this->shared_texture.CudaApiObject(), x, y,
                                    z);
  }
};

//------------------------------------------------------------------------------

/**
 * @class CudaTexture3D
 * @brief Texture-memory 3D array.
 *
 * See CudaTexture3DBase for more details.
 */
template <typename T>
class CudaTexture3D : public CudaTexture3DBase<CudaTexture3D<T>> {
 public:
  using CudaTexture3DBase<CudaTexture3D<T>>::CudaTexture3DBase;
  using CudaTexture3DBase<CudaTexture3D<T>>::operator=;

  ~CudaTexture3D() {}

  /**
   * Device-level function for getting a texture pixel value. Note, if you use
   * cudaReadModeNormalizedFloat as the texture read mode, you'll need to
   * specify the appropriate return type (i.e., float) in the template argument.
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @return the value at array(x, y, z)
   */
  template <typename ReturnType = T>
  __device__ inline ReturnType get(const int x, const int y,
                                   const int z) const {
    return tex3D<ReturnType>(this->shared_texture_.CudaApiObject(), x + 0.5f,
                             y + 0.5f, z + 0.5f);
  }

  /**
   * Device-level function for getting an interpolated texture pixel value,
   * which is enabled by specifying filter_mode as cudaFilterModeLinear in the
   * constructor. Note, if you use cudaReadModeNormalizedFloat as the texture
   * read mode, you'll need to specify the appropriate return type (i.e., float)
   * in the template argument.
   * @param x first coordinate
   * @param y second coordinate
   * @param z third coordinate
   * @return the interpolated value at array(x, y, z)
   */
  // to properly use cudaReadModeNormalizedFloat, you'll need to specify the
  // appropriate return type (e.g., float) in the template argument
  template <typename ReturnType = T>
  __device__ inline ReturnType interp(const float x, const float y,
                                      const float z) const {
    return tex3D<ReturnType>(this->shared_texture_.CudaApiObject(), x, y, z);
  }
};

//------------------------------------------------------------------------------
//
// template typedefs for CRTP model, a la Eigen
//
//------------------------------------------------------------------------------

template <typename T>
struct CudaArrayTraits<CudaTexture2DArray<T>> {
  typedef T Scalar;
  typedef std::true_type IsLayered;
};

template <typename T>
struct CudaArrayTraits<CudaTexture3D<T>> {
  typedef T Scalar;
  typedef std::false_type IsLayered;
};

}  // namespace cua

#endif  // LIBCUA_CUDA_TEXTURE3D_H_

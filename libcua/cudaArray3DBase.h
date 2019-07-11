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

#ifndef LIBCUA_CUDA_ARRAY3D_BASE_H_
#define LIBCUA_CUDA_ARRAY3D_BASE_H_

#include <curand.h>
#include <curand_kernel.h>

#include "types.h"

namespace cua {

// Convenience macro for using SFINAE to disable operations given non-writable
// subclasses.
#define ENABLE_IF_MUTABLE                       \
  template <class C = CudaArrayTraits<Derived>, \
            typename C::Mutable is_mutable = true>
// This is used in out-of-class definitions.
#define ENABLE_IF_MUTABLE_IMPL \
  template <class C, typename C::Mutable is_mutable>

namespace kernel {

//
// kernel definitions
// TODO: once we have more kernel functions, move them to a separate file
//

//
// copy values of one surface to another, possibly with different datatypes
//
template <typename SrcCls, typename DstCls>
__global__ void CudaArray3DBaseCopyTo(const SrcCls src, DstCls dst) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < src.Width() && y < src.Height() && z < src.Depth()) {
    dst.set(x, y, z, static_cast<typename DstCls::Scalar>(src.get(x, y, z)));
  }
}

//
// arithmetic operations
// op: __device__ function mapping (x,y) -> CudaArrayClass::Scalar
//
template <typename CudaArrayClass, class Function>
__global__ void CudaArray3DBaseApplyOp(CudaArrayClass array, Function op) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < array.Width() && y < array.Height() && z < array.Depth()) {
    array.set(x, y, z, op(x, y, z));
  }
}

//------------------------------------------------------------------------------

//
// fill an array with a value
//
template <typename CudaArrayClass, typename T>
__global__ void CudaArray3DBaseFill(CudaArrayClass array, const T value) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < array.Width() && y < array.Height() && z < array.Depth()) {
    array.set(x, y, z, value);
  }
}

//------------------------------------------------------------------------------

//
// fillRandom: fill with random values
//
template <typename CudaRandomStateArrayClass, typename CudaArrayClass,
          typename RandomFunction>
__global__ void CudaArray3DBaseFillRandom(CudaRandomStateArrayClass rand_state,
                                          CudaArrayClass array,
                                          RandomFunction func) {
  const unsigned int x = blockIdx.x * CudaArrayClass::kTileSize + threadIdx.x;
  const unsigned int y = blockIdx.y * CudaArrayClass::kTileSize + threadIdx.y;
  const unsigned int z = blockIdx.z * CudaArrayClass::kTileSize + threadIdx.z;

  // Each thread processes kBlockRows contiguous rows in y, and then repeats
  // this for kBlockRows contiguous depth rows in z.

  curandState_t state = rand_state.get(blockIdx.x, blockIdx.y, blockIdx.z);
  skipahead(static_cast<unsigned long long>(
                ((threadIdx.z * CudaArrayClass::kBlockRows + threadIdx.y) *
                     CudaArrayClass::kTileSize +
                 threadIdx.x) *
                CudaArrayClass::kBlockRows * CudaArrayClass::kBlockRows),
            &state);

  for (unsigned int k = 0; k < CudaArrayClass::kTileSize;
       k += CudaArrayClass::kBlockRows) {
    for (unsigned int j = 0; j < CudaArrayClass::kTileSize;
         j += CudaArrayClass::kBlockRows) {
      const auto value = func(&state);
      if (x < array.Width() && y + j < array.Height() &&
          z + k < array.Depth()) {
        array.set(x, y + j, z + k, value);
      }
    }
  }

  // update the global random state
  if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 &&
      threadIdx.z == blockDim.z - 1) {
    rand_state.set(blockIdx.x, blockIdx.y, blockIdx.z, state);
  }
}

}  // namespace kernel

//------------------------------------------------------------------------------

// Any derived class will need to declare
/*
 * template <typename T>
 * struct CudaArrayTraits<Derived<T>> {
 *   typedef T Scalar;
 * };
 */
template <typename Derived>
struct CudaArrayTraits;  // forward declaration

/**
 * @class CudaArray3DBase
 * @brief Base class for all 3D CudaArray-type objects.
 *
 * This class includes implementations for Copy, etc.
 * All derived classes need to define the following methods:
 *
 * - copy constructor on host *and* device; use `#ifndef __CUDA_ARCH__` to
 *   perform host-specific instructions
 *   - `__host__ __device__ Derived(const Derived &other);`
 * - get(): read from array position
 *   - `__device__
 *   inline Scalar get(const unsigned int x, const unsigned int y, const
 *                     unsigned int z) const;`
 *
 * These methods are suggested for getting data to/from the CPU:
 *
 * - operator=()
 *   - `Derived &operator=(const Scalar *host_array);`
 * - CopyTo()
 *   - `void CopyTo(Scalar *host_array) const;`
 *
 * These methods are necessary for derived classes that are read-write:
 *
 * - EmptyCopy(): to create a new array of the same size
 *   - `Derived EmptyCopy() const;`
 * - set(): write to array position (optional for readonly subclasses)
 *   - `__device__ inline void set(unsigned int x, unsigned int y,
 *                                 unsigned int z, Scalar value);`
 *
 * Also, any derived class will need to separately declare a CudaArrayTraits
 * struct instantiation with a "Scalar" member, e.g.,
 *
 *     template <typename T>
 *     struct CudaArrayTraits<Derived<T>> {
 *       typedef T Scalar;
 *       typedef bool Mutable;  // defined for read-write derived classes
 *     };
 */
template <typename Derived>
class CudaArray3DBase {
 public:
  //----------------------------------------------------------------------------
  // static class elements and typedefs

  /// datatype of the array
  typedef typename CudaArrayTraits<Derived>::Scalar Scalar;

  /// size type of the array
  typedef LIBCUA_DEFAULT_SIZE_TYPE SizeType;

  /// index type of the array
  typedef LIBCUA_DEFAULT_INDEX_TYPE IndexType;

  /// default block dimensions for general operations
  static const dim3 kBlockDim;

  /// operate on blocks of this width for, e.g., FillRandom
  static const SizeType kTileSize;

  /// number of rows to iterate over per thread for, e.g., FillRandom
  static const SizeType kBlockRows;

  //----------------------------------------------------------------------------
  // constructors and derived()

  /**
   * Constructor.
   * @param width number of columns in the array, assuming a row-major array
   * @param height number of rows in the array, assuming a row-major array
   * @param block_dim default block size for CUDA kernel calls involving this
   *   object, i.e., the values for blockDim.x/y/z; note that the default grid
   *   dimension is computed automatically based on the array size
   * @param stream CUDA stream for this array object
   */
  CudaArray3DBase(SizeType width, SizeType height, SizeType depth,
                  const dim3 block_dim = CudaArray3DBase<Derived>::kBlockDim,
                  const cudaStream_t stream = 0);  // default stream

  /**
   * Base-level copy constructor.
   * @param other array from which to copy array properties such as width and
   *   height
   */
  __host__ __device__ CudaArray3DBase(const CudaArray3DBase<Derived> &other)
      : width_(other.width_),
        height_(other.height_),
        depth_(other.depth_),
        block_dim_(other.block_dim_),
        grid_dim_(other.grid_dim_),
        stream_(other.stream_) {}

  /**
   * @returns a reference to the object cast to its derived class type
   */
  __host__ __device__ Derived &derived() {
    return *reinterpret_cast<Derived *>(this);
  }

  /**
   * @returns a reference to the object cast to its derived class type
   */
  __host__ __device__ const Derived &derived() const {
    return *reinterpret_cast<const Derived *>(this);
  }

  /**
   * Base-level assignment operator; merely copies size, thread dim, and stream
   * parameters. You will probably want to implement your own version of = and
   * call this one internally.
   * @param other the reference array
   * @return *this
   */
  CudaArray3DBase<Derived> &operator=(const CudaArray3DBase<Derived> &other);

  //----------------------------------------------------------------------------
  // general array operations that create a new object

  /**
   * @return a new copy of the current array.
   */
  ENABLE_IF_MUTABLE
  Derived Copy() const {
    Derived result = derived().EmptyCopy();
    Copy(result);
    return result;
  }

  //----------------------------------------------------------------------------
  // general array options that write to an existing object

  /**
   * Copy the current array to another array.
   * @ param other output array
   * @return other
   */
  template <typename OtherDerived,
            typename CudaArrayTraits<OtherDerived>::Mutable is_mutable>
  OtherDerived &CopyTo(OtherDerived &other) const;

  /**
   * Fill the array with a constant value.
   * @param value every element in the array is set to value
   */
  ENABLE_IF_MUTABLE
  inline void Fill(const Scalar value) {
    kernel::CudaArray3DBaseFill<<<grid_dim_, block_dim_, 0, stream_>>>(
        derived(), value);
  }

  /**
   * Fill the array with random values using the given random function.
   * @param rand_state should have one element per block of this array
   * @param func random function such as curand_normal with signature
   *   `T func(curandState_t *)`
   */
  template <typename CurandStateArrayType, typename RandomFunction,
            class C = CudaArrayTraits<Derived>,
            typename C::Mutable is_mutable = true>
  void FillRandom(CurandStateArrayType rand_state, RandomFunction func);

  //----------------------------------------------------------------------------
  // getters/setters

  __host__ __device__ inline SizeType Width() const { return width_; }
  __host__ __device__ inline SizeType Height() const { return height_; }
  __host__ __device__ inline SizeType Depth() const { return depth_; }
  __host__ __device__ inline SizeType Size() const {
    return width_ * height_ * depth_;
  }

  __host__ __device__ inline dim3 BlockDim() const { return block_dim_; }
  __host__ __device__ inline dim3 GridDim() const { return grid_dim_; }

  inline void SetBlockDim(const dim3 block_dim) {
    block_dim_ = block_dim;
    grid_dim_ = dim3((width_ + block_dim.x - 1) / block_dim_.x,
                     (height_ + block_dim.y - 1) / block_dim_.y,
                     (depth_ + block_dim.z - 1) / block_dim_.z);
  }

  inline cudaStream_t Stream() const { return stream_; }
  inline void SetStream(const cudaStream_t stream) { stream_ = stream; }

  //----------------------------------------------------------------------------
  // general array operations

  /**
   * Apply a general element-wise operation to the array. Here's a simple
   * example that uses a device-level C++11 lambda function to store the sum of
   * two arrays `arr1` and `arr2` into the output array:
   *
   *      // Array3DType arr1, arr2
   *      // Array3DType out
   *      out.ApplyOp([arr1, arr2] __device__(unsigned int x, unsigned int y,
   *                                          unsigned int z) {
   *        return arr1.get(x, y, z) + arr2.get(x, y, z);  // => out(x, y, z)
   *      });
   *
   * @param op `__device__` function mapping `(x,y,z) -> CudaArrayClass::Scalar`
   * @param shared_mem_bytes if `op()` uses shared memory, the size of the
   *   shared memory space required
   */
  template <class Function, class C = CudaArrayTraits<Derived>,
            typename C::Mutable is_mutable = true>
  void ApplyOp(Function op, const unsigned int shared_mem_bytes = 0) {
    kernel::CudaArray3DBaseApplyOp<<<grid_dim_, block_dim_, shared_mem_bytes,
                                     stream_>>>(derived(), op);
  }

  /**
   * Element-wise addition.
   * @param value value to add to each array element
   */
  ENABLE_IF_MUTABLE
  inline void operator+=(const Scalar value) {
    Derived &tmp = derived();
    kernel::CudaArray3DBaseApplyOp<<<grid_dim_, block_dim_, 0, stream_>>>(
        tmp, [tmp, value] __device__(IndexType x, IndexType y, IndexType z) {
          return tmp.get(x, y, z) + value;
        });
  }

  /**
   * Element-wise subtraction.
   * @param value value to subtract from each array element
   */
  ENABLE_IF_MUTABLE
  inline void operator-=(const Scalar value) {
    Derived &tmp = derived();
    kernel::CudaArray3DBaseApplyOp<<<grid_dim_, block_dim_, 0, stream_>>>(
        tmp, [tmp, value] __device__(IndexType x, IndexType y, IndexType z) {
          return tmp.get(x, y, z) - value;
        });
  }

  /**
   * Element-wise multiplication.
   * @param value value by which to multiply each array element
   */
  ENABLE_IF_MUTABLE
  inline void operator*=(const Scalar value) {
    Derived &tmp = derived();
    kernel::CudaArray3DBaseApplyOp<<<grid_dim_, block_dim_, 0, stream_>>>(
        tmp, [tmp, value] __device__(IndexType x, IndexType y, IndexType z) {
          return tmp.get(x, y, z) * value;
        });
  }

  /**
   * Element-wise division.
   * @param value value by which to divide each array element.
   */
  ENABLE_IF_MUTABLE
  inline void operator/=(const Scalar value) {
    Derived &tmp = derived();
    kernel::CudaArray3DBaseApplyOp<<<grid_dim_, block_dim_, 0, stream_>>>(
        tmp, [tmp, value] __device__(IndexType x, IndexType y, IndexType z) {
          return tmp.get(x, y, z) / value;
        });
  }

  //----------------------------------------------------------------------------
  // protected class methods and fields

 protected:
  SizeType width_, height_, depth_;

  dim3 block_dim_, grid_dim_;  // for calling kernels

  cudaStream_t stream_;  // the stream on the GPU in which the class kernels run
};

//------------------------------------------------------------------------------
//
// static member initialization
//
//------------------------------------------------------------------------------

template <typename Derived>
const dim3 CudaArray3DBase<Derived>::kBlockDim = dim3(32, 8, 4);

template <typename Derived>
const typename CudaArray3DBase<Derived>::SizeType
    CudaArray3DBase<Derived>::kTileSize = 32;

template <typename Derived>
const typename CudaArray3DBase<Derived>::SizeType
    CudaArray3DBase<Derived>::kBlockRows = 4;

//------------------------------------------------------------------------------
//
// public method implementations
//
//------------------------------------------------------------------------------

template <typename Derived>
CudaArray3DBase<Derived>::CudaArray3DBase<Derived>(const SizeType width,
                                                   const SizeType height,
                                                   const SizeType depth,
                                                   const dim3 block_dim,
                                                   const cudaStream_t stream)
    : width_(width), height_(height), depth_(depth), stream_(stream) {
  SetBlockDim(block_dim);
}

//------------------------------------------------------------------------------

template <typename Derived>
inline CudaArray3DBase<Derived> &CudaArray3DBase<Derived>::operator=(
    const CudaArray3DBase<Derived> &other) {
  if (this == &other) {
    return *this;
  }

  width_ = other.width_;
  height_ = other.height_;
  depth_ = other.depth_;

  block_dim_ = other.block_dim_;
  grid_dim_ = other.grid_dim_;
  stream_ = other.stream_;

  return *this;
}

//------------------------------------------------------------------------------

template <typename Derived>
template <typename OtherDerived,
          typename CudaArrayTraits<OtherDerived>::Mutable is_mutable>
inline OtherDerived &CudaArray3DBase<Derived>::CopyTo(
    OtherDerived &other) const {
  if (this != &other) {
    if (width_ != other.width_ || height_ != other.height_ ||
        depth_ != other.depth_) {
      other = derived().EmptyCopy();
    }

    kernel::CudaArray3DBaseCopyTo<<<grid_dim_, block_dim_>>>(derived(), other);
  }

  return other;
}

//------------------------------------------------------------------------------

template <typename Derived>
template <typename CurandStateArrayType, typename RandomFunction, class C,
          typename C::Mutable is_mutable>
inline void CudaArray3DBase<Derived>::FillRandom(
    CurandStateArrayType rand_state, RandomFunction func) {
  const dim3 block_dim(kTileSize, kBlockRows, kBlockRows);
  const dim3 grid_dim((width_ + kTileSize - 1) / kTileSize,
                      (height_ + kTileSize - 1) / kTileSize,
                      (depth_ + kTileSize - 1) / kTileSize);

  kernel::CudaArray3DBaseFillRandom<<<grid_dim, block_dim, 0, stream_>>>(
      rand_state, derived(), func);
}

//------------------------------------------------------------------------------

#undef ENABLE_IF_MUTABLE
#undef ENABLE_IF_MUTABLE_IMPL

}  // namespace cua

#endif  // LIBCUA_CUDA_ARRAY3D_BASE_H_

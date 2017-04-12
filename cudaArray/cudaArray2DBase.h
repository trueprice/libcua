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

#ifndef CUDAARRAY2DBASE_H_
#define CUDAARRAY2DBASE_H_

#include "cudaArray2DBase_kernels.h"

#include <memory>  // for shared_ptr

#include <curand.h>
#include <curand_kernel.h>

namespace cua {

template <typename Derived>
struct CudaArrayTraits;  // forward declaration

/**
 * @class CudaArray2DBase
 * @brief Base class for all 2D CudaArray-type objects.
 *
 * This class includes implementations for Copy, Transpose, Flip, etc.
 * All derived classes need to define the following members:
 *
 * 1. copy constructor on host *and* device; use `#ifndef __CUDA_ARCH__` to
 *    perform host-specific instructions  
 *    - `__host__ __device__ Derived(const Derived &other);`
 * 2. EmptyCopy(): to create a new array of the same size
 *    - `Derived EmptyCopy() const;`
 * 3. EmptyFlippedCopy(): create a new array with flipped width/height
 *    - `Derived EmptyFlippedCopy() const;`
 * 4. set(): write to array position (optional for readonly subclasses)
 *    - `__device__
 *       inline void set(const size_t x, const size_t y, Scalar value);`
 * 5. get(): read from array position
 *    - `__device__ inline Scalar get(const size_t x, const size_t y) const;`
 * 6. operator=(): suggested to have this for getting data from the CPU
 *   - `Derived &operator=(const Scalar *host_array);`
 * 7. CopyTo(): suggested to have this for getting data to the CPU
 *    - `void CopyTo(Scalar *host_array) const;`
 *
 * Also, any derived class will need to separately declare a CudaArrayTraits
 * struct instantiation with a "Scalar" member, e.g.,
 *
 *     template <typename T>
 *     struct CudaArrayTraits<Derived<T>> {
 *       typedef T Scalar;
 *     };
 */
template <typename Derived>
class CudaArray2DBase {
 public:
  //----------------------------------------------------------------------------
  // static class elements and typedefs
  
  /// datatype of the array
  typedef typename CudaArrayTraits<Derived>::Scalar Scalar;

  /// default block dimensions for general operations
  static const dim3 BLOCK_DIM;

  /// operate on blocks of this width for, e.g., transpose
  static const size_t TILE_SIZE;

  /// number of rows to iterate over per thread for, e.g., transpose
  static const size_t BLOCK_ROWS;

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
  CudaArray2DBase(const size_t width, const size_t height,
                  const dim3 block_dim = CudaArray2DBase<Derived>::BLOCK_DIM,
                  const cudaStream_t stream = 0);  // default stream

  /**
   * Base-level copy constructor.
   * @param other array from which to copy array properties such as width and
   *   height
   */
  __host__ __device__ CudaArray2DBase(const CudaArray2DBase<Derived> &other)
      : width_(other.width_),
        height_(other.height_),
        block_dim_(other.block_dim_),
        grid_dim_(other.grid_dim_),
        stream_(other.stream_) {}

  /**
   * @returns a reference to the object cast to its dervied class type
   */
  __host__ __device__ Derived &derived() {
    return *reinterpret_cast<Derived *>(this);
  }

  /**
   * @returns a reference to the object cast to its dervied class type
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
  CudaArray2DBase<Derived> &operator=(const CudaArray2DBase<Derived> &other);

  //----------------------------------------------------------------------------
  // general array operations that create a new object

  /**
   * @return a new copy of the current array.
   */
  inline Derived Copy() const {
    Derived result = derived().EmptyCopy();
    Copy(result);
    return result;
  }

  /**
   * @return a new copy of the current array, flipped along the horizontal axis.
   */
  inline Derived FlipLR() const {
    Derived result = derived().EmptyCopy();
    FlipLR(result);
    return result;
  }

  /**
   * @return a new copy of the current array, flipped along the vertical axis.
   */
  inline Derived FlipUD() const {
    Derived result = derived().EmptyCopy();
    FlipUD(result);
    return result;
  }

  /**
   * @return a new copy of the current array, rotated 180 degrees.
   */
  inline Derived Rot180() const {
    Derived result = derived().EmptyCopy();
    Rot180(result);
    return result;
  }

  /**
   * @return a new copy of the current array, rotated 90 degrees counterclockwise.
   */
  inline Derived Rot90_CCW() const {
    Derived result = derived().EmptyFlippedCopy();
    Rot90_CCW(result);
    return result;
  }

  /**
   * @return a new copy of the current array, rotated 90 degrees clockwise.
   */
  inline Derived Rot90_CW() const {
    Derived result = derived().EmptyFlippedCopy();
    Rot90_CW(result);
    return result;
  }

  /**
   * @return a new transposed copy of the current array.
   */
  inline Derived Transpose() const {
    Derived result = derived().EmptyFlippedCopy();
    Transpose(result);
    return result;
  }

  //----------------------------------------------------------------------------
  // general array options that write to an existing object

  /**
   * Copy the current array to another array.
   * @ param other output array
   * @return other
   */
  template <typename OtherDerived>  // allow copies to other scalar types
  OtherDerived &Copy(OtherDerived &other) const;

  /**
   * Flip the current array left-right and store in another array.
   * @param other output array
   * @return other
   */
  Derived &FlipLR(Derived &other) const;

  /**
   * Flip the current array up-down and store in another array.
   * @param other output array
   * @return other
   */
  Derived &FlipUD(Derived &other) const;

  /**
   * Rotate the current array 180 degrees and store in another array.
   * @param other output array
   * @return other
   */
  Derived &Rot180(Derived &other) const;

  /**
   * Rotate the current array 90 degrees counterclockwise and another array.
   * @param other output array
   * @return other
   */
  Derived &Rot90_CCW(Derived &other) const;

  /**
   * Rotate the current array 90 degrees clockwise and store in another array.
   * @param other output array
   * @return other
   */
  Derived &Rot90_CW(Derived &other) const;

  /**
   * Transpose the current array and store in another array.
   * @param other output array
   * @return other
   */
  Derived &Transpose(Derived &other) const;

  /**
   * Fill the array with a constant value.
   * @param value every element in the array is set to value
   */
  inline void Fill(const Scalar value) {
    CudaArray2DBase_fill_kernel<<<grid_dim_, block_dim_, 0, stream_>>>
        (derived(), value);
  }

  /**
   * Fill the array with uniform (quasi-)random values in the range (0,1].
   * TODO (True): the interface for this will likely change in the future.
   */
  template <typename curandStateArrayClass>
  inline void FillRandom(curandStateArrayClass rand_state) {
    FillRandom(rand_state, [] __device__(curandState_t * state) {
      return (Scalar)curand_uniform(state);
    });
  }

  /**
   * Fill the array with random values using the given random function.
   * TODO (True): the interface for this will likely change in the future.
   * curandStateArrayType: intended to be CudaRandomStateArray
   * @param func random function such as curand_normal with signature
   *   `T func(curandState_t *)`
   */
  template <typename curandStateArrayClass, typename RandomFunction>
  void FillRandom(curandStateArrayClass rand_state, RandomFunction func);

  //----------------------------------------------------------------------------
  // getters/setters

  __host__ __device__ inline size_t get_width() const { return width_; }
  __host__ __device__ inline size_t get_height() const { return height_; }

  __host__ __device__ inline dim3 get_block_dim() const { return block_dim_; }
  __host__ __device__ inline dim3 get_grid_dim() const { return grid_dim_; }

  inline void set_block_dim(const dim3 block_dim) {
    block_dim_ = block_dim;
    grid_dim_ = dim3((int)std::ceil(float(width_) / block_dim_.x),
                     (int)std::ceil(float(height_) / block_dim_.y));
  }

  inline cudaStream_t get_stream() const { return stream_; }
  inline void set_stream(const cudaStream_t stream) { stream_ = stream; }

  /**
   * Host-level function for setting the value of a single array element.
   * @param x first coordinate, i.e., the column index in a row-major array
   * @param y second coordinate, i.e., the row index in a row-major array
   * @param v the new value to assign to array(x, y)
   */
  inline void SetValue(const size_t x, const size_t y, const Scalar value) {
    if (x >= width_ || y >= height_) {
      throw "Error: CudaArray2DBase Address out of bounds in SetValue().";
    }

    CudaArray2DBase_set_kernel<<<1, 1, 0, stream_>>>(derived(), value, x, y);
  }

  /**
   * Host-level function for getting the value of a single array element.
   * @param x first coordinate, i.e., the column index in a row-major array
   * @param y second coordinate, i.e., the row index in a row-major array
   * @return the value at array(x, y)
   */
  inline Scalar GetValue(const size_t x, const size_t y) const {
    if (x >= width_ || y >= height_) {
      throw "Error: CudaArray2DBase Address out of bounds in GetValue().";
    }

    Scalar value, *dev_value;
    cudaMalloc(&dev_value, sizeof(Scalar));
    CudaArray2DBase_get_kernel<<<1, 1, 0, stream_>>>
        (derived(), dev_value, x, y);
    cudaMemcpy(&value, dev_value, sizeof(Scalar), cudaMemcpyDeviceToHost);
    cudaFree(dev_value);
    return value;
  }

  //----------------------------------------------------------------------------
  // general array operations

  /**
   * Apply a general element-wise operation to the array. Here's a simple
   * example that uses a device-level C++11 lambda function to store the sum of
   * two arrays `arr1` and `arr2` into the output array:
   *
   *      // Array2DType arr1, arr2
   *      // Array2DType out
   *      out.apply_op([arr1, arr2] __device__(const size_t x, const size_t y) {
   *        return arr1.get(x, y) + arr2.get(x, y);  // stored in out(x, y)
   *      });
   * 
   * @param op `__device__` function mapping `(x,y) -> CudaArrayClass::Scalar`
   * @param shared_mem_bytes if `op()` uses shared memory, the size of the
   *   shared memory space required
   */
  template <class Function>
  inline void apply_op(Function op, const size_t shared_mem_bytes = 0) {
    CudaArray2DBase_apply_op_kernel
           <<<grid_dim_, block_dim_, shared_mem_bytes, stream_>>>
        (derived(), op);
  }

  /**
   * Element-wise addition.
   * @param value value to add to each array element
   */
  inline void operator+=(const Scalar value) {
    Derived &tmp = derived();
    CudaArray2DBase_apply_op_kernel<<<grid_dim_, block_dim_, 0, stream_>>>
        (tmp, [tmp, value] __device__(const size_t x, const size_t y) {
          return tmp.get(x, y) + value;
        });
  }

  /**
   * Element-wise subtraction.
   * @param value value to subtract from each array element
   */
  inline void operator-=(const Scalar value) { operator+=(-value); }

  /**
   * Element-wise multiplication.
   * @param value value by which to multiply each array element
   */
  inline void operator*=(const Scalar value) {
    Derived &tmp = derived();
    CudaArray2DBase_apply_op_kernel<<<grid_dim_, block_dim_, 0, stream_>>>
        (tmp, [tmp, value] __device__(const size_t x, const size_t y) {
          return tmp.get(x, y) * value;
        });
  }

  /**
   * Element-wise division.
   * @param value value by which to divide each array element.
   */
  inline void operator/=(const Scalar value) { operator*=(Scalar(1.) / value); }

  //----------------------------------------------------------------------------
  // protected class methods and fields

 protected:
  size_t width_, height_;

  dim3 block_dim_, grid_dim_;  // for calling kernels

  cudaStream_t stream_;  // the stream on the GPU in which the class kernels run
};

//------------------------------------------------------------------------------
//
// static member initialization
//
//------------------------------------------------------------------------------

template <typename Derived>
const dim3 CudaArray2DBase<Derived>::BLOCK_DIM = dim3(32, 32);

template <typename Derived>
const size_t CudaArray2DBase<Derived>::TILE_SIZE = 32;

template <typename Derived>
const size_t CudaArray2DBase<Derived>::BLOCK_ROWS = 4;

//------------------------------------------------------------------------------
//
// public method implementations
//
//------------------------------------------------------------------------------

template <typename Derived>
CudaArray2DBase<Derived>::CudaArray2DBase<Derived>(const size_t width,
                                                   const size_t height,
                                                   const dim3 block_dim,
                                                   const cudaStream_t stream)
    : width_(width), height_(height), stream_(stream) {
  set_block_dim(block_dim);
}

//------------------------------------------------------------------------------

template <typename Derived>
CudaArray2DBase<Derived> &CudaArray2DBase<Derived>::operator=(
    const CudaArray2DBase<Derived> &other) {
  if (this == &other) {
    return *this;
  }

  width_ = other.width_;
  height_ = other.height_;

  block_dim_ = other.block_dim_;
  grid_dim_ = other.grid_dim_;
  stream_ = other.stream_;

  return *this;
}

//------------------------------------------------------------------------------

template <typename Derived>
template <typename OtherDerived>
OtherDerived &CudaArray2DBase<Derived>::Copy(OtherDerived &other) const {
  if (this != &other) {
    if (width_ != other.width_ || height_ != other.height_) {
      other = derived().EmptyCopy();
    }

    CudaArray2DBase_copy_kernel<<<grid_dim_, block_dim_>>>(derived(), other);
  }

  return other;
}

//------------------------------------------------------------------------------

template <typename Derived>
template <typename curandStateArrayClass, typename RandomFunction>
void CudaArray2DBase<Derived>::FillRandom(curandStateArrayClass rand_state,
                                          RandomFunction func) {
  const dim3 block_dim = dim3(CudaArray2DBase<Derived>::TILE_SIZE,
                              CudaArray2DBase<Derived>::BLOCK_ROWS);
  const dim3 grid_dim = dim3(
      (int)std::ceil(float(width_) / CudaArray2DBase<Derived>::TILE_SIZE),
      (int)std::ceil(float(height_) / CudaArray2DBase<Derived>::TILE_SIZE));

  CudaArray2DBase_fillRandom_kernel<<<grid_dim, block_dim, 0, stream_>>>
      (rand_state, derived(), func);
}

//------------------------------------------------------------------------------
//
// general array operations
//
//------------------------------------------------------------------------------

template <typename Derived>
Derived &CudaArray2DBase<Derived>::FlipLR(Derived &other) const {
  const dim3 block_dim = dim3(CudaArray2DBase<Derived>::TILE_SIZE,
                              CudaArray2DBase<Derived>::BLOCK_ROWS);
  const dim3 grid_dim = dim3(
      (int)std::ceil(float(width_) / CudaArray2DBase<Derived>::TILE_SIZE),
      (int)std::ceil(float(height_) / CudaArray2DBase<Derived>::TILE_SIZE));

  CudaArray2DBase_fliplr_kernel<<<grid_dim, block_dim, 0, stream_>>>
      (derived(), other);

  return other;
}

//------------------------------------------------------------------------------

template <typename Derived>
Derived &CudaArray2DBase<Derived>::FlipUD(Derived &other) const {
  const dim3 block_dim = dim3(CudaArray2DBase<Derived>::TILE_SIZE,
                              CudaArray2DBase<Derived>::BLOCK_ROWS);
  const dim3 grid_dim = dim3(
      (int)std::ceil(float(width_) / CudaArray2DBase<Derived>::TILE_SIZE),
      (int)std::ceil(float(height_) / CudaArray2DBase<Derived>::TILE_SIZE));

  CudaArray2DBase_flipud_kernel<<<grid_dim, block_dim, 0, stream_>>>
      (derived(), other);

  return other;
}

//------------------------------------------------------------------------------

template <typename Derived>
Derived &CudaArray2DBase<Derived>::Rot180(Derived &other) const {
  // compute down columns; the width should be equal to the width of a CUDA
  // thread warp; the number of rows that each block covers is equal to
  // CudaArray2DBase<Derived>::BLOCK_ROWS
  const dim3 block_dim = dim3(CudaArray2DBase<Derived>::TILE_SIZE,
                              CudaArray2DBase<Derived>::BLOCK_ROWS);
  const dim3 grid_dim = dim3(
      (int)std::ceil(float(width_) / CudaArray2DBase<Derived>::TILE_SIZE),
      (int)std::ceil(float(height_) / CudaArray2DBase<Derived>::TILE_SIZE));

  CudaArray2DBase_rot180_kernel<<<grid_dim, block_dim, 0, stream_>>>
      (derived(), other);

  return other;
}

//------------------------------------------------------------------------------

template <typename Derived>
Derived &CudaArray2DBase<Derived>::Rot90_CCW(Derived &other) const {
  // compute down columns; the width should be equal to the width of a CUDA
  // thread warp; the number of rows that each block covers is equal to
  // CudaArray2DBase<Derived>::BLOCK_ROWS
  const dim3 block_dim = dim3(CudaArray2DBase<Derived>::TILE_SIZE,
                              CudaArray2DBase<Derived>::BLOCK_ROWS);
  const dim3 grid_dim = dim3(
      (int)std::ceil(float(width_) / CudaArray2DBase<Derived>::TILE_SIZE),
      (int)std::ceil(float(height_) / CudaArray2DBase<Derived>::TILE_SIZE));
  const size_t shm_size = CudaArray2DBase<Derived>::TILE_SIZE *
                          (CudaArray2DBase<Derived>::TILE_SIZE) *
                          sizeof(Scalar);

  CudaArray2DBase_rot90_CCW_kernel<<<grid_dim, block_dim, shm_size, stream_>>>
      (derived(), other);

  return other;
}

//------------------------------------------------------------------------------

template <typename Derived>
Derived &CudaArray2DBase<Derived>::Rot90_CW(Derived &other) const {
  // compute down columns; the width should be equal to the width of a CUDA
  // thread warp; the number of rows that each block covers is equal to
  // CudaArray2DBase<Derived>::BLOCK_ROWS
  const dim3 block_dim = dim3(CudaArray2DBase<Derived>::TILE_SIZE,
                              CudaArray2DBase<Derived>::BLOCK_ROWS);
  const dim3 grid_dim = dim3(
      (int)std::ceil(float(width_) / CudaArray2DBase<Derived>::TILE_SIZE),
      (int)std::ceil(float(height_) / CudaArray2DBase<Derived>::TILE_SIZE));
  const size_t shm_size = CudaArray2DBase<Derived>::TILE_SIZE *
                          (CudaArray2DBase<Derived>::TILE_SIZE) *
                          sizeof(Scalar);

  CudaArray2DBase_rot90_CW_kernel<<<grid_dim, block_dim, shm_size, stream_>>>
      (derived(), other);

  return other;
}

//------------------------------------------------------------------------------

template <typename Derived>
Derived &CudaArray2DBase<Derived>::Transpose(Derived &other) const {
  // compute down columns; the width should be equal to the width of a CUDA
  // thread warp; the number of rows that each block covers is equal to
  // CudaArray2DBase<Derived>::BLOCK_ROWS
  const dim3 block_dim = dim3(CudaArray2DBase<Derived>::TILE_SIZE,
                              CudaArray2DBase<Derived>::BLOCK_ROWS);
  const dim3 grid_dim = dim3(
      (int)std::ceil(float(width_) / CudaArray2DBase<Derived>::TILE_SIZE),
      (int)std::ceil(float(height_) / CudaArray2DBase<Derived>::TILE_SIZE));
  const size_t shm_size = CudaArray2DBase<Derived>::TILE_SIZE *
                          (CudaArray2DBase<Derived>::TILE_SIZE) *
                          sizeof(Scalar);

  CudaArray2DBase_transpose_kernel<<<grid_dim, block_dim, shm_size, stream_>>>
      (derived(), other);

  return other;
}

}  // namespace cua

#endif  // CUDAARRAY2D_H_

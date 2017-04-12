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

#ifndef CUDAARRAY2D_KERNELS_H_
#define CUDAARRAY2D_KERNELS_H_

#include <curand.h>
#include <curand_kernel.h>

namespace cua {

//------------------------------------------------------------------------------
//
// class-specific kernel functions for the CudaArray2DBase class
// TODO (True): provide proper documentation for these kernel functions
//
//------------------------------------------------------------------------------

//
// copy values of one surface to another, possibly with different datatypes
//
template <typename SrcCls, typename DstCls>
__global__ void CudaArray2DBase_copy_kernel(const SrcCls src, DstCls dst) {
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < src.get_width() && y < src.get_height()) {
    dst.set(x, y, (typename DstCls::Scalar)src.get(x, y));
  }
}

//------------------------------------------------------------------------------

//
// fill an array with a value
//
template <typename CudaArrayClass, typename T>
__global__ void CudaArray2DBase_fill_kernel(CudaArrayClass mat, const T value) {
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < mat.get_width() && y < mat.get_height()) {
    mat.set(x, y, value);
  }
}

//------------------------------------------------------------------------------

//
// fillRandom: fill CudaArray2DBase with random values
//
template <typename CudaRandomStateArrayClass, typename CudaArrayClass,
          typename RandomFunction>
__global__ void CudaArray2DBase_fillRandom_kernel(
    CudaRandomStateArrayClass rand_state, CudaArrayClass mat,
    RandomFunction func) {
  const size_t x = blockIdx.x * CudaArrayClass::TILE_SIZE + threadIdx.x;

  if (x < mat.get_width()) {
    const size_t y = blockIdx.y * CudaArrayClass::TILE_SIZE + threadIdx.y;

    // each thread iterates down columns, so we need to offset the random state
    // for this thread by CudaArray2DBase::BLOCK_ROWS, in addition to any
    // within-block offset in y

    curandState_t state = rand_state.get(blockIdx.x, blockIdx.y);
    skipahead((threadIdx.y * CudaArrayClass::TILE_SIZE + threadIdx.x) *
                  CudaArrayClass::BLOCK_ROWS,
              &state);

    const size_t max_y = min(y + CudaArrayClass::TILE_SIZE, mat.get_height());

    for (size_t j = y; j < max_y; j += CudaArrayClass::BLOCK_ROWS) {
      mat.set(x, j, func(&state));
    }

    // update the global random state
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
      rand_state.set(blockIdx.x, blockIdx.y, state);
    }
  }
}

//------------------------------------------------------------------------------

//
// set a single value in a CudaArray2DBase object
// NOTE: we assume array bounds have been checked prior to calling this kernel
//
template <typename CudaArrayClass, typename T>
__global__ void CudaArray2DBase_set_kernel(CudaArrayClass mat, const T value,
                                           const int x, const int y) {
  mat.set(x, y, value);
}

//------------------------------------------------------------------------------

//
// get a single value in a CudaArray2DBase object
// NOTE: we assume array bounds have been checked prior to calling this kernel
//
template <typename CudaArrayClass, typename T>
__global__ void CudaArray2DBase_get_kernel(CudaArrayClass mat, const T *value,
                                           const int x, const int y) {
  *value = mat.get(x, y);
}

//------------------------------------------------------------------------------

//
// copy the array to memory allocated for its transpose
//
template <typename SrcCls, typename DstCls>
__global__ void CudaArray2DBase_transpose_kernel(const SrcCls src, DstCls dst) {
  __shared__ typename SrcCls::Scalar tile[SrcCls::TILE_SIZE][SrcCls::TILE_SIZE];

  size_t x = blockIdx.x * SrcCls::TILE_SIZE + threadIdx.x;
  size_t y = blockIdx.y * SrcCls::TILE_SIZE + threadIdx.y;

  if (x < src.get_width()) {
    const size_t max_y = min(y + SrcCls::TILE_SIZE, src.get_height());
    for (size_t j = 0; (y + j) < max_y; j += SrcCls::BLOCK_ROWS) {
      tile[threadIdx.y + j][threadIdx.x] = src.get(x, y + j);
    }
  }

  __syncthreads();

  // note we still iterate down columns to ensure coalesced memory access
  x = blockIdx.y * SrcCls::TILE_SIZE + threadIdx.x;
  y = blockIdx.x * SrcCls::TILE_SIZE + threadIdx.y;

  if (x < dst.get_width()) {
    const size_t max_y = min(y + SrcCls::TILE_SIZE, dst.get_height());
    for (size_t j = 0; (y + j) < max_y; j += SrcCls::BLOCK_ROWS) {
      dst.set(x, y + j, tile[threadIdx.x][threadIdx.y + j]);
    }
  }
}

//------------------------------------------------------------------------------
//
// array operations
//
//------------------------------------------------------------------------------

//
// Implementation note:
// flipud, fliplr, rot180 all do not require shared memory, so
// they all have the same structure, aside from the dst(x',y') = src(x,y) line
//

template <typename SrcCls, typename DstCls>
__global__ void CudaArray2DBase_fliplr_kernel(const SrcCls src, DstCls dst) {
  const size_t x = blockIdx.x * SrcCls::TILE_SIZE + threadIdx.x;

  const size_t w = src.get_width();

  if (x < w) {
    const size_t y = blockIdx.y * SrcCls::TILE_SIZE + threadIdx.y;
    const size_t h = src.get_height();
    const size_t max_y = min(y + SrcCls::TILE_SIZE, h);

    for (size_t j = y; j < max_y; j += SrcCls::BLOCK_ROWS) {
      dst.set(w - x - 1, j, src.get(x, j));
    }
  }
}

//------------------------------------------------------------------------------

template <typename SrcCls, typename DstCls>
__global__ void CudaArray2DBase_flipud_kernel(const SrcCls src, DstCls dst) {
  const size_t x = blockIdx.x * SrcCls::TILE_SIZE + threadIdx.x;

  const size_t w = src.get_width();

  if (x < w) {
    const size_t y = blockIdx.y * SrcCls::TILE_SIZE + threadIdx.y;
    const size_t h = src.get_height();
    const size_t max_y = min(y + SrcCls::TILE_SIZE, h);

    for (size_t j = y; j < max_y; j += SrcCls::BLOCK_ROWS) {
      dst.set(x, h - j - 1, src.get(x, j));
    }
  }
}

//------------------------------------------------------------------------------

template <typename SrcCls, typename DstCls>
__global__ void CudaArray2DBase_rot180_kernel(const SrcCls src, DstCls dst) {
  const size_t x = blockIdx.x * SrcCls::TILE_SIZE + threadIdx.x;

  const size_t w = src.get_width();

  if (x < w) {
    const size_t y = blockIdx.y * SrcCls::TILE_SIZE + threadIdx.y;
    const size_t h = src.get_height();
    const size_t max_y = min(y + SrcCls::TILE_SIZE, h);

    for (size_t j = y; j < max_y; j += SrcCls::BLOCK_ROWS) {
      dst.set(w - x - 1, h - j - 1, src.get(x, j));
    }
  }
}

//------------------------------------------------------------------------------

template <typename SrcCls, typename DstCls>
__global__ void CudaArray2DBase_rot90_CCW_kernel(const SrcCls src, DstCls dst) {
  __shared__ typename SrcCls::Scalar tile[SrcCls::TILE_SIZE][SrcCls::TILE_SIZE];

  size_t x = blockIdx.x * SrcCls::TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * SrcCls::TILE_SIZE + threadIdx.y;

  const size_t w = src.get_width();
  const size_t h = src.get_height();

  if (x < w) {
    const size_t max_y = min(y + SrcCls::TILE_SIZE, h);
    for (size_t j = 0; (y + j) < max_y; j += SrcCls::BLOCK_ROWS) {
      tile[threadIdx.y + j][threadIdx.x] = src.get(x, y + j);
    }
  }

  __syncthreads();

  // note we still iterate down columns to ensure coalesced memory access
  x = blockIdx.y * SrcCls::TILE_SIZE + threadIdx.x;            // L to R
  y = w - 1 - (blockIdx.x * SrcCls::TILE_SIZE + threadIdx.y);  // B to T

  if (x < h) {
    const int min_y = max(y - (int)SrcCls::TILE_SIZE + 1, 0);
    for (int j = 0; (y - j) >= min_y; j += SrcCls::BLOCK_ROWS) {
      dst.set(x, y - j, tile[threadIdx.x][threadIdx.y + j]);
    }
  }
}

//------------------------------------------------------------------------------

template <typename SrcCls, typename DstCls>
__global__ void CudaArray2DBase_rot90_CW_kernel(const SrcCls src, DstCls dst) {
  __shared__ typename SrcCls::Scalar tile[SrcCls::TILE_SIZE][SrcCls::TILE_SIZE];

  int x = blockIdx.x * SrcCls::TILE_SIZE + threadIdx.x;
  size_t y = blockIdx.y * SrcCls::TILE_SIZE + threadIdx.y;

  const size_t w = src.get_width();
  const size_t h = src.get_height();

  if (x < w) {
    const size_t max_y = min(y + SrcCls::TILE_SIZE, h);
    for (size_t j = 0; (y + j) < max_y; j += SrcCls::BLOCK_ROWS) {
      tile[threadIdx.y + j][threadIdx.x] = src.get(x, y + j);
    }
  }

  __syncthreads();

  // note we still iterate down columns to ensure coalesced memory access
  x = h - 1 - (blockIdx.y * SrcCls::TILE_SIZE + threadIdx.x);  // R to L
  y = blockIdx.x * SrcCls::TILE_SIZE + threadIdx.y;            // T to B

  if (x >= 0) {
    const size_t max_y = min(y + SrcCls::TILE_SIZE, w);
    for (size_t j = 0; (y + j) < max_y; j += SrcCls::BLOCK_ROWS) {
      dst.set(x, y + j, tile[threadIdx.x][threadIdx.y + j]);
    }
  }
}

//------------------------------------------------------------------------------

//
// kernel for general element-wise array operations
// op: __device__ function mapping (x,y) -> CudaArrayClass::Scalar
//
template <typename CudaArrayClass, class Function>
__global__ void CudaArray2DBase_apply_op_kernel(CudaArrayClass mat,
                                                Function op) {
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < mat.get_width() && y < mat.get_height()) {
    mat.set(x, y, op(x, y));
  }
}

}  // namespace cua

#endif  // CUDAARRAY2D_KERNELS_H_

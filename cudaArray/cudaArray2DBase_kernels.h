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

#ifndef CUDA_ARRAY2D_BASE_KERNELS_H_
#define CUDA_ARRAY2D_BASE_KERNELS_H_

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

  if (x < src.Width() && y < src.Height()) {
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

  if (x < mat.Width() && y < mat.Height()) {
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

  if (x < mat.Width()) {
    const size_t y = blockIdx.y * CudaArrayClass::TILE_SIZE + threadIdx.y;

    // each thread iterates down columns, so we need to offset the random state
    // for this thread by CudaArray2DBase::BLOCK_ROWS, in addition to any
    // within-block offset in y

    curandState_t state = rand_state.get(blockIdx.x, blockIdx.y);
    skipahead((threadIdx.y * CudaArrayClass::TILE_SIZE + threadIdx.x) *
                  CudaArrayClass::BLOCK_ROWS,
              &state);

    const size_t max_y = min(y + CudaArrayClass::TILE_SIZE, mat.Height());

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

  if (x < src.Width()) {
    const size_t max_y = min(y + SrcCls::TILE_SIZE, src.Height());
    for (size_t j = 0; (y + j) < max_y; j += SrcCls::BLOCK_ROWS) {
      tile[threadIdx.y + j][threadIdx.x] = src.get(x, y + j);
    }
  }

  __syncthreads();

  // note we still iterate down columns to ensure coalesced memory access
  x = blockIdx.y * SrcCls::TILE_SIZE + threadIdx.x;
  y = blockIdx.x * SrcCls::TILE_SIZE + threadIdx.y;

  if (x < dst.Width()) {
    const size_t max_y = min(y + SrcCls::TILE_SIZE, dst.Height());
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

  const size_t w = src.Width();

  if (x < w) {
    const size_t y = blockIdx.y * SrcCls::TILE_SIZE + threadIdx.y;
    const size_t h = src.Height();
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

  const size_t w = src.Width();

  if (x < w) {
    const size_t y = blockIdx.y * SrcCls::TILE_SIZE + threadIdx.y;
    const size_t h = src.Height();
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

  const size_t w = src.Width();

  if (x < w) {
    const size_t y = blockIdx.y * SrcCls::TILE_SIZE + threadIdx.y;
    const size_t h = src.Height();
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

  const size_t w = src.Width();
  const size_t h = src.Height();

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
    const int min_y = max(y - static_cast<int>(SrcCls::TILE_SIZE) + 1, 0);
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

  const size_t w = src.Width();
  const size_t h = src.Height();

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

  if (x < mat.Width() && y < mat.Height()) {
    mat.set(x, y, op(x, y));
  }
}

}  // namespace cua

#endif  // CUDA_ARRAY2D_BASE_KERNELS_H_

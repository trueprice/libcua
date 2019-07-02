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

#ifndef LIBCUA_CUDA_ARRAY2D_BASE_KERNELS_H_
#define LIBCUA_CUDA_ARRAY2D_BASE_KERNELS_H_

#include <curand.h>
#include <curand_kernel.h>

namespace cua {

namespace kernel {

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
__global__ void CudaArray2DBaseCopy(const SrcCls src, DstCls dst) {
  const typename SrcCls::IndexType x = blockIdx.x * blockDim.x + threadIdx.x;
  const typename SrcCls::IndexType y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < src.Width() && y < src.Height()) {
    dst.set(x, y, (typename DstCls::Scalar)src.get(x, y));
  }
}

//------------------------------------------------------------------------------

//
// fill an array with a value
//
template <typename CudaArrayClass, typename T>
__global__ void CudaArray2DBaseFill(CudaArrayClass array, const T value) {
  const typename CudaArrayClass::IndexType x =
      blockIdx.x * blockDim.x + threadIdx.x;
  const typename CudaArrayClass::IndexType y =
      blockIdx.y * blockDim.y + threadIdx.y;

  if (x < array.Width() && y < array.Height()) {
    array.set(x, y, value);
  }
}

//------------------------------------------------------------------------------

//
// fillRandom: fill CudaArray2DBase with random values
//
template <typename CudaRandomStateArrayClass, typename CudaArrayClass,
          typename RandomFunction>
__global__ void CudaArray2DBaseFillRandom(CudaRandomStateArrayClass rand_state,
                                          CudaArrayClass array,
                                          RandomFunction func) {
  const typename CudaArrayClass::IndexType x =
      blockIdx.x * CudaArrayClass::kTileSize + threadIdx.x;
  const typename CudaArrayClass::IndexType y =
      blockIdx.y * CudaArrayClass::kTileSize + threadIdx.y;

  // Each thread processes kBlockRows contiguous rows in y.
  curandState_t state = rand_state.get(blockIdx.x, blockIdx.y);
  skipahead(static_cast<unsigned long long>(
                (threadIdx.y * CudaArrayClass::kTileSize + threadIdx.x) *
                CudaArrayClass::kBlockRows),
            &state);

  for (typename CudaArrayClass::IndexType j = 0; j < CudaArrayClass::kTileSize;
       j += CudaArrayClass::kBlockRows) {
    const auto value = func(&state);
    if (x < array.Width() && y + j < array.Height()) {
      array.set(x, y + j, value);
    }
  }

  // update the global random state
  if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
    rand_state.set(blockIdx.x, blockIdx.y, state);
  }
}

//------------------------------------------------------------------------------

//
// set a single value in a CudaArray2DBase object
//
template <typename CudaArrayClass, typename T>
__global__ void CudaArray2DBaseSet(CudaArrayClass array, const T value,
                                   const int x, const int y) {
  array.set(x, y, value);
}

//------------------------------------------------------------------------------

//
// get a single value in a CudaArray2DBase object
// NOTE: we assume array bounds have been checked prior to calling this kernel
//
template <typename CudaArrayClass, typename T>
__global__ void CudaArray2DBaseGet(CudaArrayClass array, const T *value,
                                   const int x, const int y) {
  *value = array.get(x, y);
}

//------------------------------------------------------------------------------

//
// copy the array to memory allocated for its transpose
//
template <typename SrcCls, typename DstCls>
__global__ void CudaArray2DBaseTranspose(const SrcCls src, DstCls dst) {
  __shared__ typename SrcCls::Scalar tile[SrcCls::kTileSize][SrcCls::kTileSize];

  typename SrcCls::IndexType x = blockIdx.x * SrcCls::kTileSize + threadIdx.x;
  typename SrcCls::IndexType y = blockIdx.y * SrcCls::kTileSize + threadIdx.y;

  if (x < src.Width()) {
    const typename SrcCls::SizeType max_y =
        min(y + SrcCls::kTileSize, src.Height());
    for (typename SrcCls::IndexType j = 0; (y + j) < max_y;
         j += SrcCls::kBlockRows) {
      tile[threadIdx.y + j][threadIdx.x] = src.get(x, y + j);
    }
  }

  __syncthreads();

  // note we still iterate down columns to ensure coalesced memory access
  x = blockIdx.y * SrcCls::kTileSize + threadIdx.x;
  y = blockIdx.x * SrcCls::kTileSize + threadIdx.y;

  if (x < dst.Width()) {
    const typename SrcCls::SizeType max_y =
        min(y + SrcCls::kTileSize, dst.Height());
    for (typename SrcCls::IndexType j = 0; (y + j) < max_y;
         j += SrcCls::kBlockRows) {
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
__global__ void CudaArray2DBaseFlipLR(const SrcCls src, DstCls dst) {
  const typename SrcCls::IndexType x =
      blockIdx.x * SrcCls::kTileSize + threadIdx.x;

  const typename SrcCls::SizeType w = src.Width();

  if (x < w) {
    const typename SrcCls::IndexType y =
        blockIdx.y * SrcCls::kTileSize + threadIdx.y;
    const typename SrcCls::SizeType h = src.Height();
    const typename SrcCls::SizeType max_y = min(y + SrcCls::kTileSize, h);

    for (typename SrcCls::IndexType j = y; j < max_y; j += SrcCls::kBlockRows) {
      dst.set(w - x - 1, j, src.get(x, j));
    }
  }
}

//------------------------------------------------------------------------------

template <typename SrcCls, typename DstCls>
__global__ void CudaArray2DBaseFlipUD(const SrcCls src, DstCls dst) {
  const typename SrcCls::IndexType x =
      blockIdx.x * SrcCls::kTileSize + threadIdx.x;

  const typename SrcCls::SizeType w = src.Width();

  if (x < w) {
    const typename SrcCls::IndexType y =
        blockIdx.y * SrcCls::kTileSize + threadIdx.y;
    const typename SrcCls::SizeType h = src.Height();
    const typename SrcCls::SizeType max_y = min(y + SrcCls::kTileSize, h);

    for (typename SrcCls::IndexType j = y; j < max_y; j += SrcCls::kBlockRows) {
      dst.set(x, h - j - 1, src.get(x, j));
    }
  }
}

//------------------------------------------------------------------------------

template <typename SrcCls, typename DstCls>
__global__ void CudaArray2DBaseRot180(const SrcCls src, DstCls dst) {
  const typename SrcCls::IndexType x =
      blockIdx.x * SrcCls::kTileSize + threadIdx.x;

  const typename SrcCls::SizeType w = src.Width();

  if (x < w) {
    const typename SrcCls::IndexType y =
        blockIdx.y * SrcCls::kTileSize + threadIdx.y;
    const typename SrcCls::SizeType h = src.Height();
    const typename SrcCls::SizeType max_y = min(y + SrcCls::kTileSize, h);

    for (typename SrcCls::IndexType j = y; j < max_y; j += SrcCls::kBlockRows) {
      dst.set(w - x - 1, h - j - 1, src.get(x, j));
    }
  }
}

//------------------------------------------------------------------------------

template <typename SrcCls, typename DstCls>
__global__ void CudaArray2DBaseRot90_CCW(const SrcCls src, DstCls dst) {
  __shared__ typename SrcCls::Scalar tile[SrcCls::kTileSize][SrcCls::kTileSize];

  typename SrcCls::IndexType x = blockIdx.x * SrcCls::kTileSize + threadIdx.x;
  typename SrcCls::IndexType y = blockIdx.y * SrcCls::kTileSize + threadIdx.y;

  const typename SrcCls::SizeType w = src.Width();
  const typename SrcCls::SizeType h = src.Height();

  if (x < w) {
    const typename SrcCls::SizeType max_y = min(y + SrcCls::kTileSize, h);
    for (typename SrcCls::IndexType j = 0; (y + j) < max_y;
         j += SrcCls::kBlockRows) {
      tile[threadIdx.y + j][threadIdx.x] = src.get(x, y + j);
    }
  }

  __syncthreads();

  // note we still iterate down columns to ensure coalesced memory access
  x = blockIdx.y * SrcCls::kTileSize + threadIdx.x;            // L to R
  y = w - 1 - (blockIdx.x * SrcCls::kTileSize + threadIdx.y);  // B to T

  if (x < h) {
    const typename SrcCls::SizeType min_y =
        max(y - static_cast<int>(SrcCls::kTileSize) + 1, 0);
    for (typename SrcCls::IndexType j = 0; (y - j) >= min_y;
         j += SrcCls::kBlockRows) {
      dst.set(x, y - j, tile[threadIdx.x][threadIdx.y + j]);
    }
  }
}

//------------------------------------------------------------------------------

template <typename SrcCls, typename DstCls>
__global__ void CudaArray2DBaseRot90_CW(const SrcCls src, DstCls dst) {
  __shared__ typename SrcCls::Scalar tile[SrcCls::kTileSize][SrcCls::kTileSize];

  typename SrcCls::IndexType x = blockIdx.x * SrcCls::kTileSize + threadIdx.x;
  typename SrcCls::IndexType y = blockIdx.y * SrcCls::kTileSize + threadIdx.y;

  const typename SrcCls::SizeType w = src.Width();
  const typename SrcCls::SizeType h = src.Height();

  if (x < w) {
    const typename SrcCls::SizeType max_y = min(y + SrcCls::kTileSize, h);
    for (typename SrcCls::IndexType j = 0; (y + j) < max_y;
         j += SrcCls::kBlockRows) {
      tile[threadIdx.y + j][threadIdx.x] = src.get(x, y + j);
    }
  }

  __syncthreads();

  // note we still iterate down columns to ensure coalesced memory access
  x = h - 1 - (blockIdx.y * SrcCls::kTileSize + threadIdx.x);  // R to L
  y = blockIdx.x * SrcCls::kTileSize + threadIdx.y;            // T to B

  if (x >= 0) {
    const typename SrcCls::SizeType max_y = min(y + SrcCls::kTileSize, w);
    for (typename SrcCls::IndexType j = 0; (y + j) < max_y;
         j += SrcCls::kBlockRows) {
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
__global__ void CudaArray2DBaseApplyOp(CudaArrayClass array, Function op) {
  const typename CudaArrayClass::IndexType x =
      blockIdx.x * blockDim.x + threadIdx.x;
  const typename CudaArrayClass::IndexType y =
      blockIdx.y * blockDim.y + threadIdx.y;

  if (x < array.Width() && y < array.Height()) {
    array.set(x, y, op(x, y));
  }
}

}  // namespace kernel

}  // namespace cua

#endif  // LIBCUA_CUDA_ARRAY2D_BASE_KERNELS_H_

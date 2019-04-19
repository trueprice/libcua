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

#ifndef CUDA_ARRAY2D_BASE_TEST_H_
#define CUDA_ARRAY2D_BASE_TEST_H_

#include <vector>

#include "util.h"

namespace cua {

namespace test {

//------------------------------------------------------------------------------

// Ops for testing ApplyOp need to be defined outside of the test function
// (this was unfortunately the only way I could get it to compile), so I went
// ahead and moved all the test functionality to this class.
template <typename CudaArrayType>
struct CudaArray2DTestWrapper {
  CudaArray2DTestWrapper(size_t width = 10, size_t height = 10)
      : array(width, height), result(width * height) {}

  //----------------------------------------------------------------------------

  template <typename HostFunction>
  void DownloadAndCheck(const HostFunction& host_function) {
    CUDA_CHECK_ERROR
    array.CopyTo(result.data());
    CUDA_CHECK_ERROR

    for (size_t y = 0; y < array.Height(); ++y) {
      for (size_t x = 0; x < array.Width(); ++x) {
        const size_t i = y * array.Width() + x;
        EXPECT_EQ(result[i], host_function(x, y));
      }
    }
  }

  //----------------------------------------------------------------------------

  void CheckFill(typename CudaArrayType::Scalar value) {
    array.Fill(value);
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceAdd(typename CudaArrayType::Scalar value) {
    array.Fill(0.);
    CUDA_CHECK_ERROR
    array += value;
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
    array += value;
    DownloadAndCheck([=](size_t x, size_t y) { return value + value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceSubtract(typename CudaArrayType::Scalar value) {
    array.Fill(value + value);
    CUDA_CHECK_ERROR
    array -= value;
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
    array -= value;
    DownloadAndCheck([=](size_t x, size_t y) { return 0; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceMultiply(typename CudaArrayType::Scalar value) {
    array.Fill(1.);
    CUDA_CHECK_ERROR
    array *= value;
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
    array *= value;
    DownloadAndCheck([=](size_t x, size_t y) { return value * value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceDivide(typename CudaArrayType::Scalar value) {
    array.Fill(value * value);
    CUDA_CHECK_ERROR
    array /= value;
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
    array /= value;
    DownloadAndCheck([=](size_t x, size_t y) { return 1; });
  }

  //----------------------------------------------------------------------------

  void CheckApplyOpConstant(typename CudaArrayType::Scalar value) {
    array.ApplyOp([=] __device__(size_t x, size_t y) { return value; });
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
  }

  //----------------------------------------------------------------------------

  void CheckApplyOpLinear() {
    // Note the *this capture!
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#star-this-capture
    array.ApplyOp([=, *this] __device__(size_t x, size_t y) {
      return static_cast<typename CudaArrayType::Scalar>(y * array.Width() + x);
    });
    DownloadAndCheck([=](size_t x, size_t y) {
      return static_cast<typename CudaArrayType::Scalar>(y * array.Width() + x);
    });
  }

  //----------------------------------------------------------------------------

  void CheckApplyOpUpdate(typename CudaArrayType::Scalar value) {
    // Note the *this capture!
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#star-this-capture
    array.Fill(value);
    CUDA_CHECK_ERROR
    array.ApplyOp([=, *this] __device__(size_t x, size_t y) {
      return value + array.get(x, y);
    });
    DownloadAndCheck([=](size_t x, size_t y) { return value + value; });
  }

  //----------------------------------------------------------------------------

  CudaArrayType array;
  std::vector<typename CudaArrayType::Scalar> result;
};


}  // namespace test

}  // namespace cua

#endif  // CUDA_ARRAY2D_BASE_TEST_H_

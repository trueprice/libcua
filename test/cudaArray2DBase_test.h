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
class CudaArray2DTestWrapper
    : public PrimitiveConverter<typename CudaArrayType::Scalar> {
 public:
  typedef typename CudaArrayType::Scalar Scalar;
  using PrimitiveConverter<Scalar>::AsScalar;

  //----------------------------------------------------------------------------

  CudaArray2DTestWrapper(size_t width = 10, size_t height = 10)
      : array_(width, height) {}

  //----------------------------------------------------------------------------

  template <typename HostFunction>
  static void DownloadAndCheck(const CudaArrayType& array,
                               const HostFunction& host_function) {
    CUDA_CHECK_ERROR
    std::vector<Scalar> result(array.Size());
    array.CopyTo(result.data());
    CUDA_CHECK_ERROR

    for (size_t y = 0; y < array.Height(); ++y) {
      for (size_t x = 0; x < array.Width(); ++x) {
        const size_t i = y * array.Width() + x;
        EXPECT_EQ(result[i], host_function(x, y));
      }
    }
  }

  template <typename HostFunction>
  void DownloadAndCheck(const HostFunction& host_function) {
    DownloadAndCheck(array_, host_function);
  }

  //----------------------------------------------------------------------------

  void CheckView() {
    array_.Fill(AsScalar(0));
    CUDA_CHECK_ERROR
    ASSERT_GT(array_.Height(), 1);
    for (size_t col = 0; col < array_.Width(); ++col) {
      auto view = array_.View(col, 1, 1, array_.Height());
      view.Fill(AsScalar(col));
      CUDA_CHECK_ERROR
    }
    DownloadAndCheck(
        [](size_t x, size_t y) { return AsScalar((y > 0) ? x : 0); });
  }

  //----------------------------------------------------------------------------

  void CheckViewDownload() {
    array_.Fill(AsScalar(0));
    CUDA_CHECK_ERROR
    ASSERT_GT(array_.Width(), 2);
    ASSERT_GT(array_.Height(), 2);
    auto view = array_.View(1, 1, array_.Width() - 2, array_.Height() - 2);
    view.ApplyOp([] __device__(size_t x, size_t y) { return AsScalar(x + y); });
    DownloadAndCheck(view, [](size_t x, size_t y) { return AsScalar(x + y); });
  }

  //----------------------------------------------------------------------------

  void CheckFill(Scalar value) {
    array_.Fill(value);
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceAdd(Scalar value) {
    array_.Fill(0.);
    CUDA_CHECK_ERROR
    array_ += value;
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
    array_ += value;
    DownloadAndCheck([=](size_t x, size_t y) { return value + value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceSubtract(Scalar value) {
    array_.Fill(value + value);
    CUDA_CHECK_ERROR
    array_ -= value;
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
    array_ -= value;
    DownloadAndCheck([=](size_t x, size_t y) { return 0; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceMultiply(Scalar value) {
    array_.Fill(1.);
    CUDA_CHECK_ERROR
    array_ *= value;
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
    array_ *= value;
    DownloadAndCheck([=](size_t x, size_t y) { return value * value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceDivide(Scalar value) {
    array_.Fill(value * value);
    CUDA_CHECK_ERROR
    array_ /= value;
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
    array_ /= value;
    DownloadAndCheck([=](size_t x, size_t y) { return 1; });
  }

  //----------------------------------------------------------------------------

  void CheckApplyOpConstant(Scalar value) {
    array_.ApplyOp([=] __device__(size_t x, size_t y) { return value; });
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
  }

  //----------------------------------------------------------------------------

  void CheckApplyOpLinear() {
    // Note the *this capture!
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#star-this-capture
    array_.ApplyOp([ =, *this ] __device__(size_t x, size_t y) {
      return AsScalar(y * array_.Width() + x);
    });
    DownloadAndCheck([=](size_t x, size_t y) {
      return AsScalar(y * array_.Width() + x);
    });
  }

  //----------------------------------------------------------------------------

  void CheckApplyOpUpdate(Scalar value) {
    // Note the *this capture!
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#star-this-capture
    array_.Fill(value);
    CUDA_CHECK_ERROR
    array_.ApplyOp([ =, *this ] __device__(size_t x, size_t y) {
      return value + array_.get(x, y);
    });
    DownloadAndCheck([=](size_t x, size_t y) { return value + value; });
  }

  //----------------------------------------------------------------------------

 private:
  CudaArrayType array_;
};

}  // namespace test

}  // namespace cua

#endif  // CUDA_ARRAY2D_BASE_TEST_H_

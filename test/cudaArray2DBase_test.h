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
        EXPECT_EQ(result[i], host_function(x, y)) << "Coordinate: " << x << " "
                                                  << y;
      }
    }
  }

  template <typename HostFunction>
  void DownloadAndCheck(const HostFunction& host_function) {
    DownloadAndCheck(array_, host_function);
  }

  //----------------------------------------------------------------------------

  void CheckUpload() {
    std::vector<Scalar> data(array_.Size());
    for (size_t i = 0; i < array_.Size(); ++i) {
      data[i] = AsScalar(i);
    }
    array_ = data.data();
    DownloadAndCheck([=](size_t x, size_t y) {
      return AsScalar(y * array_.Width() + x);
    });
  }

  //----------------------------------------------------------------------------

  void CheckView() {
    ASSERT_GT(array_.Height(), 1);

    array_.Fill(AsScalar(0));
    CUDA_CHECK_ERROR

    for (size_t col = 0; col < array_.Width(); ++col) {
      auto view = array_.View(col, 1, 1, array_.Height() - 1);
      view.Fill(AsScalar(col));
      CUDA_CHECK_ERROR
    }

    DownloadAndCheck(
        [](size_t x, size_t y) { return AsScalar((y > 0) ? x : 0); });
  }

  //----------------------------------------------------------------------------

  void CheckViewDownload() {
    ASSERT_GT(array_.Width(), 2);
    ASSERT_GT(array_.Height(), 2);

    const Scalar kFillValue = AsScalar(14);
    array_.Fill(kFillValue);
    CUDA_CHECK_ERROR

    auto view = array_.View(1, 1, array_.Width() - 2, array_.Height() - 2);
    view.ApplyOp([] __device__(size_t x, size_t y) { return AsScalar(x + y); });

    DownloadAndCheck(view, [](size_t x, size_t y) { return AsScalar(x + y); });
    DownloadAndCheck([=](size_t x, size_t y) {
      return (x > 0 && x < array_.Width() - 1 && y > 0 &&
              y < array_.Height() - 1)
                 ? AsScalar(x - 1 + y - 1)
                 : kFillValue;
    });
  }

  //----------------------------------------------------------------------------

  void CheckViewUpload() {
    ASSERT_GT(array_.Width(), 2);
    ASSERT_GT(array_.Height(), 2);

    const Scalar kFillValue = AsScalar(14);
    array_.Fill(kFillValue);
    CUDA_CHECK_ERROR

    auto view = array_.View(1, 1, array_.Width() - 2, array_.Height() - 2);

    std::vector<Scalar> data(view.Size());
    for (size_t i = 0; i < view.Size(); ++i) {
      data[i] = AsScalar(i);
    }
    view = data.data();
    DownloadAndCheck(view, [=](size_t x, size_t y) {
      return AsScalar(y * view.Width() + x);
    });
  }

  //----------------------------------------------------------------------------

  void CheckNestedViews() {
    ASSERT_GT(array_.Width(), 4);
    ASSERT_GT(array_.Height(), 4);

    const Scalar kFillValue0 = AsScalar(1);
    const Scalar kFillValue1 = AsScalar(2);
    const Scalar kFillValue2 = AsScalar(3);

    array_.Fill(kFillValue0);
    CUDA_CHECK_ERROR

    auto view1 = array_.View(1, 1, array_.Width() - 2, array_.Height() - 2);
    view1.Fill(kFillValue1);
    auto view2 = view1.View(1, 1, view1.Width() - 2, view1.Height() - 2);
    view2.Fill(kFillValue2);

    DownloadAndCheck([=](size_t x, size_t y) {
      if (x > 1 && x < array_.Width() - 2 && y > 1 && y < array_.Height() - 2) {
        return kFillValue2;
      } else if (x > 0 && x < array_.Width() - 1 && y > 0 &&
                 y < array_.Height() - 1) {
        return kFillValue1;
      } else {
        return kFillValue0;
      }
    });
  }

  //----------------------------------------------------------------------------

  void CheckFill(Scalar value) {
    array_.Fill(value);
    DownloadAndCheck([=](size_t x, size_t y) { return value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceAdd(Scalar value) {
    array_.Fill(AsScalar(0));
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
    DownloadAndCheck([](size_t x, size_t y) { return AsScalar(0); });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceMultiply(Scalar value) {
    array_.Fill(AsScalar(1));
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
    DownloadAndCheck([](size_t x, size_t y) { return AsScalar(1); });
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

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

#ifndef CUDA_ARRAY2D_BASE_TEST_H_
#define CUDA_ARRAY2D_BASE_TEST_H_

#include <vector>

#include "cudaArray2D.h"
#include "cudaSurface2D.h"
#include "cudaTexture2D.h"
#include "util.h"

//------------------------------------------------------------------------------

template <typename CudaArrayType>
class CudaArray2DBaseTest
    : public ::testing::Test,
      public PrimitiveConverter<typename CudaArrayType::Scalar> {
 public:
  typedef typename CudaArrayType::Scalar Scalar;
  typedef typename CudaArrayType::SizeType SizeType;
  typedef typename CudaArrayType::IndexType IndexType;
  using PrimitiveConverter<Scalar>::AsScalar;

  //----------------------------------------------------------------------------

  CudaArray2DBaseTest(SizeType width = 10, SizeType height = 10)
      : array_(width, height) {}

  //----------------------------------------------------------------------------

  template <typename SourceCudaArrayType, typename HostFunction>
  static void DownloadAndCheck(const SourceCudaArrayType& array,
                               const HostFunction& host_function) {
    CUDA_CHECK_ERROR
    std::vector<Scalar> result(array.Size());
    array.CopyTo(result.data());
    CUDA_CHECK_ERROR

    for (IndexType y = 0; y < array.Height(); ++y) {
      for (IndexType x = 0; x < array.Width(); ++x) {
        const IndexType i = y * array.Width() + x;
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
    for (IndexType i = 0; i < array_.Size(); ++i) {
      data[i] = AsScalar(i);
    }
    array_ = data.data();
    DownloadAndCheck([=](IndexType x, IndexType y) {
      return AsScalar(y * array_.Width() + x);
    });
  }

  //----------------------------------------------------------------------------

  void CheckView() {
    ASSERT_GT(array_.Height(), 1);

    array_.Fill(AsScalar(0));
    CUDA_CHECK_ERROR

    for (IndexType col = 0; col < array_.Width(); ++col) {
      auto view = array_.View(col, 1, 1, array_.Height() - 1);
      view.Fill(AsScalar(col));
      CUDA_CHECK_ERROR
    }

    DownloadAndCheck(
        [](IndexType x, IndexType y) { return AsScalar((y > 0) ? x : 0); });
  }

  //----------------------------------------------------------------------------

  void CheckViewDownload() {
    ASSERT_GT(array_.Width(), 2);
    ASSERT_GT(array_.Height(), 2);

    const Scalar kFillValue = AsScalar(14);
    array_.Fill(kFillValue);
    CUDA_CHECK_ERROR

    auto view = array_.View(1, 1, array_.Width() - 2, array_.Height() - 2);
    view.ApplyOp(
        [] __device__(IndexType x, IndexType y) { return AsScalar(x + y); });

    DownloadAndCheck(view,
                     [](IndexType x, IndexType y) { return AsScalar(x + y); });
    DownloadAndCheck([=](IndexType x, IndexType y) {
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
    for (IndexType i = 0; i < view.Size(); ++i) {
      data[i] = AsScalar(i);
    }
    view = data.data();
    DownloadAndCheck(view, [=](IndexType x, IndexType y) {
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

    DownloadAndCheck([=](IndexType x, IndexType y) {
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
    DownloadAndCheck([=](IndexType x, IndexType y) { return value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceAdd(Scalar value) {
    array_.Fill(AsScalar(0));
    CUDA_CHECK_ERROR
    array_ += value;
    DownloadAndCheck([=](IndexType x, IndexType y) { return value; });
    array_ += value;
    DownloadAndCheck([=](IndexType x, IndexType y) { return value + value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceSubtract(Scalar value) {
    array_.Fill(value + value);
    CUDA_CHECK_ERROR
    array_ -= value;
    DownloadAndCheck([=](IndexType x, IndexType y) { return value; });
    array_ -= value;
    DownloadAndCheck([](IndexType x, IndexType y) { return AsScalar(0); });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceMultiply(Scalar value) {
    array_.Fill(AsScalar(1));
    CUDA_CHECK_ERROR
    array_ *= value;
    DownloadAndCheck([=](IndexType x, IndexType y) { return value; });
    array_ *= value;
    DownloadAndCheck([=](IndexType x, IndexType y) { return value * value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceDivide(Scalar value) {
    array_.Fill(value * value);
    CUDA_CHECK_ERROR
    array_ /= value;
    DownloadAndCheck([=](IndexType x, IndexType y) { return value; });
    array_ /= value;
    DownloadAndCheck([](IndexType x, IndexType y) { return AsScalar(1); });
  }

  //----------------------------------------------------------------------------

  void CheckApplyOpConstant(Scalar value) {
    array_.ApplyOp([=] __device__(IndexType x, IndexType y) { return value; });
    DownloadAndCheck([=](IndexType x, IndexType y) { return value; });
  }

  //----------------------------------------------------------------------------

  void CheckApplyOpLinear() {
    // Unfortunately, we can't use *this capture within the testing framework,
    // so we'll avoid accessing array_ in the lambda.
    const SizeType width = array_.Width();
    array_.ApplyOp([=] __device__(IndexType x, IndexType y) {
      return AsScalar(y * width + x);
    });
    DownloadAndCheck([=](IndexType x, IndexType y) {
      return AsScalar(y * array_.Width() + x);
    });
  }

  //----------------------------------------------------------------------------

  void CheckApplyOpUpdate(Scalar value) {
    array_.Fill(value);
    CUDA_CHECK_ERROR

    // Unfortunately, we can't use *this capture within the testing framework,
    // so we'll avoid accessing array_ in the lambda.
    CudaArrayType local_array(array_);  // shallow copy for lambda capture
    array_.ApplyOp([=] __device__(IndexType x, IndexType y) {
      return value + local_array.get(x, y);
    });
    DownloadAndCheck([=](IndexType x, IndexType y) { return value + value; });
  }

  //----------------------------------------------------------------------------
  
  template <typename OtherType>
  void CheckCopyTo() {
    const SizeType width = array_.Width();
    array_.ApplyOp([=] __device__(IndexType x, IndexType y) {
      return AsScalar(y * width + x);
    });

    OtherType other(array_.Width(), array_.Height());
    array_.CopyTo(&other);

    DownloadAndCheck(other, [=](IndexType x, IndexType y) {
      return AsScalar(y * array_.Width() + x);
    });
  }

  void CheckCopyToArray() {
    CheckCopyTo<cua::CudaArray2D<Scalar>>();
  }

  void CheckCopyToSurface() {
    if (TypeInfo<Scalar>::supported_for_textures::value) {
      CheckCopyTo<cua::CudaSurface2D<Scalar>>();
    }
  }

  void CheckCopyToTexture() {
    if (TypeInfo<Scalar>::supported_for_textures::value) {
      CheckCopyTo<cua::CudaTexture2D<Scalar>>();
    }
  }

  //----------------------------------------------------------------------------

 private:
  CudaArrayType array_;
};

//------------------------------------------------------------------------------
//
// Test suite definition.
//
//------------------------------------------------------------------------------

TYPED_TEST_SUITE_P(CudaArray2DBaseTest);

TYPED_TEST_P(CudaArray2DBaseTest, TestUpload) { this->CheckUpload(); }

TYPED_TEST_P(CudaArray2DBaseTest, TestView) { this->CheckView(); }

TYPED_TEST_P(CudaArray2DBaseTest, TestViewDownload) {
  this->CheckViewDownload();
}

TYPED_TEST_P(CudaArray2DBaseTest, TestViewUpload) { this->CheckViewUpload(); }

TYPED_TEST_P(CudaArray2DBaseTest, TestNestedViews) { this->CheckNestedViews(); }

TYPED_TEST_P(CudaArray2DBaseTest, TestFill) {
  this->CheckFill(this->AsScalar(3));
  this->CheckFill(this->AsScalar(0));
}

TYPED_TEST_P(CudaArray2DBaseTest, TestInPlaceAdd) {
  this->CheckInPlaceAdd(this->AsScalar(3));
}

TYPED_TEST_P(CudaArray2DBaseTest, TestInPlaceSubtract) {
  this->CheckInPlaceSubtract(this->AsScalar(3));
}

TYPED_TEST_P(CudaArray2DBaseTest, TestInPlaceMultiply) {
  this->CheckInPlaceMultiply(this->AsScalar(3));
}

TYPED_TEST_P(CudaArray2DBaseTest, TestInPlaceDivide) {
  this->CheckInPlaceDivide(this->AsScalar(3));
}

TYPED_TEST_P(CudaArray2DBaseTest, TestApplyOpConstant) {
  this->CheckApplyOpConstant(this->AsScalar(3));
  this->CheckApplyOpConstant(this->AsScalar(0));
}

TYPED_TEST_P(CudaArray2DBaseTest, TestApplyOpLinear) {
  this->CheckApplyOpLinear();
}

TYPED_TEST_P(CudaArray2DBaseTest, TestApplyOpUpdate) {
  this->CheckApplyOpUpdate(this->AsScalar(3));
}

TYPED_TEST_P(CudaArray2DBaseTest, TestCopyToArray) {
  this->CheckCopyToArray();
}

TYPED_TEST_P(CudaArray2DBaseTest, TestCopyToSurface) {
  this->CheckCopyToSurface();
}

TYPED_TEST_P(CudaArray2DBaseTest, TestCopyToTexture) {
  this->CheckCopyToTexture();
}

REGISTER_TYPED_TEST_SUITE_P(CudaArray2DBaseTest, TestUpload, TestView,
                            TestViewDownload, TestViewUpload, TestNestedViews,
                            TestFill, TestInPlaceAdd, TestInPlaceSubtract,
                            TestInPlaceMultiply, TestInPlaceDivide,
                            TestApplyOpConstant, TestApplyOpLinear,
                            TestApplyOpUpdate, TestCopyToArray,
                            TestCopyToSurface, TestCopyToTexture);

#endif  // CUDA_ARRAY2D_BASE_TEST_H_

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

#ifndef CUDA_ARRAY3D_BASE_TEST_H_
#define CUDA_ARRAY3D_BASE_TEST_H_

#include <vector>

#include "gtest/gtest.h"

#include "cudaArray3D.h"
#include "cudaSurface3D.h"
#include "cudaTexture3D.h"
#include "util.h"

//------------------------------------------------------------------------------

template <typename CudaArrayType>
class CudaArray3DBaseTest
    : public ::testing::Test,
      public PrimitiveConverter<typename CudaArrayType::Scalar> {
 public:
  typedef typename CudaArrayType::Scalar Scalar;
  typedef typename CudaArrayType::SizeType SizeType;
  typedef typename CudaArrayType::IndexType IndexType;
  using PrimitiveConverter<Scalar>::AsScalar;

  //----------------------------------------------------------------------------

  CudaArray3DBaseTest(SizeType width = 10, SizeType height = 10,
                      SizeType depth = 10)
      : array_(width, height, depth) {}

  //----------------------------------------------------------------------------

  template <typename SourceCudaArrayType, typename HostFunction>
  static void DownloadAndCheck(const SourceCudaArrayType& array,
                               const HostFunction& host_function) {
    CUDA_CHECK_ERROR
    std::vector<Scalar> result(array.Size());
    array.CopyTo(result.data());
    CUDA_CHECK_ERROR

    for (IndexType z = 0; z < array.Depth(); ++z) {
      for (IndexType y = 0; y < array.Height(); ++y) {
        for (IndexType x = 0; x < array.Width(); ++x) {
          const IndexType i = (z * array.Height() + y) * array.Width() + x;
          EXPECT_EQ(result[i], host_function(x, y, z)) << "Coordinate: " << x
                                                       << " " << y << " " << z;
          ;
        }
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
    DownloadAndCheck([=](IndexType x, IndexType y, IndexType z) {
      return AsScalar((z * array_.Height() + y) * array_.Width() + x);
    });
  }

  //----------------------------------------------------------------------------

  void CheckView() {
    ASSERT_GT(array_.Height(), 1);
    ASSERT_GT(array_.Depth(), 1);

    array_.Fill(AsScalar(0));
    CUDA_CHECK_ERROR

    for (IndexType col = 0; col < array_.Width(); ++col) {
      auto view =
          array_.View(col, 1, 1, 1, array_.Height() - 1, array_.Depth() - 1);
      view.Fill(AsScalar(col));
      CUDA_CHECK_ERROR
    }

    DownloadAndCheck([](IndexType x, IndexType y, IndexType z) {
      return AsScalar((z > 0 && y > 0) ? x : 0);
    });
  }

  //----------------------------------------------------------------------------

  void CheckViewDownload() {
    ASSERT_GT(array_.Width(), 2);
    ASSERT_GT(array_.Height(), 2);
    ASSERT_GT(array_.Depth(), 2);

    const Scalar kFillValue = AsScalar(14);
    array_.Fill(kFillValue);
    CUDA_CHECK_ERROR

    auto view = array_.View(1, 1, 1, array_.Width() - 2, array_.Height() - 2,
                            array_.Depth() - 2);
    view.ApplyOp([] __device__(IndexType x, IndexType y, IndexType z) {
      return AsScalar(x + y + z);
    });

    DownloadAndCheck(view, [](IndexType x, IndexType y, IndexType z) {
      return AsScalar(x + y + z);
    });
    DownloadAndCheck([=](IndexType x, IndexType y, IndexType z) {
      return (x > 0 && x < array_.Width() - 1 && y > 0 &&
              y < array_.Height() - 1 && z > 0 && z < array_.Depth() - 1)
                 ? AsScalar(x - 1 + y - 1 + z - 1)
                 : kFillValue;
    });
  }

  //----------------------------------------------------------------------------

  void CheckViewUpload() {
    ASSERT_GT(array_.Width(), 2);
    ASSERT_GT(array_.Height(), 2);
    ASSERT_GT(array_.Depth(), 2);

    const Scalar kFillValue = AsScalar(14);
    array_.Fill(kFillValue);
    CUDA_CHECK_ERROR

    auto view = array_.View(1, 1, 1, array_.Width() - 2, array_.Height() - 2,
                            array_.Depth() - 2);

    std::vector<Scalar> data(view.Size());
    for (IndexType i = 0; i < view.Size(); ++i) {
      data[i] = AsScalar(i);
    }
    view = data.data();
    DownloadAndCheck(view, [=](IndexType x, IndexType y, IndexType z) {
      return AsScalar((z * view.Height() + y) * view.Width() + x);
    });
  }

  //----------------------------------------------------------------------------

  void CheckNestedViews() {
    ASSERT_GT(array_.Width(), 4);
    ASSERT_GT(array_.Height(), 4);
    ASSERT_GT(array_.Depth(), 4);

    const Scalar kFillValue0 = AsScalar(1);
    const Scalar kFillValue1 = AsScalar(2);
    const Scalar kFillValue2 = AsScalar(3);

    array_.Fill(kFillValue0);
    CUDA_CHECK_ERROR

    auto view1 = array_.View(1, 1, 1, array_.Width() - 2, array_.Height() - 2,
                             array_.Depth() - 2);
    view1.Fill(kFillValue1);
    auto view2 = view1.View(1, 1, 1, view1.Width() - 2, view1.Height() - 2,
                            view1.Depth() - 2);
    view2.Fill(kFillValue2);

    DownloadAndCheck([=](IndexType x, IndexType y, IndexType z) {
      if (x > 1 && x < array_.Width() - 2 && y > 1 && y < array_.Height() - 2 &&
          z > 1 && z < array_.Depth() - 2) {
        return kFillValue2;
      } else if (x > 0 && x < array_.Width() - 1 && y > 0 &&
                 y < array_.Height() - 1 && z > 0 && z < array_.Depth() - 1) {
        return kFillValue1;
      } else {
        return kFillValue0;
      }
    });
  }

  //----------------------------------------------------------------------------

  void CheckFill(Scalar value) {
    array_.Fill(value);
    DownloadAndCheck(
        [=](IndexType x, IndexType y, IndexType z) { return value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceAdd(Scalar value) {
    array_.Fill(AsScalar(0));
    CUDA_CHECK_ERROR
    array_ += value;
    DownloadAndCheck(
        [=](IndexType x, IndexType y, IndexType z) { return value; });
    array_ += value;
    DownloadAndCheck(
        [=](IndexType x, IndexType y, IndexType z) { return value + value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceSubtract(Scalar value) {
    array_.Fill(value + value);
    CUDA_CHECK_ERROR
    array_ -= value;
    DownloadAndCheck(
        [=](IndexType x, IndexType y, IndexType z) { return value; });
    array_ -= value;
    DownloadAndCheck(
        [](IndexType x, IndexType y, IndexType z) { return AsScalar(0); });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceMultiply(Scalar value) {
    array_.Fill(AsScalar(1));
    CUDA_CHECK_ERROR
    array_ *= value;
    DownloadAndCheck(
        [=](IndexType x, IndexType y, IndexType z) { return value; });
    array_ *= value;
    DownloadAndCheck(
        [=](IndexType x, IndexType y, IndexType z) { return value * value; });
  }

  //----------------------------------------------------------------------------

  void CheckInPlaceDivide(Scalar value) {
    array_.Fill(value * value);
    CUDA_CHECK_ERROR
    array_ /= value;
    DownloadAndCheck(
        [=](IndexType x, IndexType y, IndexType z) { return value; });
    array_ /= value;
    DownloadAndCheck(
        [](IndexType x, IndexType y, IndexType z) { return AsScalar(1); });
  }

  //----------------------------------------------------------------------------

  void CheckApplyOpConstant(Scalar value) {
    array_.ApplyOp([=] __device__(IndexType x, IndexType y, IndexType z) {
      return value;
    });
    DownloadAndCheck(
        [=](IndexType x, IndexType y, IndexType z) { return value; });
  }

  //----------------------------------------------------------------------------

  void CheckApplyOpLinear() {
    // Unfortunately, we can't use *this capture within the testing framework,
    // so we'll avoid accessing array_ in the lambda.
    const SizeType width = array_.Width();
    const SizeType height = array_.Height();
    array_.ApplyOp([=] __device__(IndexType x, IndexType y, IndexType z) {
      return AsScalar((z * height + y) * width + x);
    });
    DownloadAndCheck([=](IndexType x, IndexType y, IndexType z) {
      return AsScalar((z * array_.Height() + y) * array_.Width() + x);
    });
  }

  //----------------------------------------------------------------------------

  void CheckApplyOpUpdate(Scalar value) {
    array_.Fill(value);
    CUDA_CHECK_ERROR

    CudaArrayType local_array(array_);  // shallow copy for lambda capture
    array_.ApplyOp([=] __device__(IndexType x, IndexType y, IndexType z) {
      return value + local_array.get(x, y, z);
    });
    DownloadAndCheck(
        [=](IndexType x, IndexType y, IndexType z) { return value + value; });
  }

  //----------------------------------------------------------------------------

  template <typename OtherType>
  void CheckCopyTo() {
    const SizeType width = array_.Width();
    const SizeType height = array_.Height();
    array_.ApplyOp([=] __device__(IndexType x, IndexType y, IndexType z) {
      return AsScalar((z * height + y) * width + x);
    });

    OtherType other(array_.Width(), array_.Height(), array_.Depth());
    array_.CopyTo(&other);

    DownloadAndCheck(other, [=](IndexType x, IndexType y, IndexType z) {
      return AsScalar((z * array_.Height() + y) * array_.Width() + x);
    });
  }

  void CheckCopyToArray() {
    CheckCopyTo<cua::CudaArray3D<Scalar>>();
  }

  void CheckCopyToSurface3D() {
    if (TypeInfo<Scalar>::supported_for_textures::value) {
      CheckCopyTo<cua::CudaSurface3D<Scalar>>();
    }
  }

  void CheckCopyToSurface2DArray() {
    if (TypeInfo<Scalar>::supported_for_textures::value) {
      CheckCopyTo<cua::CudaSurface2DArray<Scalar>>();
    }
  }

  void CheckCopyToTexture3D() {
    if (TypeInfo<Scalar>::supported_for_textures::value) {
      CheckCopyTo<cua::CudaTexture3D<Scalar>>();
    }
  }

  void CheckCopyToTexture2DArray() {
    if (TypeInfo<Scalar>::supported_for_textures::value) {
      CheckCopyTo<cua::CudaTexture2DArray<Scalar>>();
    }
  }

  //----------------------------------------------------------------------------

  /*
  template <typename OtherType,
            typename std::enable_if<std::is_same<
                typename OtherType::Scalar, float>::value>::type other_is_float>
  void CheckCastToFloat() {
    const SizeType width = array_.Width();
    const SizeType height = array_.Height();
    array_.ApplyOp([=] __device__(IndexType x, IndexType y, IndexType z) {
      return AsScalar((z * height + y) * width + x);
    });

    OtherType other(array_.Width(), array_.Height(), array_.Depth());
    array_.CopyTo(&other);

    DownloadAndCheck(other, [=](IndexType x, IndexType y, IndexType z) {
      return static_cast<float>((z * array_.Height() + y) * array_.Width() + x);
    });
  }

  void CheckCastToFloatArray() {
    if (std::is_same<Scalar, unsigned int>::value) {
      CheckCastToFloat<cua::CudaArray3D<float>>();
    }
  }

  void CheckCastToFloatSurface() {
    if (std::is_same<Scalar, unsigned int>::value) {
      CheckCastToFloat<cua::CudaSurface3D<float>>();
    }
  }

  void CheckCastToFloatTexture() {
    if (std::is_same<Scalar, unsigned int>::value) {
      CheckCastToFloat<cua::CudaTexture3D<float>>();
    }
  }
  */

  //----------------------------------------------------------------------------

 private:
  CudaArrayType array_;
};

//------------------------------------------------------------------------------
//
// Test suite definition.
//
//------------------------------------------------------------------------------

TYPED_TEST_SUITE_P(CudaArray3DBaseTest);

TYPED_TEST_P(CudaArray3DBaseTest, TestUpload) { this->CheckUpload(); }

TYPED_TEST_P(CudaArray3DBaseTest, TestView) { this->CheckView(); }

TYPED_TEST_P(CudaArray3DBaseTest, TestViewDownload) {
  this->CheckViewDownload();
}

TYPED_TEST_P(CudaArray3DBaseTest, TestViewUpload) { this->CheckViewUpload(); }

TYPED_TEST_P(CudaArray3DBaseTest, TestNestedViews) { this->CheckNestedViews(); }

TYPED_TEST_P(CudaArray3DBaseTest, TestFill) {
  this->CheckFill(this->AsScalar(3));
  this->CheckFill(this->AsScalar(0));
}

TYPED_TEST_P(CudaArray3DBaseTest, TestInPlaceAdd) {
  this->CheckInPlaceAdd(this->AsScalar(3));
}

TYPED_TEST_P(CudaArray3DBaseTest, TestInPlaceSubtract) {
  this->CheckInPlaceSubtract(this->AsScalar(3));
}

TYPED_TEST_P(CudaArray3DBaseTest, TestInPlaceMultiply) {
  this->CheckInPlaceMultiply(this->AsScalar(3));
}

TYPED_TEST_P(CudaArray3DBaseTest, TestInPlaceDivide) {
  this->CheckInPlaceDivide(this->AsScalar(3));
}

TYPED_TEST_P(CudaArray3DBaseTest, TestApplyOpConstant) {
  this->CheckApplyOpConstant(this->AsScalar(3));
  this->CheckApplyOpConstant(this->AsScalar(0));
}

TYPED_TEST_P(CudaArray3DBaseTest, TestApplyOpLinear) {
  this->CheckApplyOpLinear();
}

TYPED_TEST_P(CudaArray3DBaseTest, TestApplyOpUpdate) {
  this->CheckApplyOpUpdate(this->AsScalar(3));
}

TYPED_TEST_P(CudaArray3DBaseTest, TestCopyToArray) {
  this->CheckCopyToArray();
}

TYPED_TEST_P(CudaArray3DBaseTest, TestCopyToSurface3D) {
  this->CheckCopyToSurface3D();
}

TYPED_TEST_P(CudaArray3DBaseTest, TestCopyToSurface2DArray) {
  this->CheckCopyToSurface2DArray();
}

TYPED_TEST_P(CudaArray3DBaseTest, TestCopyToTexture3D) {
  this->CheckCopyToTexture3D();
}

TYPED_TEST_P(CudaArray3DBaseTest, TestCopyToTexture2DArray) {
  this->CheckCopyToTexture2DArray();
}

REGISTER_TYPED_TEST_SUITE_P(CudaArray3DBaseTest, TestUpload, TestView,
                            TestViewDownload, TestViewUpload, TestNestedViews,
                            TestFill, TestInPlaceAdd, TestInPlaceSubtract,
                            TestInPlaceMultiply, TestInPlaceDivide,
                            TestApplyOpConstant, TestApplyOpLinear,
                            TestApplyOpUpdate, TestCopyToArray,
                            TestCopyToSurface3D, TestCopyToSurface2DArray,
                            TestCopyToTexture3D, TestCopyToTexture2DArray);

#endif  // CUDA_ARRAY3D_BASE_TEST_H_

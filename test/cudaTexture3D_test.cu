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

#include "cudaArray3D.h"
#include "cudaTexture3D.h"

#include "gtest/gtest.h"

#include "util.h"

namespace {

template <typename CudaTextureType>
class CudaTexture3DTest
    : public ::testing::Test,
      public PrimitiveConverter<typename CudaTextureType::Scalar> {
 public:
  typedef typename CudaTextureType::Scalar Scalar;
  using PrimitiveConverter<Scalar>::AsScalar;

  //----------------------------------------------------------------------------

  CudaTexture3DTest(size_t width = 10, size_t height = 10, size_t depth = 10)
      : texture_(width, height, depth) {}

  //----------------------------------------------------------------------------

  template <typename CudaArrayType, typename HostFunction>
  static void DownloadAndCheck(const CudaArrayType& array,
                               const HostFunction& host_function) {
    CUDA_CHECK_ERROR
    std::vector<Scalar> result(array.Size());
    array.CopyTo(result.data());
    CUDA_CHECK_ERROR

    for (size_t z = 0; z < array.Depth(); ++z) {
      for (size_t y = 0; y < array.Height(); ++y) {
        for (size_t x = 0; x < array.Width(); ++x) {
          const size_t i = (z * array.Height() + y) * array.Width() + x;
          EXPECT_EQ(result[i], host_function(x, y, z)) << "Coordinate: " << x
                                                       << " " << y << " " << z;
        }
      }
    }
  }

  template <typename HostFunction>
  void DownloadAndCheck(const HostFunction& host_function) {
    DownloadAndCheck(texture_, host_function);
  }

  //----------------------------------------------------------------------------

  void Upload() {
    std::vector<Scalar> data(texture_.Size());
    for (size_t i = 0; i < texture_.Size(); ++i) {
      data[i] = AsScalar(i);
    }
    texture_ = data.data();
  }

  //----------------------------------------------------------------------------

  void CheckUpload() {
    Upload();
    DownloadAndCheck([=](size_t x, size_t y, size_t z) {
      return AsScalar((z * texture_.Height() + y) * texture_.Width() + x);
    });
  }

  //----------------------------------------------------------------------------

  void CheckGet() {
    Upload();

    // Unfortunately, we can't use *this capture within the testing framework,
    // so we'll avoid accessing texture_ in the lambda.
    cua::CudaArray3D<Scalar> array(texture_.Width(), texture_.Height(),
                                   texture_.Depth());
    CudaTextureType local_texture(texture_);  // shallow copy for lambda capture
    array.ApplyOp([=] __device__(size_t x, size_t y, size_t z) {
      return local_texture.get(x, y, z);
    });

    DownloadAndCheck(array, [=](size_t x, size_t y, size_t z) {
      return AsScalar((z * texture_.Height() + y) * texture_.Width() + x);
    });
  }

  //----------------------------------------------------------------------------

 private:
  CudaTextureType texture_;
};

//------------------------------------------------------------------------------
//
// Test suite definition.
//
//------------------------------------------------------------------------------

TYPED_TEST_SUITE_P(CudaTexture3DTest);

TYPED_TEST_P(CudaTexture3DTest, TestUpload) { this->CheckUpload(); }

TYPED_TEST_P(CudaTexture3DTest, TestGet) { this->CheckGet(); }

REGISTER_TYPED_TEST_SUITE_P(CudaTexture3DTest, TestUpload, TestGet);

typedef ::testing::Types<
    cua::CudaTexture2DArray<float>, cua::CudaTexture2DArray<float2>,
    cua::CudaTexture2DArray<float4>, cua::CudaTexture2DArray<unsigned char>,
    cua::CudaTexture2DArray<uchar2>, cua::CudaTexture2DArray<uchar4>,
    cua::CudaTexture2DArray<unsigned int>, cua::CudaTexture2DArray<uint2>,
    cua::CudaTexture2DArray<uint4>, cua::CudaTexture3D<float>,
    cua::CudaTexture3D<float2>, cua::CudaTexture3D<float4>,
    cua::CudaTexture3D<unsigned char>, cua::CudaTexture3D<uchar2>,
    cua::CudaTexture3D<uchar4>, cua::CudaTexture3D<unsigned int>,
    cua::CudaTexture3D<uint2>, cua::CudaTexture3D<uint4> >
    Types;

INSTANTIATE_TYPED_TEST_SUITE_P(CudaTexture3DTest, CudaTexture3DTest, Types);

//------------------------------------------------------------------------------

}  // namespace

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

#include "cudaArray2D.h"

#include "gtest/gtest.h"

#include "cudaArray2DBase_test.h"
#include "util.h"

namespace {

//------------------------------------------------------------------------------

TEST(CudaArray2DTest, TestInPlaceOps) {
  cua::test::CudaArray2DTestWrapper<cua::CudaArray2D<float>> wrapper;
  wrapper.CheckInPlaceAdd(3.f);
  wrapper.CheckInPlaceSubtract(3.f);
  wrapper.CheckInPlaceMultiply(3.f);
  wrapper.CheckInPlaceDivide(3.f);
}

//------------------------------------------------------------------------------

TEST(CudaArray2DTest, TestFill) {
#define TYPE_TEST(TYPE, ...)                                           \
  {                                                                    \
    cua::test::CudaArray2DTestWrapper<cua::CudaArray2D<TYPE>> wrapper; \
    wrapper.CheckFill(TYPE{__VA_ARGS__});                              \
    wrapper.CheckFill(TYPE{0});                                        \
  }
  TYPE_TEST(float, 2.f)
  TYPE_TEST(float3, 2.f, 3.f, 4.f)
  TYPE_TEST(float4, 2.f, 3.f, 4.f, 5.f)
  TYPE_TEST(double, 2.)
  TYPE_TEST(char, 2);
  TYPE_TEST(uchar4, 2, 3, 4, 5);
#undef TYPE_TEST
}

//------------------------------------------------------------------------------

TEST(CudaArray2DTest, TestApplyOp) {
  {
    cua::test::CudaArray2DTestWrapper<cua::CudaArray2D<float>> wrapper;
    wrapper.CheckApplyOpConstant(0.f);
    wrapper.CheckApplyOpConstant(1.f);
    wrapper.CheckApplyOpLinear();
    wrapper.CheckApplyOpUpdate(3.f);
  }
  {
    cua::test::CudaArray2DTestWrapper<cua::CudaArray2D<float4>> wrapper;
    wrapper.CheckApplyOpConstant(float4{0.f});
    wrapper.CheckApplyOpConstant(float4{1.f, 2.f, 3.f, 4.f});
    wrapper.CheckApplyOpUpdate(float4{1.f, 2.f, 3.f, 4.f});
  }
  {
    cua::test::CudaArray2DTestWrapper<cua::CudaArray2D<unsigned int>> wrapper;
    wrapper.CheckApplyOpConstant(0);
    wrapper.CheckApplyOpConstant(1);
    wrapper.CheckApplyOpLinear();
    wrapper.CheckApplyOpUpdate(3);
  }
}

//------------------------------------------------------------------------------

}  // namespace
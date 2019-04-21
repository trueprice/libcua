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

#ifndef CUDA_RANDOM_TEST_H_
#define CUDA_RANDOM_TEST_H_

#include "cudaRandomStateArray2D.h"
#include "cudaRandomStateArray3D.h"

#include <vector>

#include "gtest/gtest.h"

#include "cudaArray2D.h"
#include "cudaSurface2D.h"
#include "cudaArray3D.h"
#include "cudaSurface3D.h"
#include "util.h"

//------------------------------------------------------------------------------

template <typename CudaArrayType>
class CudaRandomArray2DTest
    : public PrimitiveConverter<typename CudaArrayType::Scalar> {
 public:
  typedef typename CudaArrayType::Scalar Scalar;
  using PrimitiveConverter<Scalar>::AsScalar;

  //----------------------------------------------------------------------------

  CudaRandomArray2DTest(size_t width = 100, size_t height = 100,
                        size_t seed = 0)
      : array_(width, height),
        random_state_(
            (width + CudaArrayType::TILE_SIZE - 1) / CudaArrayType::TILE_SIZE,
            (height + CudaArrayType::TILE_SIZE - 1) / CudaArrayType::TILE_SIZE,
            seed) {}

  //----------------------------------------------------------------------------
  
  template <typename RandomFunction>
  void CheckFillRandom(RandomFunction func, bool check_result = false) {
    array_.Fill(AsScalar(1));
    CUDA_CHECK_ERROR
    array_.FillRandom(random_state_, func);
    CUDA_CHECK_ERROR

    // Check that numbers from the uniform distribution are in [0,1) -- assume
    // that the given random seed never results in exactly 0 being returned,
    // though.
    if (check_result) {
      std::vector<Scalar> result(array_.Size());
      array_.CopyTo(result.data());
      CUDA_CHECK_ERROR

      for (size_t y = 0; y < array_.Height(); ++y) {
        for (size_t x = 0; x < array_.Width(); ++x) {
          const size_t i = y * array_.Width() + x;
          EXPECT_TRUE(All(result[i] > AsScalar(0))) << "Coordinate: " << x
                                                    << " " << y << std::endl
                                                    << "Value: " << result[i];
          EXPECT_TRUE(All(result[i] < AsScalar(1))) << "Coordinate: " << x
                                                    << " " << y << std::endl
                                                    << "Value: " << result[i];
        }
      }
    }
  }

  //----------------------------------------------------------------------------

 private:
  CudaArrayType array_;
  cua::CudaRandomStateArray2D random_state_;
};

//------------------------------------------------------------------------------

template <typename CudaArrayType>
class CudaRandomArray3DTest
    : public PrimitiveConverter<typename CudaArrayType::Scalar> {
 public:
  typedef typename CudaArrayType::Scalar Scalar;
  using PrimitiveConverter<Scalar>::AsScalar;

  //----------------------------------------------------------------------------

  CudaRandomArray3DTest(size_t width = 100, size_t height = 100,
                        size_t depth = 100, size_t seed = 0)
      : array_(width, height, depth),
        random_state_(
            (width + CudaArrayType::TILE_SIZE - 1) / CudaArrayType::TILE_SIZE,
            (height + CudaArrayType::TILE_SIZE - 1) / CudaArrayType::TILE_SIZE,
            (depth + CudaArrayType::TILE_SIZE - 1) / CudaArrayType::TILE_SIZE,
            seed) {}

  //----------------------------------------------------------------------------
  
  template <typename RandomFunction>
  void CheckFillRandom(RandomFunction func, bool check_result = false) {
    array_.Fill(AsScalar(1));
    CUDA_CHECK_ERROR
    array_.FillRandom(random_state_, func);
    CUDA_CHECK_ERROR

    // Check that numbers from the uniform distribution are in [0,1) -- assume
    // that the given random seed never results in exactly 0 being returned,
    // though.
    if (check_result) {
      std::vector<Scalar> result(array_.Size());
      array_.CopyTo(result.data());
      CUDA_CHECK_ERROR

      for (size_t z = 0; z < array_.Height(); ++z) {
        for (size_t y = 0; y < array_.Height(); ++y) {
          for (size_t x = 0; x < array_.Width(); ++x) {
            const size_t i = (z * array_.Height() + y) * array_.Width() + x;
            EXPECT_TRUE(All(result[i] > AsScalar(0)))
                << "Coordinate: " << x << " " << y << " " << z << std::endl
                << "Value: " << result[i];
            EXPECT_TRUE(All(result[i] < AsScalar(1)))
                << "Coordinate: " << x << " " << y << " " << z << std::endl
                << "Value: " << result[i];
          }
        }
      }
    }
  }

  //----------------------------------------------------------------------------

 private:
  CudaArrayType array_;
  cua::CudaRandomStateArray3D random_state_;
};

//
// Due to the test implementation, our lambda functions need to be defined
// outside of the TEST().
//

template <typename TestType>
void TestUniform() {
  auto func = [] __device__(curandState_t * state) {
    return curand_uniform(state);
  };
  TestType().CheckFillRandom(func, true);
}

template <typename TestType>
void TestUniformDouble() {
  auto func = [] __device__(curandState_t * state) {
    return curand_uniform_double(state);
  };
  TestType().CheckFillRandom(func, true);
}

template <typename TestType>
void TestLogNormal(float mean, float stddev) {
  auto func = [=] __device__(curandState_t * state) {
    return curand_log_normal(state, mean, stddev);
  };
  TestType().CheckFillRandom(func);
}

template <typename TestType>
void TestUnsignedInt() {
  auto func = [] __device__(curandState_t * state) { return curand(state); };
  TestType().CheckFillRandom(func);
}

//------------------------------------------------------------------------------
//
// Test instances
//
//------------------------------------------------------------------------------

TEST(RandomTest, UnsignedInt2D) {
  TestUnsignedInt<CudaRandomArray2DTest<cua::CudaArray2D<unsigned int>>>();
  TestUnsignedInt<CudaRandomArray2DTest<cua::CudaSurface2D<unsigned int>>>();
}

TEST(RandomTest, UniformFloat2D) {
  TestUniform<CudaRandomArray2DTest<cua::CudaArray2D<float>>>();
  TestUniform<CudaRandomArray2DTest<cua::CudaSurface2D<float>>>();
}

TEST(RandomTest, UniformDouble2D) {
  TestUniformDouble<CudaRandomArray2DTest<cua::CudaArray2D<double>>>();
}

TEST(RandomTest, LogNormal2D) {
  TestLogNormal<CudaRandomArray2DTest<cua::CudaArray2D<float>>>(0.f, 1.f);
  TestLogNormal<CudaRandomArray2DTest<cua::CudaSurface2D<float>>>(0.f, 1.f);
  TestLogNormal<CudaRandomArray2DTest<cua::CudaArray2D<float>>>(2.f, 4.f);
  TestLogNormal<CudaRandomArray2DTest<cua::CudaSurface2D<float>>>(2.f, 4.f);
}

TEST(RandomTest, UnsignedInt3D) {
  TestUnsignedInt<CudaRandomArray3DTest<cua::CudaArray3D<unsigned int>>>();
  TestUnsignedInt<CudaRandomArray3DTest<cua::CudaSurface3D<unsigned int>>>();
}

TEST(RandomTest, UniformFloat3D) {
  TestUniform<CudaRandomArray3DTest<cua::CudaArray3D<float>>>();
  TestUniform<CudaRandomArray3DTest<cua::CudaSurface3D<float>>>();
}

TEST(RandomTest, UniformDouble3D) {
  TestUniformDouble<CudaRandomArray3DTest<cua::CudaArray3D<double>>>();
}

TEST(RandomTest, LogNormal3D) {
  TestLogNormal<CudaRandomArray3DTest<cua::CudaArray3D<float>>>(0.f, 1.f);
  TestLogNormal<CudaRandomArray3DTest<cua::CudaSurface3D<float>>>(0.f, 1.f);
  TestLogNormal<CudaRandomArray3DTest<cua::CudaArray3D<float>>>(2.f, 4.f);
  TestLogNormal<CudaRandomArray3DTest<cua::CudaSurface3D<float>>>(2.f, 4.f);
}

#endif  // CUDA_RANDOM_TEST_H_

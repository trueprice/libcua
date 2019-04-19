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

#ifndef TEST_UTIL_H_
#define TEST_UTIL_H_

/*
 * CUDA doesn't define standard operators for its vector datatypes.
 */

inline bool operator==(const uchar4 &a, const uchar4 &b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline bool operator==(const float2 &a, const float2 &b) {
  return a.x == b.x && a.y == b.y;
}

inline bool operator==(const float3 &a, const float3 &b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline bool operator==(const float4 &a, const float4 &b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

//------------------------------------------------------------------------------

inline float4 operator+(const float4 &a, const float4 &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

//------------------------------------------------------------------------------
// Error checking

inline void CudaCheckError(const char *filename, int line) {
  cudaDeviceSynchronize();
  const cudaError_t status = cudaPeekAtLastError();
  ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status) << " ("
                                 << filename << ":" << line << ")";
}

#define CUDA_CHECK_ERROR CudaCheckError(__FILE__, __LINE__);

#endif  // TEST_UTIL_H_

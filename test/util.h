// CudaArray: header-only library for interfacing with CUDA array-type objects
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

#ifndef TEST_UTIL_H_
#define TEST_UTIL_H_

//------------------------------------------------------------------------------
// Error-checking macro.

inline void CudaCheckError(const char *filename, int line) {
  cudaDeviceSynchronize();
  const cudaError_t status = cudaPeekAtLastError();
  ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status) << " ("
                                 << filename << ":" << line << ")";
}

#define CUDA_CHECK_ERROR CudaCheckError(__FILE__, __LINE__);

//------------------------------------------------------------------------------

/*
 * CUDA doesn't define standard operators for its vector datatypes, so here we
 * are.
 */

namespace {

typedef unsigned char uchar;

}  // namespace

#define DEFINE_FOR_ALL_DIMS(TYPE)   \
  DEFINE_FOR_TYPE_AND_DIM(TYPE, 2) \
  DEFINE_FOR_TYPE_AND_DIM(TYPE, 3) \
  DEFINE_FOR_TYPE_AND_DIM(TYPE, 4)

#define DEFINE_FOR_ALL_TYPES \
  DEFINE_FOR_ALL_DIMS(float)    \
  DEFINE_FOR_ALL_DIMS(char)     \
  DEFINE_FOR_ALL_DIMS(uchar)    \
  DEFINE_FOR_ALL_DIMS(short)    \
  DEFINE_FOR_ALL_DIMS(ushort)   \
  DEFINE_FOR_ALL_DIMS(int)      \
  DEFINE_FOR_ALL_DIMS(uint)

//------------------------------------------------------------------------------
// Unary operators (All() and Any()).

template <typename T>
inline bool All(const T &v) { return v != T{0}; }
template <typename T>
inline bool Any(const T &v) { return v != T{0}; }

template <>
inline bool All(const bool &v) { return v; }
template <>
inline bool Any(const bool &v) { return v; }

// Adds the z and w elements, if necessary.
#define CASE2(TYPE, OP)
#define CASE3(TYPE, OP) OP(v.z != TYPE{0})
#define CASE4(TYPE, OP) OP(v.z != TYPE{0}) OP(v.w != TYPE{0})

#define OPERATOR(NAME, OP, TYPE, DIM)                              \
  template <>                                                      \
  inline bool NAME(const TYPE##DIM &v) {                           \
    return (v.x != TYPE{0})OP(v.y != TYPE{0}) CASE##DIM(TYPE, OP); \
  }

#define DEFINE_FOR_TYPE_AND_DIM(TYPE, DIM) \
  OPERATOR(All, &&, TYPE, DIM) \
  OPERATOR(Any, ||, TYPE, DIM) \

DEFINE_FOR_ALL_TYPES

#undef CASE2
#undef CASE3
#undef CASE4
#undef OPERATOR
#undef DEFINE_FOR_TYPE_AND_DIM

//------------------------------------------------------------------------------
// bool operator==(a, b)

// Adds the z and w elements, if necessary.
#define CASE2
#define CASE3 &&(a.z == b.z)
#define CASE4 &&(a.z == b.z) && (a.w == b.w)

#define OPERATOR(TYPE, DIM)                                        \
  inline bool operator==(const TYPE##DIM &a, const TYPE##DIM &b) { \
    return a.x == b.x && a.y == b.y CASE##DIM;                     \
  }

#define DEFINE_FOR_TYPE_AND_DIM(TYPE, DIM) OPERATOR(TYPE, DIM)

DEFINE_FOR_ALL_TYPES

#undef CASE2
#undef CASE3
#undef CASE4
#undef OPERATOR
#undef DEFINE_FOR_TYPE_AND_DIM

//------------------------------------------------------------------------------
// Element-wise binary operators for vector-valued numeric types (a + b, etc.).

#define CASE2(TYPE, OP)
#define CASE3(TYPE, OP) , static_cast<TYPE>(a.z OP b.z)
#define CASE4(TYPE, OP) \
  , static_cast<TYPE>(a.z OP b.z), static_cast<TYPE>(a.w OP b.w)

#define OPERATOR(TYPE, DIM, OP)                                          \
  inline TYPE##DIM operator OP(const TYPE##DIM &a, const TYPE##DIM &b) { \
    return {static_cast<TYPE>(a.x OP b.x),                               \
            static_cast<TYPE>(a.y OP b.y) CASE##DIM(TYPE, OP)};          \
  }

#define DEFINE_FOR_TYPE_AND_DIM(TYPE, DIM) \
  OPERATOR(TYPE, DIM, +)                   \
  OPERATOR(TYPE, DIM, -)                   \
  OPERATOR(TYPE, DIM, *)                   \
  OPERATOR(TYPE, DIM, /)                   \
  OPERATOR(TYPE, DIM, >)                   \
  OPERATOR(TYPE, DIM, <)

DEFINE_FOR_ALL_TYPES

#undef CASE2
#undef CASE3
#undef CASE4
#undef OPERATOR
#undef DEFINE_FOR_TYPE_AND_DIM

//------------------------------------------------------------------------------
// Allow for safely casting primitives to vector-valued types -- duplicates the
// given primitive value to each vector element.

template <typename ScalarType>
struct PrimitiveConverter {
  template <typename T>
  static inline ScalarType AsScalar(T value) {
    return static_cast<ScalarType>(value);
  }
};

#define CASE2(TYPE)
#define CASE3(TYPE) , cast_value
#define CASE4(TYPE) , cast_value, cast_value
#define DEFINE_FOR_TYPE_AND_DIM(TYPE, DIM)              \
  template <>                                           \
  struct PrimitiveConverter<TYPE##DIM> {                \
    template <typename T>                               \
    static inline TYPE##DIM AsScalar(T value) {         \
      const TYPE cast_value = static_cast<TYPE>(value); \
      return {cast_value, cast_value CASE##DIM(TYPE)};  \
    }                                                   \
  };

DEFINE_FOR_ALL_TYPES

#undef CASE2
#undef CASE3
#undef CASE4
#undef DEFINE_FOR_TYPE_AND_DIM

//------------------------------------------------------------------------------

#undef DEFINE_FOR_ALL_DIMS
#undef DEFINE_FOR_ALL_TYPES

#endif  // TEST_UTIL_H_

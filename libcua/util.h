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

#ifndef LIBCUA_UTIL_H_
#define LIBCUA_UTIL_H_

#include <string>
#include <type_traits>

namespace cua {

namespace internal {

//------------------------------------------------------------------------------

// Return either the input argument, if it is not -1, or the current GPU.
inline int GetDevice(int device = -1) {
  if (device == -1) {
    cudaGetDevice(&device);
  }
  return device;
}

// Set the current device only if it is not already in use. This is to avoid
// triggering cudaErrorDeviceAlreadyInUse.
inline cudaError_t SetDevice(int device) {
  int current_device;
  cudaGetDevice(&current_device);
  if (device != current_device) {
    return cudaSetDevice(device);
  }
  return cudaSuccess;
}

//------------------------------------------------------------------------------

template <typename T>
inline std::string ArraySizeToString2D(const T &array) {
  return "(" + std::to_string(array.Width()) + ", " +
         std::to_string(array.Height()) + ")";
}

template <typename T>
inline std::string ArraySizeToString3D(const T &array) {
  return "(" + std::to_string(array.Width()) + ", " +
         std::to_string(array.Height()) + ", " + std::to_string(array.Depth()) +
         ")";
}

//------------------------------------------------------------------------------

template <typename T>
inline void CheckNotNull(const T *value) {
#ifndef LIBCUA_IGNORE_RUNTIME_EXCEPTIONS
  if (value == nullptr) {
    throw std::runtime_error("Value must not be null.");
  }
#endif
}

//------------------------------------------------------------------------------

template <typename T1, typename T2>
inline void CheckSameDevice(const T1 &array1, const T2 &array2) {
#ifndef LIBCUA_IGNORE_RUNTIME_EXCEPTIONS
  if (array1.Device() != array2.Device()) {
    throw std::runtime_error("Arrays are on different GPUs (" +
                             std::to_string(array1.Device()) + " vs " +
                             std::to_string(array2.Device()) + ").");
  }
#endif
}

//------------------------------------------------------------------------------

template <typename T1, typename T2>
inline void CheckCompatibleTypes(const T1 &array1, const T2 &array2) {
  static_assert(std::is_same<typename T1::Scalar, typename T2::Scalar>::value,
                "Arrays have different scalar types.");
}

//------------------------------------------------------------------------------

template <typename T1, typename T2>
inline void CheckSizeEqual2D(const T1 &array1, const T2 &array2) {
  CheckCompatibleTypes(array1, array2);
#ifndef LIBCUA_IGNORE_RUNTIME_EXCEPTIONS
  if (array1.Width() != array2.Width() || array1.Height() != array2.Height()) {
    throw std::runtime_error("Arrays have different sizes (" +
                             ArraySizeToString2D(array1) + " vs " +
                             ArraySizeToString2D(array2) + ").");
  }
#endif
}

template <typename T1, typename T2>
inline void CheckFlippedSizeEqual2D(const T1 &array1, const T2 &array2) {
  CheckCompatibleTypes(array1, array2);
#ifndef LIBCUA_IGNORE_RUNTIME_EXCEPTIONS
  if (array1.Width() != array2.Height() || array1.Height() != array2.Width()) {
    throw std::runtime_error("Arrays have incompatible sizes (" +
                             ArraySizeToString2D(array1) + " vs " +
                             ArraySizeToString2D(array2) + ").");
  }
#endif
}

template <typename T1, typename T2>
inline void CheckSizeEqual3D(const T1 &array1, const T2 &array2) {
  CheckCompatibleTypes(array1, array2);
#ifndef LIBCUA_IGNORE_RUNTIME_EXCEPTIONS
  if (array1.Width() != array2.Width() || array1.Height() != array2.Height() ||
      array1.Depth() != array2.Depth()) {
    throw std::runtime_error("Arrays have different sizes (" +
                             ArraySizeToString3D(array1) + " vs " +
                             ArraySizeToString3D(array2) + ").");
  }
#endif
}

//------------------------------------------------------------------------------

}  // namespace internal

}  // namespace cua

#endif  // LIBCUA_UTIL_H_

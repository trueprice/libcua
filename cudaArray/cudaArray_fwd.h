#ifndef CUDA_ARRAY_FWD_H_
// CudaArray: header-only library for interfacing with CUDA array-type objects
// Copyright (C) 2017  True Price <jtprice at cs.unc.edu>
//
// MIT License
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define CUDA_ARRAY_FWD_H_

/**
 * @namespace cua
 * @brief cudaArray namespace
 */
namespace cua {

template <typename T>
class CudaArray2D;

template <typename T>
class CudaArray3D;

class CudaRandomStateArray2D;

template <typename T>
class CudaSurface2D;

template <typename T>
class CudaTexture2D;

}  // namespace cua

#endif  // CUDA_ARRAY_FWD_H_

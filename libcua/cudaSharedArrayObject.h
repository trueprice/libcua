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

#ifndef CUDA_SHARED_ARRAY_OBJECT_H_
#define CUDA_SHARED_ARRAY_OBJECT_H_

#include <memory>

namespace cua {

/**
 * @class CudaSharedArrayObject
 * @brief Class for sharing a textures or surfaces among instances
 *
 * T: underlying datatype of the created array
 * CUDA_API_ObjType: cudaTextureObject_t or cudaSurfaceObject_t
 * CUDA_API_DestroyObj: function for destroying an object of type
 *   CUDA_API_ObjType
 */
template <typename T, typename CUDA_API_ObjType,
          cudaError_t CUDA_API_DestroyObj(CUDA_API_ObjType)>
class CudaSharedArrayObject {
 public:
  //----------------------------------------------------------------------------

  CudaSharedArrayObject() : count(std::shared_ptr<int>(new int(1))) {}

  //------------------------------------------------------------------------------

  __host__ __device__ CudaSharedArrayObject(const CudaSharedArrayObject &other)
      : dev_array(other.dev_array),
        cuda_api_obj(other.cuda_api_obj),
        count(nullptr) {
#ifdef __CUDA_ARCH__
#else
    count = other.count;
    ++(*count);
#endif
  }

  //------------------------------------------------------------------------------

  ~CudaSharedArrayObject() {  // host function only
    decrement_();
  }

  //------------------------------------------------------------------------------

  CudaSharedArrayObject &operator=(const CudaSharedArrayObject &other) {
    decrement_();

    dev_array = other.dev_array;
    cuda_api_obj = other.cuda_api_obj;
    count = other.count;
    ++(*count);

    return *this;
  }

  //------------------------------------------------------------------------------

  inline cudaArray *get_dev_array() const { return dev_array; }

  //------------------------------------------------------------------------------

  __device__ inline const CUDA_API_ObjType &get_cuda_api_object() const {
    return cuda_api_obj;
  }

  //------------------------------------------------------------------------------

 protected:
  inline void decrement_() {
    if (--(*count) == 0) {
      CUDA_API_DestroyObj(cuda_api_obj);
      cudaFreeArray(dev_array);
    }
  }

  //------------------------------------------------------------------------------

  cudaArray *dev_array;
  CUDA_API_ObjType cuda_api_obj;
  std::shared_ptr<int> count;  // monitor the number of instances
};

//------------------------------------------------------------------------------
//
// Instances -- these just define the creation routine
//
//------------------------------------------------------------------------------

// shared surface
template <typename T>
class CudaSharedSurfaceObject
    : public CudaSharedArrayObject<T, cudaSurfaceObject_t,
                                   cudaDestroySurfaceObject> {
 public:
  // layered: if true, creates an array of 2D arrays, rather than a 3D array
  CudaSharedSurfaceObject(const size_t width, const size_t height,
                          const size_t depth = 1, const bool layered = false)
      : CudaSharedArrayObject<T, cudaTextureObject_t,
                              cudaDestroySurfaceObject>() {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();

    // allocate either a 3D array, multiple 2D arrays, or a 2D array
    unsigned int cudaFlags = cudaArraySurfaceLoadStore;
    if (layered) {
      cudaFlags |= cudaArrayLayered;
    }

    if (depth > 1 || layered) {
      const cudaExtent dims = make_cudaExtent(width, height, depth);
      cudaMalloc3DArray(&this->dev_array, &channel_desc, dims, cudaFlags);
    } else {
      cudaMallocArray(&this->dev_array, &channel_desc, width, height,
                      cudaFlags);
    }

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = this->dev_array;

    cudaCreateSurfaceObject(&this->cuda_api_obj, &res_desc);
  }

  ~CudaSharedSurfaceObject() {}
};

//------------------------------------------------------------------------------

// shared texture
template <typename T>
class CudaSharedTextureObject
    : public CudaSharedArrayObject<T, cudaTextureObject_t,
                                   cudaDestroyTextureObject> {
 public:
  CudaSharedTextureObject(
      const size_t width, const size_t height,
      const cudaTextureFilterMode filterMode = cudaFilterModePoint,
      const cudaTextureAddressMode addressMode = cudaAddressModeBorder,
      const cudaTextureReadMode readMode = cudaReadModeElementType)
      : CudaSharedTextureObject(width, height, 1, filterMode, addressMode,
                                readMode) {}

  CudaSharedTextureObject(
      const size_t width, const size_t height, const size_t depth = 1,
      const cudaTextureFilterMode filterMode = cudaFilterModePoint,
      const cudaTextureAddressMode addressMode = cudaAddressModeBorder,
      const cudaTextureReadMode readMode = cudaReadModeElementType)
      : CudaSharedArrayObject<T, cudaTextureObject_t,
                              cudaDestroyTextureObject>() {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();

    if (depth > 1) {
      const cudaExtent dims = make_cudaExtent(width, height, depth);
      cudaMalloc3DArray(&this->dev_array, &channel_desc, dims);
    } else {
      cudaMallocArray(&this->dev_array, &channel_desc, width, height);
    }

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = this->dev_array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = addressMode;
    texDesc.addressMode[1] = addressMode;
    texDesc.addressMode[2] = addressMode;
    texDesc.filterMode = filterMode;
    texDesc.readMode = readMode;

    cudaCreateTextureObject(&this->cuda_api_obj, &res_desc, &texDesc, nullptr);
  }
};

}  // namespace cua

#endif  // CUDA_SHARED_ARRAY_OBJECT_H_

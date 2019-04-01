#ifndef CUDA_STRUCT_H_
#define CUDA_STRUCT_H_

#include <cuda.h>

#include <memory>

namespace cua {

// Copy assignment is shallow copy to facilitate passing around this object in
// host code.
template <typename StructT>
class CudaStruct {
 public:
  CudaStruct();
  __device__ inline StructT& get() const { return *data_ptr_; }
  __device__ inline StructT* getPtr() { return data_ptr_; }
  __host__ void CopyTo(StructT* host_struct);
  __host__ CudaStruct<StructT>& operator=(const StructT& host_struct);
  // __host__ __device__ CudaStruct<StructT>& operator=(const
  // CudaStruct<StructT>& other);
  __host__ __device__ CudaStruct(const CudaStruct<StructT>& other);

  ~CudaStruct();

 private:
  std::shared_ptr<StructT> shared_data_ptr_;
  StructT* data_ptr_;
};

template <typename StructT>
CudaStruct<StructT>::CudaStruct() {
  cudaMalloc(reinterpret_cast<void**>(&data_ptr_), sizeof(StructT));
  shared_data_ptr_ = std::shared_ptr<StructT>(data_ptr_, cudaFree);
}

template <typename StructT>
CudaStruct<StructT>::~CudaStruct() {
  shared_data_ptr_.reset();
  data_ptr_ = nullptr;
}

template <typename StructT>
void CudaStruct<StructT>::CopyTo(StructT* host_struct) {
  cudaMemcpy(reinterpret_cast<void*>(host_struct),
             reinterpret_cast<void*>(data_ptr_), sizeof(StructT),
             cudaMemcpyDeviceToHost);
}

template <typename StructT>
CudaStruct<StructT>& CudaStruct<StructT>::operator=(
    const StructT& host_struct) {
  cudaMemcpy(reinterpret_cast<void*>(data_ptr_),
             const_cast<void*>(reinterpret_cast<const void*>(&host_struct)),
             sizeof(StructT), cudaMemcpyHostToDevice);
  return *this;
}

/*
template <typename StructT>
CudaStruct<StructT>& CudaStruct<StructT>::operator=(
    const CudaStruct<StructT>& other) {
  if (this == &other) {
    return *this;
  } else {  // Do shallow copy.

    return *this;
  }
}
*/

template <typename StructT>
CudaStruct<StructT>::CudaStruct<StructT>(const CudaStruct<StructT>& other)
    : shared_data_ptr_(nullptr), data_ptr_(other.data_ptr_) {
#ifdef __CUDA_ARCH__
#else
  shared_data_ptr_ =
      other.shared_data_ptr_;  // Allow this only on the host device.
#endif
}

}  // namespace cua

#endif  // CUDA_STRUCT_H_

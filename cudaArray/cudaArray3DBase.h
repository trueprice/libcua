// Author: True Price <jtprice at cs.unc.edu>
// TODO: in the future, expand this class to support more CudaArray3D features
// (fill, fillRandom, etc.; no need for transpose, etc., though)

#ifndef CUDAARRAY3DBASE_H_
#define CUDAARRAY3DBASE_H_

namespace cua {

//
// kernel definitions
// TODO: once we have more kernel functions, move them to a separate file
//

//
// copy values of one surface to another, possibly with different datatypes
//
template <typename SrcCls, typename DstCls>
__global__ void CudaArray3DBase_copy_kernel(const SrcCls src, DstCls dst) {
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < src.get_width() && y < src.get_height() && z < src.get_depth()) {
    dst.set(x, y, z, (typename DstCls::Scalar)src.get(x, y, z));
  }
}

//
// arithmetic operations
// op: __device__ function mapping (x,y) -> CudaArrayClass::Scalar
//
template <typename CudaArrayClass, class Function>
__global__ void CudaArray3DBase_apply_op_kernel(CudaArrayClass mat,
                                                Function op) {
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < mat.get_width() && y < mat.get_height() && z < mat.get_depth()) {
    mat.set(x, y, z, op(x, y, z));
  }
}

//------------------------------------------------------------------------------

// Any derived class will need to declare
/*
 * template <typename T>
 * struct CudaArrayTraits<Derived<T>> {
 *   typedef T Scalar;
 * };
 */
template <typename Derived>
struct CudaArrayTraits;  // forward declaration

/**
 * @class CudaArray3DBase
 * @brief Base class for all 3D CudaArray-type objects.
 *
 * This class includes implementations for Copy, etc.
 * All derived classes need to define the following members:
 *
 * 1. copy constructor on host *and* device; use `#ifndef __CUDA_ARCH__` to
 *    perform host-specific instructions
 *    - `__host__ __device__ Derived(const Derived &other);`
 * 2. EmptyCopy(): to create a new array of the same size
 *    - `Derived EmptyCopy() const;`
 * 3. set(): write to array position (optional for readonly subclasses)
 *    - `__device__ inline void set(const size_t x, const size_t y,
 *                                  const size_t z, Scalar value);`
 * 4. get(): read from array position
 *    - `__device__
 *    inline Scalar get(const size_t x, const size_t y, const size_t z) const;`
 * 5. operator=(): suggested to have this for getting data from the CPU
 *   - `Derived &operator=(const Scalar *host_array);`
 * 6. CopyTo(): suggested to have this for getting data to the CPU
 *    - `void CopyTo(Scalar *host_array) const;`
 *
 * Also, any derived class will need to separately declare a CudaArrayTraits
 * struct instantiation with a "Scalar" member, e.g.,
 *
 *     template <typename T>
 *     struct CudaArrayTraits<Derived<T>> {
 *       typedef T Scalar;
 *     };
 */
template <typename Derived>
class CudaArray3DBase {
 public:
  //----------------------------------------------------------------------------
  // static class elements and typedefs

  /// datatype of the array
  typedef typename CudaArrayTraits<Derived>::Scalar Scalar;

  /// default block dimensions for general operations
  static const dim3 BLOCK_DIM;

  //----------------------------------------------------------------------------
  // constructors and derived()

  /**
   * Constructor.
   * @param width number of columns in the array, assuming a row-major array
   * @param height number of rows in the array, assuming a row-major array
   * @param block_dim default block size for CUDA kernel calls involving this
   *   object, i.e., the values for blockDim.x/y/z; note that the default grid
   *   dimension is computed automatically based on the array size
   * @param stream CUDA stream for this array object
   */
  CudaArray3DBase(const size_t width, const size_t height, const size_t depth,
                  const dim3 block_dim = CudaArray3DBase<Derived>::BLOCK_DIM,
                  const cudaStream_t stream = 0);  // default stream

  /**
   * Base-level copy constructor.
   * @param other array from which to copy array properties such as width and
   *   height
   */
  __host__ __device__ CudaArray3DBase(const CudaArray3DBase<Derived> &other)
      : width_(other.width_),
        height_(other.height_),
        depth_(other.depth_),
        block_dim_(other.block_dim_),
        grid_dim_(other.grid_dim_),
        stream_(other.stream_) {}

  /**
   * @returns a reference to the object cast to its dervied class type
   */
  __host__ __device__ Derived &derived() {
    return *reinterpret_cast<Derived *>(this);
  }

  /**
   * @returns a reference to the object cast to its dervied class type
   */
  __host__ __device__ const Derived &derived() const {
    return *reinterpret_cast<const Derived *>(this);
  }

  /**
   * Base-level assignment operator; merely copies size, thread dim, and stream
   * parameters. You will probably want to implement your own version of = and
   * call this one internally.
   * @param other the reference array
   * @return *this
   */
  CudaArray3DBase<Derived> &operator=(const CudaArray3DBase<Derived> &other);

  //----------------------------------------------------------------------------
  // general array operations that create a new object

  /**
   * @return a new copy of the current array.
   */
  Derived copy() const {
    Derived result = derived().emptyCopy();
    copy(result);
    return result;
  }

  //----------------------------------------------------------------------------
  // general array options that write to an existing object

  /**
   * Copy the current array to another array.
   * @ param other output array
   * @return other
   */
  template <typename OtherDerived>  // allow copies to other scalar types
  OtherDerived &copy(OtherDerived &other) const;

  //----------------------------------------------------------------------------
  // getters/setters

  __host__ __device__ inline size_t get_width() const { return width_; }
  __host__ __device__ inline size_t get_height() const { return height_; }
  __host__ __device__ inline size_t get_depth() const { return depth_; }

  __host__ __device__ inline dim3 get_block_dim() const { return block_dim_; }
  __host__ __device__ inline dim3 get_grid_dim() const { return grid_dim_; }

  inline void set_block_dim(const dim3 block_dim) {
    block_dim_ = block_dim;
    grid_dim_ = dim3((int)std::ceil(float(width_) / block_dim_.x),
                     (int)std::ceil(float(height_) / block_dim_.y),
                     (int)std::ceil(float(depth_) / block_dim_.z));
  }

  inline cudaStream_t get_stream() const { return stream_; }
  inline void set_stream(const cudaStream_t stream) { stream_ = stream; }

  //----------------------------------------------------------------------------
  // general array operations

  /**
   * Apply a general element-wise operation to the array. Here's a simple
   * example that uses a device-level C++11 lambda function to store the sum of
   * two arrays `arr1` and `arr2` into the output array:
   *
   *      // Array3DType arr1, arr2
   *      // Array3DType out
   *      out.apply_op([arr1, arr2] __device__(const size_t x, const size_t y,
   *                                           const size_t z) {
   *        return arr1.get(x, y, z) + arr2.get(x, y, z);  // => out(x, y, z)
   *      });
   *
   * @param op `__device__` function mapping `(x,y,z) -> CudaArrayClass::Scalar`
   * @param shared_mem_bytes if `op()` uses shared memory, the size of the
   *   shared memory space required
   */
  template <class Function>
  void apply_op(Function op, const size_t shared_mem_bytes = 0) {
    CudaArray3DBase_apply_op_kernel
        <<<grid_dim_, block_dim_, shared_mem_bytes, stream_>>>(derived(), op);
  }

  //----------------------------------------------------------------------------
  // protected class methods and fields

 protected:
  size_t width_, height_, depth_;

  dim3 block_dim_, grid_dim_;  // for calling kernels

  cudaStream_t stream_;  // the stream on the GPU in which the class kernels run
};

//------------------------------------------------------------------------------
//
// static member initialization
//
//------------------------------------------------------------------------------

template <typename Derived>
const dim3 CudaArray3DBase<Derived>::BLOCK_DIM = dim3(32, 8, 4);

//------------------------------------------------------------------------------
//
// public method implementations
//
//------------------------------------------------------------------------------

template <typename Derived>
CudaArray3DBase<Derived>::CudaArray3DBase<Derived>(const size_t width,
                                                   const size_t height,
                                                   const size_t depth,
                                                   const dim3 block_dim,
                                                   const cudaStream_t stream)
    : width_(width), height_(height), depth_(depth), stream_(stream) {
  set_block_dim(block_dim);
}

//------------------------------------------------------------------------------

template <typename Derived>
CudaArray3DBase<Derived> &CudaArray3DBase<Derived>::operator=(
    const CudaArray3DBase<Derived> &other) {
  if (this == &other) {
    return *this;
  }

  width_ = other.width_;
  height_ = other.height_;
  depth_ = other.depth_;

  block_dim_ = other.block_dim_;
  grid_dim_ = other.grid_dim_;
  stream_ = other.stream_;

  return *this;
}

//------------------------------------------------------------------------------

template <typename Derived>
template <typename OtherDerived>
OtherDerived &CudaArray3DBase<Derived>::copy(OtherDerived &other) const {
  if (this != &other) {
    if (width_ != other.width_ || height_ != other.height_ ||
        depth_ != other.depth_) {
      other = derived().emptyCopy();
    }

    CudaArray3DBase_copy_kernel<<<grid_dim_, block_dim_>>>(derived(), other);
  }

  return other;
}

} // namespace cua

#endif // CUDAARRAY3DBASE_H_

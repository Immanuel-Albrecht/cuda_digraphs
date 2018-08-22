#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>



// check kernel launch error
#define cudaCheckErr(errorMessage) {    \
  cudaError_t err = cudaGetLastError(); \
  if(cudaSuccess != err){               \
    fprintf(stderr, "%s(%i) : %s : %s.\n", __FILE__, __LINE__, errorMessage, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                 \
}}

// launch cuda kernel (VS15 turns '<<<' into '< <<'; workaround)
#define cudaKernel(global_fn,N_BLOCKS,N_THREADS) (global_fn)<<<(N_BLOCKS),(N_THREADS)>>>

// check runtime call error
#define cudaSafeCall(call) {  \
  cudaError err = call;       \
  if(cudaSuccess != err){     \
    fprintf(stderr, "%s(%i) : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
    exit(EXIT_FAILURE);       \
}}

// check runtime call error, but do not exit program [CUDA throws errors if thread calculations take too long on win10]
#define cudaSafeCallNoExit(call) do {  \
  cudaError err = call;       \
  if(cudaSuccess != err){     \
    fprintf(stderr, "%s(%i) : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
}} while (0)


// allocate a vector on device
#define cudaSafeMallocVector(ptr, len) cudaSafeCall( cudaMalloc(&(ptr), sizeof( (ptr) [0])* len) )

// allocate some plain old data structure on device
#define cudaSafeMallocPOD(ptr) cudaSafeCall( cudaMalloc(&(ptr), sizeof( (ptr) [0]) ) )


 
// move vector from host to device
#define cudaSafeVectorToDev(dev_dst, src, len) { \
  static_assert( sizeof( (dev_dst)[0]) == sizeof( (src)[0]), "cudaSafeVectorToDev source and destination arrays have different base type sizes!"); \
  cudaSafeCall( cudaMemcpy( (dev_dst), (src), sizeof((dev_dst)[0])*len, cudaMemcpyHostToDevice) ); \
}

// move vector from device to host
#define cudaSafeVectorToHost(dst, dev_src, len) { \
  static_assert( sizeof( (dst)[0]) == sizeof( (dev_src)[0]), "cudaSafeVectorToHost source and destination arrays have different base type sizes!"); \
  cudaSafeCall( cudaMemcpy( (dst), (dev_src), sizeof((dev_src)[0])*len, cudaMemcpyDeviceToHost) ); \
}

#define cudaSafeDeviceVectorMemcpy(dev_dst, dev_src, len) { \
  static_assert( sizeof( (dev_dst)[0]) == sizeof( (dev_src)[0]), "cudaSafeDeviceVectorMemcpy source and destination arrays have different base type sizes!"); \
  cudaSafeCall( cudaMemcpy( (dev_dst), (dev_src), sizeof((dev_src)[0])*len, cudaMemcpyDeviceToDevice) ); \
}


/* INDEXING HELPERS FROM THE CHEAT SHEET*/

__device__ int getGlobalIdx_1D_3D();
__device__ int getGlobalIdx_2D_3D();
__device__ int getGlobalIdx_3D_3D();
__device__ int getGlobalIdx_1D_2D();
__device__ int getGlobalIdx_2D_2D();
__device__ int getGlobalIdx_3D_2D();
__device__ int getGlobalIdx_1D_1D();
__device__ int getGlobalIdx_2D_1D();
__device__ int getGlobalIdx_3D_1D();
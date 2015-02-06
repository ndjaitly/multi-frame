#include <cuda.h>  
#include <cublas.h>
extern "C"
{
inline bool check_cublas_error() {
    cublasStatus status = cublasGetError();

    return status != CUBLAS_STATUS_SUCCESS;
}

inline bool checkCUDAError() {
    cudaError_t err = cudaGetLastError();

    /* 
       Giving issues compiling in the latest version
    if (cudaSuccess != err)
        printf("%s\n", cudaGetErrorString( err));
    */
    return cudaSuccess != err;
}

extern const char* get_last_cuda_error() {
    cudaError_t err = cudaGetLastError();

    return cudaGetErrorString( err);
}
}

#include <cuda.h>  
#include "Convolutions_kernels.cuh"
#include "cudamat.cuh" 

extern "C" 
{

inline bool checkCUDAError() 
{
    cudaError_t err = cudaGetLastError();

    //if (cudaSuccess != err)
        //printf("%s\n", cudaGetErrorString( err));
    return cudaSuccess != err;
}

extern int ShiftedConvolution(cudamat *signal1, cudamat *signal2, cudamat *target, int kernelWidth, 
    cudamat *scratchPad)
{

   if (!signal1->on_device || !target->on_device || !signal2->on_device || !scratchPad->on_device)
      return ERROR_NOT_ON_DEVICE;

   if (signal1->size[0] != 1 && signal1->size[1] != 1)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   int signalLength = signal1->size[0] * signal1->size[1] ; 
   int numKernels = signal2->size[1] ; 

   if (signal2->size[0] != signalLength)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   if (target->size[0] != kernelWidth || target->size[1] != numKernels)
      return ERROR_INCOMPATIBLE_DIMENSIONS;

   if (signal1->is_trans)
      return ERROR_TRANSPOSED;
   if (signal2->is_trans)
      return ERROR_TRANSPOSED;
   if (target->is_trans)
      return ERROR_TRANSPOSED;

   // Do calculation on device:  
   int numThreadsPerBlock = 256 ;  
   const int numPtsPerBlock = 512 ;

   int numBlocksPerKernel = signalLength/numPtsPerBlock + (signalLength%numPtsPerBlock == 0 ? 0:1); 
   if (scratchPad->size[0]*scratchPad->size[1] < kernelWidth*numBlocksPerKernel*numKernels)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;


   int sharedMemSize = 4*(2*numPtsPerBlock+kernelWidth-1) ;
   if (sharedMemSize > 16*1024)
      throw "Specified parameters require kernel with shared memory greater than 16KB. Exiting" ; 
   dim3 gDim(numKernels, numBlocksPerKernel, 1) ; 
   KernPartialConvolve<<<gDim,numThreadsPerBlock,sharedMemSize>>>(signal1->data_device, 
                                                                  signal2->data_device, 
                                                                  scratchPad->data_device,
                                                                  signalLength, 
                                                                  kernelWidth, 
                                                                  numPtsPerBlock) ; 

   dim3 gDimSum(numKernels, kernelWidth,1) ; 
   KernPartialConvolveSum<<<gDimSum, numThreadsPerBlock, sizeof(float)*numThreadsPerBlock>>>(scratchPad->data_device,
			target->data_device, kernelWidth, numBlocksPerKernel, numKernels) ; 

   cudaThreadSynchronize();

   if (checkCUDAError())
      return CUDA_ERROR;
   return 0 ; 
}

// Use this convolution only when kernelWidth is small compared to signalLength because it 
// does one convolution per thread. If both are long, consider coding (/using NVIDIA's fft sample)
// with fft coefficient products.
extern int Convolve(cudamat *signal, cudamat *kernels, cudamat *target)
{
   if (!signal->on_device || !target->on_device || !kernels->on_device)
      return ERROR_NOT_ON_DEVICE;

   if (signal->size[0] != 1 && signal->size[1] != 1)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   int signalLength = signal->size[0] * signal->size[1] ;
   int kernelWidth = kernels->size[0] ; 
   int numKernels = kernels->size[1] ; 

   if (target->size[0] != signalLength ||  target->size[1] != numKernels)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;


   if (signal->is_trans)
      return ERROR_TRANSPOSED ;
   if (kernels->is_trans)
      return ERROR_TRANSPOSED ;
   if (target->is_trans)
      return ERROR_TRANSPOSED ;


   // Do calculation on device:  
   int block_size = 128 ;  
   int numPtsPerBlock = 128 ; 
   int numBlocksPerSignal = signalLength/numPtsPerBlock + (signalLength%numPtsPerBlock == 0 ? 0:1); 
   dim3 gridD(numKernels, numBlocksPerSignal,1) ; 
   int sharedMemSize = sizeof(float)*((numPtsPerBlock+kernelWidth-1) + kernelWidth) ; 

   KernConvolve <<< gridD, block_size, sharedMemSize >>>(signal->data_device, 
                                          kernels->data_device, 
                                          target->data_device, 
                                          signalLength, 
                                          kernelWidth, 
                                          numPtsPerBlock) ; 

   cudaThreadSynchronize();

   if (checkCUDAError())
      return CUDA_ERROR;

   return 0 ; 

}

extern int ReverseConvolve(cudamat *convolvedSignals, cudamat *kernels, cudamat * reverseConvolvedSignals)
{
   int kernelWidth = kernels->size[0] ; 
   int numKernels = kernels->size[1] ; 
   int signalLength = convolvedSignals->size[0] ;

   if (!convolvedSignals->on_device || !kernels->on_device || !reverseConvolvedSignals->on_device)
      return ERROR_NOT_ON_DEVICE;

   if (reverseConvolvedSignals->size[0] != signalLength || reverseConvolvedSignals->size[1] != numKernels)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   if (convolvedSignals->size[1] !=  numKernels)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;


   // Do calculation on device:  
   int numThreadsPerBlock = 32 ;  
   int numPtsPerThread = 1 ; 
   int numPtsPerBlock = numThreadsPerBlock*numPtsPerThread ; 
   int numBlocks = signalLength/numPtsPerBlock + (signalLength%numPtsPerBlock == 0 ? 0:1); 
   dim3 gridD(numKernels, numBlocks,1) ; 

   KernReverseConvolve <<< gridD, numThreadsPerBlock >>>(convolvedSignals->data_device, 
                                                     kernels->data_device, 
                                                     reverseConvolvedSignals->data_device, 
                                                     signalLength, kernelWidth, numKernels, 
                                                     numPtsPerBlock, numPtsPerThread) ; 

   cudaThreadSynchronize();

   if (checkCUDAError())
      return CUDA_ERROR;

   return 0 ; 
}

extern int Reconstruct(cudamat *convolvedSignals, 
                       cudamat *kernels, 
                       cudamat *reverseConvolvedSignals, 
                       cudamat *reconstruction)
{
   int kernelWidth = kernels->size[0] ; 
   int numKernels = kernels->size[1] ; 
   int signalLength = convolvedSignals->size[0] ;

   if (!convolvedSignals->on_device || !kernels->on_device || !reverseConvolvedSignals->on_device)
      return ERROR_NOT_ON_DEVICE;

   if (reverseConvolvedSignals->size[0]*reverseConvolvedSignals->size[1] < signalLength * numKernels)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   if (convolvedSignals->size[1] !=  numKernels)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;


   // Do calculation on device:  
   int numThreadsPerBlock = 256 ;  
   int numPtsPerThread = 1 ; 
   int numPtsPerBlock = numThreadsPerBlock*numPtsPerThread ; 
   int numBlocks = signalLength/numPtsPerBlock + (signalLength%numPtsPerBlock == 0 ? 0:1); 
   dim3 gridD(numKernels, numBlocks,1) ; 

   KernReverseConvolve <<< gridD, numThreadsPerBlock >>>(convolvedSignals->data_device, 
                                                     kernels->data_device, 
                                                     reverseConvolvedSignals->data_device, 
                                                     signalLength, kernelWidth, numKernels, 
                                                     numPtsPerBlock, numPtsPerThread) ; 

   KernAddSignals <<< numBlocks, numThreadsPerBlock >>>(reverseConvolvedSignals->data_device, 
                                                    reconstruction->data_device,
                                                    signalLength, 
                                                    numKernels, 
                                                    numPtsPerBlock, 
                                                    numPtsPerThread) ; 

   cudaThreadSynchronize();

   if (checkCUDAError())
      return CUDA_ERROR;

   return 0 ; 
}

}

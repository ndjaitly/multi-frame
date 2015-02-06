#include <stdio.h>
#include <cuda.h>  
#include "cudamat.cuh" 
#include "SoftMax_kernels.cuh"

extern "C" 
{
inline bool checkCUDAError() 
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
        printf("%s\n", cudaGetErrorString( err));
    return cudaSuccess != err;
}

extern int CumulativeSum(cudamat *probabilities, cudamat *cumSum, int softMaxWidth)
{

   if (!probabilities->on_device || !cumSum->on_device)
      return ERROR_NOT_ON_DEVICE;

   int signalLength = probabilities->size[0] ; 
   int numFeatures = probabilities->size[1] ; 

   if (cumSum->size[0] != signalLength || cumSum->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   int numThreadsPerBlock = 64 ;
   int numTotalSoftMaxes = (signalLength/softMaxWidth) + (signalLength % softMaxWidth == 0 ? 0:1) ; 

   dim3 gridD(numFeatures, numTotalSoftMaxes,1) ;
   int sharedMem = sizeof(float)*(2*softMaxWidth) ; 

   kCumulativeSum <<< gridD, numThreadsPerBlock, sharedMem >>> 
                                            (probabilities->data_device, 
                                             cumSum->data_device, 
                                             softMaxWidth, 
                                             signalLength) ;
   cudaThreadSynchronize();

   if (checkCUDAError())
      return CUDA_ERROR;
   return 0 ; 
}

extern int MultinomialSamples(cudamat *unifRandNums, 
                              cudamat *probabilities, 
                              cudamat *samples, 
                              int softMaxWidth, 
                              int startShift)
{

   if (!probabilities->on_device || !samples->on_device || !unifRandNums->on_device)
      return ERROR_NOT_ON_DEVICE;

   int signalLength = probabilities->size[0] ; 
   int numFeatures = probabilities->size[1] ; 

   if (samples->size[0] != signalLength || samples->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   int numThreadsPerBlock = 128 ;
   int numTotalSoftMaxes = ((signalLength-startShift)/softMaxWidth) + ((signalLength-startShift) % softMaxWidth == 0 ? 0:1) ; 

   if (unifRandNums->size[0] * unifRandNums->size[1] < numTotalSoftMaxes*numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ; 

   dim3 gridD(numFeatures, numTotalSoftMaxes,1) ;
   int sharedMem = sizeof(float)*(2*softMaxWidth) ; 


   kMultinomialSample <<< gridD, numThreadsPerBlock, sharedMem >>>(
                                                               unifRandNums->data_device,
                                                               probabilities->data_device, 
                                                               samples->data_device, 
                                                               softMaxWidth, 
                                                               startShift,
                                                               signalLength
                                                              ) ;
   cudaThreadSynchronize();

   if (checkCUDAError())
      return CUDA_ERROR;
   return 0 ; 
}


extern int SoftMaxStackApproxWithPositionBiases(
                                           cudamat *activations, 
                                           cudamat *probabilities, 
                                           cudamat *stdevs, 
                                           cudamat *featureBiases, 
                                           cudamat *positionBiases, 
                                           int softMaxWidth
                                               )
{


   if (!activations->on_device || !probabilities->on_device || !stdevs->on_device || !featureBiases->on_device
        || !positionBiases->on_device)
      return ERROR_NOT_ON_DEVICE;

   int signalLength = activations->size[0] ; 
   int numFeatures = activations->size[1] ; 

   if (probabilities->size[0] != signalLength || probabilities->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   if (stdevs->size[0] != signalLength || stdevs->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   // feature biases much be a row or column vector
   if (featureBiases->size[0] != 1 && featureBiases->size[1] != 1)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   if (featureBiases->size[0]*featureBiases->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   // position biases much be a row or column vector
   if (positionBiases->size[0] != 1 && positionBiases->size[1] != 1)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;
   
   if (positionBiases->size[0]*positionBiases->size[1] != softMaxWidth)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   int numThreadsPerBlock = 128 ; 
   int numTotalSoftMaxes = (signalLength/softMaxWidth) + (signalLength % softMaxWidth == 0 ? 0:1) ; 

   dim3 gridD(numFeatures, numTotalSoftMaxes,1) ;
   int sharedMem = sizeof(float)*(softMaxWidth + numThreadsPerBlock) ; 
   kSoftMaxStackApproxWithPositionBiases <<< gridD, numThreadsPerBlock, sharedMem >>>
                                       (
                                        activations->data_device, 
                                        probabilities->data_device, 
                                        stdevs->data_device, 
                                        featureBiases->data_device, 
                                        positionBiases->data_device, 
                                        softMaxWidth, 
                                        signalLength
                                       ) ;
   cudaThreadSynchronize();

   if (checkCUDAError())
      return CUDA_ERROR;
   return 0 ; 
}


extern int SoftMaxReluWithPositionBiases(
                                         cudamat *activations, 
                                         cudamat *probabilities, 
                                         cudamat *meanValues, 
                                         cudamat *featureStdevs, 
                                         cudamat *featureBiases, 
                                         cudamat *positionBiases, 
                                         int softMaxWidth,
                                         int shift
                                        )
{


   if (!activations->on_device || !probabilities->on_device || !featureBiases->on_device
        || !positionBiases->on_device || !meanValues->on_device || !featureStdevs->on_device)
      return ERROR_NOT_ON_DEVICE;

   int signalLength = activations->size[0] ; 
   int numFeatures = activations->size[1] ; 

   if (probabilities->size[0] != signalLength || probabilities->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   if (meanValues->size[0] != signalLength || meanValues->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   if (featureStdevs->size[0] != signalLength || featureStdevs->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   // feature biases much be a row or column vector
   if (featureBiases->size[0] != 1 && featureBiases->size[1] != 1)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   if (featureBiases->size[0]*featureBiases->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   // position biases much be a row or column vector
   if (positionBiases->size[0] != 1 && positionBiases->size[1] != 1)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;
   
   if (positionBiases->size[0]*positionBiases->size[1] != softMaxWidth)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   int numThreadsPerBlock = 128 ; 
   int numTotalSoftMaxes = ((signalLength-shift)/softMaxWidth) + ((signalLength-shift) % softMaxWidth == 0 ? 0:1) ; 
   float minExpForSum = -20.0 ; 

   dim3 gridD(numFeatures, numTotalSoftMaxes,1) ;
   int sharedMem = sizeof(float)*(2*softMaxWidth + numThreadsPerBlock) ; 

   kSoftMaxReluWithPositionBiases <<< gridD, numThreadsPerBlock, sharedMem >>>
                                       (
                                        activations->data_device, 
                                        probabilities->data_device, 
                                        meanValues->data_device, 
                                        featureStdevs->data_device, 
                                        featureBiases->data_device,
                                        positionBiases->data_device,
                                        softMaxWidth,
                                        shift,
                                        signalLength,
                                        minExpForSum
                                       ) ;

   cudaThreadSynchronize();

   if (checkCUDAError())
      return CUDA_ERROR;
   return 0 ; 
}

extern int SoftMaxWithOffAndPositionBiases(cudamat *activations, 
                                           cudamat *probabilities, 
                                           cudamat *featureBiases, 
                                           cudamat *positionBiases, 
                                           int softMaxWidth)
{


   if (!activations->on_device || !probabilities->on_device || !featureBiases->on_device
        || !positionBiases->on_device)
      return ERROR_NOT_ON_DEVICE;

   int signalLength = activations->size[0] ; 
   int numFeatures = activations->size[1] ; 

   if (probabilities->size[0] != signalLength || probabilities->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   // feature biases much be a row or column vector
   if (featureBiases->size[0] != 1 && featureBiases->size[1] != 1)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   if (featureBiases->size[0]*featureBiases->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   // position biases much be a row or column vector
   if (positionBiases->size[0] != 1 && positionBiases->size[1] != 1)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;
   
   if (positionBiases->size[0]*positionBiases->size[1] != softMaxWidth)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   int numThreadsPerBlock = 128 ; 
   int numTotalSoftMaxes = (signalLength/softMaxWidth) + (signalLength % softMaxWidth == 0 ? 0:1) ; 

   dim3 gridD(numFeatures, numTotalSoftMaxes,1) ;
   int sharedMem = sizeof(float)*(softMaxWidth + numThreadsPerBlock) ; 
   kSoftMaxWithOffAndPositionBiases <<< gridD, numThreadsPerBlock, sharedMem >>>
                                       (
                                        activations->data_device, 
                                        probabilities->data_device, 
                                        featureBiases->data_device, 
                                        positionBiases->data_device, 
                                        softMaxWidth, 
                                        signalLength
                                       ) ;
   cudaThreadSynchronize();

   if (checkCUDAError())
      return CUDA_ERROR;
   return 0 ; 
}

extern int SoftMaxWithOff(cudamat *activations, cudamat *probabilities, cudamat *featureBiases, int softMaxWidth)
{


   if (!activations->on_device || !probabilities->on_device || !featureBiases->on_device)
      return ERROR_NOT_ON_DEVICE;

   int signalLength = activations->size[0] ; 
   int numFeatures = activations->size[1] ; 

   if (probabilities->size[0] != signalLength || probabilities->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   if (featureBiases->size[0]*featureBiases->size[1] != numFeatures)
      return ERROR_INCOMPATIBLE_DIMENSIONS ;

   int numThreadsPerBlock = 64 ; 
   int numTotalSoftMaxes = (signalLength/softMaxWidth) + (signalLength % softMaxWidth == 0 ? 0:1) ; 
   int numPtsPerThread = softMaxWidth/numThreadsPerBlock ; 
   if (numPtsPerThread*numThreadsPerBlock < softMaxWidth)
      numPtsPerThread++ ; 

   dim3 gridD(numFeatures, numTotalSoftMaxes,1) ;
   int sharedMem = sizeof(float)*(softMaxWidth + numThreadsPerBlock) ; 
   kSoftMaxWithOff <<< gridD, numThreadsPerBlock, sharedMem >>>
                                       (activations->data_device, probabilities->data_device, 
                                        featureBiases->data_device, softMaxWidth, signalLength, 
                                        numPtsPerThread) ;
   cudaThreadSynchronize();

   if (checkCUDAError())
      return CUDA_ERROR;
   return 0 ; 
}

}

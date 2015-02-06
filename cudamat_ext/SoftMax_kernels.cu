#include <cuda.h>  
#include <float.h>
const float MAX_EXP = 80. ; 

__global__ void kCumulativeSum(
                               float *probabilities, 
                               float *sumProbabilities, 
                               int softMaxWidth, 
                               int signalLength
                              )
{
   extern __shared__ float sData[] ; 
   float *dataPrev = (float *) &sData[0] ; 
   float *dataNext = (float *) &sData[softMaxWidth] ; 

   // first copy data for current softMax
   int signalIndex = signalLength * blockIdx.x ;
   int dataIndex = blockIdx.y * softMaxWidth + threadIdx.x ;

   int maxLength = min(softMaxWidth, signalLength-dataIndex+threadIdx.x) ; 
   for (int index = threadIdx.x ; index < maxLength && dataIndex < signalLength ; 
        index+= blockDim.x, dataIndex += blockDim.x)
   {
      dataPrev[index] = probabilities[signalIndex+dataIndex] ; 
   }
   __syncthreads() ; 

   // Now compute cumulative sum
   dataIndex = blockIdx.y * softMaxWidth  + threadIdx.x ;
   for (int round = 1 ; round <  maxLength ; round = round<<1)
   {
      for (int index = threadIdx.x ; index < maxLength ; index+= blockDim.x)
      {
         float val = dataPrev[index] ; 
         if (index >= round)
            val += dataPrev[index-round] ;
         dataNext[index] = val ; 
      }
      float *temp = dataPrev ; 
      dataPrev = dataNext ; 
      dataNext = temp ; 
      __syncthreads() ; 
   }

   // Write out the data.
   for (int index = threadIdx.x ; index < maxLength && dataIndex < signalLength ; 
        index+= blockDim.x, dataIndex += blockDim.x)
   {
      sumProbabilities[signalIndex+dataIndex] = dataPrev[index] ;
   }
}

__global__ void kMultinomialSample(
                                   float *unifRandNums, 
                                   float *probabilities, 
                                   float *samples, 
                                   int softMaxWidth, 
                                   int startShift,
                                   int signalLength
                                  )
{
   extern __shared__ float sData[] ; 
   float *dataPrev = (float *) &sData[0] ; 
   float *dataNext = (float *) &sData[softMaxWidth] ; 


   // first copy data for current softMax
   int signalIndex = signalLength * blockIdx.x ;
   int dataIndex = blockIdx.y * softMaxWidth + threadIdx.x + startShift ;

   float randNum = unifRandNums[blockIdx.x*gridDim.y + blockIdx.y] ; 

   int maxLength = min(softMaxWidth, signalLength-dataIndex+threadIdx.x) ; 
   for (int index = threadIdx.x ; index < maxLength && dataIndex < signalLength ; 
        index+= blockDim.x, dataIndex += blockDim.x)
   {
      dataPrev[index] = probabilities[signalIndex+dataIndex] ; 
   }
   __syncthreads() ; 

   // Now compute cumulative sum
   dataIndex = blockIdx.y * softMaxWidth  + threadIdx.x + startShift ;
   for (int round = 1 ; round <  maxLength ; round = round<<1)
   {
      for (int index = threadIdx.x ; index < maxLength ; index+= blockDim.x)
      {
         float val = dataPrev[index] ; 
         if (index >= round)
            val += dataPrev[index-round] ;
         dataNext[index] = val ; 
      }
      float *temp = dataPrev ; 
      dataPrev = dataNext ; 
      dataNext = temp ; 
      __syncthreads() ; 
   }

   // Find the appropriate index where cumulative[i-1]<= r < cumulative[i]
   for (int index = threadIdx.x ; index < maxLength && dataIndex < signalLength ; 
        index+= blockDim.x, dataIndex += blockDim.x)
   {
      if (index == 0)
      {
         if (randNum < dataPrev[index])
            samples[signalIndex+dataIndex] = 1 ;
         else
            samples[signalIndex+dataIndex] = 0 ;

      }
      else
      {
         if (randNum >= dataPrev[index-1] && randNum < dataPrev[index])
            samples[signalIndex+dataIndex] = 1 ;
         else
            samples[signalIndex+dataIndex] = 0 ;
      }
   }
}

__global__ void kSoftMaxStackApproxWithPositionBiases(
                                               float *activations, 
                                               float *probabilities, 
                                               float *stdevs, 
                                               float *featureBiases,
                                               float *positionBiases,
                                               int softMaxWidth, 
                                               int signalLength
                                               )
{
   extern __shared__ float sData[] ; 
   float *arrData = (float *) &sData[0] ; 
   float *threadStores = (float *) &sData[softMaxWidth] ; 

   int signalIndex = blockIdx.x*signalLength ; 
   int dataIndex = blockIdx.y*softMaxWidth + threadIdx.x ; 
  
   float maxVal = -FLT_MAX ;  
   for (int index = threadIdx.x ; index < softMaxWidth && dataIndex < signalLength ; 
            index += blockDim.x, dataIndex += blockDim.x)
   {
      arrData[index] = activations[signalIndex+dataIndex] + positionBiases[index] ; 
      if (maxVal < arrData[index])
         maxVal = arrData[index] ; 
   }
   threadStores[threadIdx.x] = maxVal ; 
   __syncthreads() ; 

   // do a reduction to find the max of all maxes. 
   for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
   {
      if (threadIdx.x < s)
      {
         threadStores[threadIdx.x] = 
              threadStores[threadIdx.x+s]*(threadStores[threadIdx.x + s] >=threadStores[threadIdx.x]) + 
              threadStores[threadIdx.x] * (threadStores[threadIdx.x] > threadStores[threadIdx.x+s])  ;
      }
      __syncthreads();
   }

   // now we have max. Lets subract it from all elements, and compute intermediate logSumExp over all elements 
   // a thread is responsible for
   float bias = featureBiases[blockIdx.x] ; 
   maxVal = threadStores[0] ; 

   __syncthreads();

   float sumExp = 0. ;

   dataIndex = blockIdx.y*softMaxWidth + threadIdx.x ; 
  
   for (int index = threadIdx.x ; index < softMaxWidth && dataIndex < signalLength ; 
            index += blockDim.x, dataIndex += blockDim.x)
   {
      arrData[index] = arrData[index] - maxVal ; 
      sumExp += __expf(arrData[index]) ; 
   }

   threadStores[threadIdx.x] = sumExp ;
   __syncthreads() ; 

   // compute normalization constant over sumExp by summing together all the intermediate values.
   for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
   {
       if (threadIdx.x < s)
          threadStores[threadIdx.x] += threadStores[threadIdx.x + s];
       __syncthreads();
   }
   __syncthreads() ; 

   sumExp = threadStores[0]  ; 

   float maxValNew = fmaxf(maxVal, bias) ;
  
   bias = bias - maxValNew ;  
   float reluSumExp = sumExp * __expf(maxVal-maxValNew) + __expf(bias) ;

   // notice, reusing maxValNew and maxVal variables for new causes.

   // maxValNew is the negative delta from the previous one.
   maxValNew = maxVal-maxValNew ; 
   // maxVal is -log(exp(biases)/(exp(biases)+sum(exp(x_i))))
   maxVal = -bias + __logf(reluSumExp) ;


   // write out probabilities and standard deviations
   dataIndex = blockIdx.y*softMaxWidth + threadIdx.x ; 
   for (int index = threadIdx.x ; index < softMaxWidth && dataIndex < signalLength ; 
            index += blockDim.x, dataIndex += blockDim.x)
   {
      float r_x_i = __expf(arrData[index])/ sumExp ; 
      float relu_r_x_i = __expf(arrData[index]+maxValNew)/ reluSumExp ;  
      probabilities[signalIndex+dataIndex]  = r_x_i * maxVal ; 
      float variance = r_x_i * ((1-r_x_i)*maxVal + relu_r_x_i) ; 
      if (variance < 0) // stupid overflows, underflows etc, causing a perfectly reasonable calculation to look negative..
         variance = 0.0 ; 
      stdevs[signalIndex+dataIndex]  = sqrtf(variance) ; 
   }
   __syncthreads() ; 
  
}

__global__ void kSoftMaxReluWithPositionBiases(
                                                 float *activations, 
                                                 float *probabilities, 
                                                 float *meanValues, 
                                                 float *featureStdevs, 
                                                 float *featureBiases,
                                                 float *positionBiases,
                                                 int softMaxWidth, 
                                                 int shift, 
                                                 int signalLength, 
                                                 float minExpForSum
                                              )
{
   extern __shared__ float sData[] ; 
   float *arrData = (float *) &sData[0] ; 
   float *arrActivation = (float *) &sData[softMaxWidth] ; 
   float *threadStores = (float *) &sData[2*softMaxWidth] ; 

   int signalIndex = blockIdx.x*signalLength ; 
   int dataIndex = blockIdx.y*softMaxWidth + threadIdx.x + shift ; 
  
   float maxVal = -FLT_MAX ;  
   for (int index = threadIdx.x ; index < softMaxWidth && dataIndex < signalLength ; 
            index += blockDim.x, dataIndex += blockDim.x)
   {
      float x_i = activations[signalIndex+dataIndex] + positionBiases[index] ; 
      arrActivation[index] = x_i ; 
      float val = x_i ; 
      x_i = x_i - .5 ; 
      while (x_i >= minExpForSum)
      {
        if (x_i > MAX_EXP)
           val = val + x_i ;
        else
           val = val +  __logf(1. + __expf(x_i)) ; 
        x_i = x_i - 1.0 ; 
      }
      
      arrData[index] = val ; 
      if (maxVal < arrData[index])
         maxVal = arrData[index] ; 
   }
   threadStores[threadIdx.x] = maxVal ; 
   __syncthreads() ; 

   // do a reduction to find the max of all maxes. 
   for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
   {
      if (threadIdx.x < s)
      {
         threadStores[threadIdx.x] = 
              threadStores[threadIdx.x+s]*(threadStores[threadIdx.x + s] >=threadStores[threadIdx.x]) + 
              threadStores[threadIdx.x] * (threadStores[threadIdx.x] > threadStores[threadIdx.x+s])  ;
      }
      __syncthreads();
   }

   // now we have max. Lets subract it from all elements, and compute intermediate logSumExp over all elements 
   // a thread is responsible for
   float bias = featureBiases[blockIdx.x] ; 
   maxVal = fmaxf(threadStores[0], bias) ; 

   __syncthreads();

   float sumExp = 0. ;

   dataIndex = blockIdx.y*softMaxWidth + threadIdx.x + shift ; 
  
   for (int index = threadIdx.x ; index < softMaxWidth && dataIndex < signalLength ; 
            index += blockDim.x, dataIndex += blockDim.x)
   {
      arrData[index] = arrData[index] - maxVal ; 
      arrActivation[index] = arrActivation[index] - bias ;
      sumExp += __expf(arrData[index]) ; 
   }

   threadStores[threadIdx.x] = sumExp ;
   __syncthreads() ; 

   // compute normalization constant over sumExp by summing together all the intermediate values.
   for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
   {
       if (threadIdx.x < s)
          threadStores[threadIdx.x] += threadStores[threadIdx.x + s];
       __syncthreads();
   }
   __syncthreads() ; 
   sumExp = threadStores[0] + __expf(bias-maxVal) ;

   // write out probabilities. 
   dataIndex = blockIdx.y*softMaxWidth + threadIdx.x + shift ; 
   for (int index = threadIdx.x ; index < softMaxWidth && dataIndex < signalLength ; 
            index += blockDim.x, dataIndex += blockDim.x)
   {
      float prob = __expf(arrData[index])/ sumExp ; 
      probabilities[signalIndex+dataIndex]  = prob ; 
      // using the relu approximation. Remember to threshold
      meanValues[signalIndex+dataIndex]  = arrActivation[index] ; 
      featureStdevs[signalIndex+dataIndex]  = sqrtf(1. / (1 + __expf(-arrActivation[index])));
   }
   __syncthreads() ; 
  
}

__global__ void kSoftMaxWithOffAndPositionBiases(
                                                 float *activations, 
                                                 float *probabilities, 
                                                 float *featureBiases,
                                                 float *positionBiases,
                                                 int softMaxWidth, 
                                                 int signalLength
                                                )
{
   extern __shared__ float sData[] ; 
   float *arrData = (float *) &sData[0] ; 
   float *threadStores = (float *) &sData[softMaxWidth] ; 

   int signalIndex = blockIdx.x*signalLength ; 
   int dataIndex = blockIdx.y*softMaxWidth + threadIdx.x ; 
  
   float maxVal = -FLT_MAX ;  
   for (int index = threadIdx.x ; index < softMaxWidth && dataIndex < signalLength ; 
            index += blockDim.x, dataIndex += blockDim.x)
   {
      arrData[index] = activations[signalIndex+dataIndex] + positionBiases[index] ; 
      if (maxVal < arrData[index])
         maxVal = arrData[index] ; 
   }
   threadStores[threadIdx.x] = maxVal ; 
   __syncthreads() ; 

   // do a reduction to find the max of all maxes. 
   for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
   {
      if (threadIdx.x < s)
      {
         threadStores[threadIdx.x] = 
              threadStores[threadIdx.x+s]*(threadStores[threadIdx.x + s] >=threadStores[threadIdx.x]) + 
              threadStores[threadIdx.x] * (threadStores[threadIdx.x] > threadStores[threadIdx.x+s])  ;
      }
      __syncthreads();
   }

   // now we have max. Lets subract it from all elements, and compute intermediate logSumExp over all elements 
   // a thread is responsible for
   float bias = featureBiases[blockIdx.x] ; 
   maxVal = fmaxf(threadStores[0], bias) ; 

   __syncthreads();

   float sumExp = 0. ;

   dataIndex = blockIdx.y*softMaxWidth + threadIdx.x ; 
  
   for (int index = threadIdx.x ; index < softMaxWidth && dataIndex < signalLength ; 
            index += blockDim.x, dataIndex += blockDim.x)
   {
      arrData[index] = arrData[index] - maxVal ; 
      sumExp += __expf(arrData[index]) ; 
   }

   threadStores[threadIdx.x] = sumExp ;
   __syncthreads() ; 

   // compute normalization constant over sumExp by summing together all the intermediate values.
   for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
   {
       if (threadIdx.x < s)
          threadStores[threadIdx.x] += threadStores[threadIdx.x + s];
       __syncthreads();
   }
   __syncthreads() ; 
   sumExp = threadStores[0] + __expf(bias-maxVal) ;

   // write out probabilities. 
   dataIndex = blockIdx.y*softMaxWidth + threadIdx.x ; 
   for (int index = threadIdx.x ; index < softMaxWidth && dataIndex < signalLength ; 
            index += blockDim.x, dataIndex += blockDim.x)
   {
      probabilities[signalIndex+dataIndex]  = __expf(arrData[index])/ sumExp ; 
   }
   __syncthreads() ; 
  
}



__global__ void kSoftMaxWithOff(float *activations, float *probabilities, float *biases,
                       int softMaxWidth, int signalLength, int numPtsPerThread)
{
   extern __shared__ float sData[] ; 
   float *arrData = (float *) &sData[0] ; 
   float *threadStores = (float *) &sData[softMaxWidth] ; 

   int signalIndex = blockIdx.x*signalLength ; 
   int blockIndex = blockIdx.y*softMaxWidth ; 
   int softMaxIndex = blockIndex + threadIdx.x*numPtsPerThread ;

   // copy to local memory
   int numToCopy = numPtsPerThread ; 
   if (softMaxIndex + numToCopy > signalLength)
      numToCopy = signalLength - softMaxIndex ; 
   if (softMaxIndex + numToCopy > blockIndex + softMaxWidth)
      numToCopy = blockIndex + softMaxWidth - softMaxIndex ; 
  
   float maxVal = -FLT_MAX ;  
   for (int index = 0 ; index < numToCopy ; index++)
   {
      arrData[softMaxIndex+index-blockIndex] = activations[signalIndex+softMaxIndex+index] ; 
      if (maxVal < arrData[softMaxIndex+index])
         maxVal = arrData[softMaxIndex+index-blockIndex] ; 
   }
   threadStores[threadIdx.x] = maxVal ; 
   __syncthreads() ; 

   // do a reduction to find the max of all maxes. 
   for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
   {
      if (threadIdx.x < s)
      {
         threadStores[threadIdx.x] = 
              threadStores[threadIdx.x+s]*(threadStores[threadIdx.x + s] >=threadStores[threadIdx.x]) + 
              threadStores[threadIdx.x] * (threadStores[threadIdx.x] > threadStores[threadIdx.x+s])  ;
      }
      __syncthreads();
   }

   // now we have max. Lets subract it from all elements, and compute intermediate logSumExp over all elements 
   // a thread is responsible for
   float bias = biases[blockIdx.x] ; 
   maxVal = threadStores[0] * (threadStores[0] > bias) + bias * (bias >= threadStores[0]) ; 
   __syncthreads();

   float sumExp = 0. ;  
   for (int index = 0 ; index < numToCopy ; index++)
   {
      arrData[softMaxIndex+index-blockIndex] = arrData[softMaxIndex+index-blockIndex] - maxVal ; 
      sumExp += __expf(arrData[softMaxIndex+index-blockIndex]) ;
   }

   threadStores[threadIdx.x] = sumExp ;
   __syncthreads() ; 

   // compute normalization constant over sumExp by summing together all the intermediate values.
   for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
   {
       if (threadIdx.x < s)
          threadStores[threadIdx.x] += threadStores[threadIdx.x + s];
       __syncthreads();
   }
   __syncthreads() ; 
   sumExp = threadStores[0] + __expf(bias-maxVal) ;

   for (int index = 0 ; index < numToCopy ; index++)
   {
      probabilities[signalIndex+softMaxIndex+index]  = __expf(arrData[softMaxIndex+index-blockIndex])/ sumExp ; 
   }
   __syncthreads() ; 
  
}


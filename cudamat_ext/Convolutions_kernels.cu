#include <cuda.h>  
// Kernel that executes convolution. Nothing fancy is done. We don't even try to avoid
// block effects here.
__global__ void KernConvolve(float *data, 
                             float *kernels, 
                             float *dataOut, 
                             int signalLength, 
                             int kernelWidth, 
                             int numPtsPerBlock)
{
   // first fetch the data for this block. 

   extern __shared__ float sData[] ; 
   float *arrData = (float *) &sData[0] ; 
   float *arrKernel = (float *) &sData[numPtsPerBlock+kernelWidth-1] ; 

   // copy first the data vector.
   int dataIndexInSignal = blockIdx.y * numPtsPerBlock + threadIdx.x - kernelWidth/2; 
   int numPtsToCopy = numPtsPerBlock + kernelWidth-1 ; 
  
   for (int index = threadIdx.x ; index < numPtsToCopy ;
        index+= blockDim.x, dataIndexInSignal += blockDim.x)
   {
      if (dataIndexInSignal < 0 || dataIndexInSignal >= signalLength)
         arrData[index] = 0 ; 
      else
         arrData[index] = data[dataIndexInSignal] ; 
   }

   __syncthreads() ; 

   // copy the kernel next. 
   int dataIndexInKernel = blockIdx.x * kernelWidth + threadIdx.x ;
  
   for (int index = threadIdx.x ; index < kernelWidth ; 
        index+= blockDim.x, dataIndexInKernel += blockDim.x)
   {
      arrKernel[index] = kernels[dataIndexInKernel] ; 
   }

   __syncthreads() ; 


   // perform the convolution and write out the result.

   //output position.
   dataIndexInSignal = blockIdx.y * numPtsPerBlock + threadIdx.x ; 

   for (int index = threadIdx.x ; index < numPtsPerBlock && dataIndexInSignal < signalLength ; 
        index+= blockDim.x, dataIndexInSignal += blockDim.x)
   {
      float val = 0.0 ; 
      for (int wtIndex = 0 ; wtIndex < kernelWidth ; wtIndex++)
      {
         val += arrKernel[wtIndex] * arrData[index+wtIndex] ;
      }
      // index of output data point in signal
      int outIndex = blockIdx.x * signalLength + dataIndexInSignal ; 
      dataOut[outIndex] = val ; 
   }
}


// Kernel that executes convolution. Nothing fancy is done. We don't even try to avoid
// block effects here.
__global__ void KernPartialConvolve(float *dPtrSignal1, float *dPtrSignal2, float *dPtrBlockProducts, 
                                int signalLength, int kernelWidth, int numPtsPerBlock)
{
   extern __shared__ float sData[] ; 
   float *arrData1 = (float *) &sData[0] ; 
   float *arrData2 = (float *) &sData[numPtsPerBlock] ; 

   // copy first data vector.
   int dataIndexInSignal = blockIdx.y * numPtsPerBlock + threadIdx.x ; 
  
   for (int index = threadIdx.x ; index < numPtsPerBlock && dataIndexInSignal < signalLength ; 
        index+= blockDim.x, dataIndexInSignal += blockDim.x)
   {
      arrData1[index] = dPtrSignal1[dataIndexInSignal] ; 
   }

   __syncthreads() ; 

   // copy second data vector.
   int numPtsPerBlock2 = numPtsPerBlock + kernelWidth - 1 ;
   int signalIndex = blockIdx.x * signalLength ; 
   dataIndexInSignal = blockIdx.y * numPtsPerBlock + threadIdx.x - (kernelWidth-1)/2; 

   for (int index = threadIdx.x ; index < numPtsPerBlock2 && dataIndexInSignal < signalLength ; 
        index+= blockDim.x, dataIndexInSignal += blockDim.x)
   {
      if (dataIndexInSignal < 0)
         arrData2[index] = 0  ; 
      else
         arrData2[index] = dPtrSignal2[signalIndex+dataIndexInSignal] ;
   }

   __syncthreads() ; 

   dataIndexInSignal = blockIdx.y * numPtsPerBlock ;
   int maxIndex = numPtsPerBlock+kernelWidth-1 ; 

   if (signalLength + (kernelWidth-1)/2-dataIndexInSignal < maxIndex)
      maxIndex = signalLength + (kernelWidth-1)/2 - dataIndexInSignal ; 

   int dataIndexInBlock = blockIdx.x*gridDim.y*kernelWidth + blockIdx.y ; 
   for (int shift = threadIdx.x ; shift < kernelWidth ; shift+=blockDim.x)
   {
      float val = 0.0 ; 
      for (int index = 0 ; index < numPtsPerBlock ; index++)
      {
         if (index+shift >= maxIndex)
            break ; 
         val += arrData1[index]*arrData2[index+shift] ; 
      }
      dPtrBlockProducts[dataIndexInBlock + (kernelWidth-1-shift)*gridDim.y] = val ; 
   }

}

// Kernel that sums the results from PartialConvolve. 
// Each kernel will have dimension of kernelWidth, so kernelWidth sums have to be computed per block.
// Each block will handle one dimension of a kernel sum. So the number of blocks is (kernel dimension) x (# of kernels)
__global__ void KernPartialConvolveSum(float *dPtrBlockProducts, float *dPtrResults,
                                int kernelWidth, int numPiecesPerKernel, 
                                int numKernels)
{
   // Results from partial convolve resulted in numPiecesPerKernel partial sums for 
   // every dimension of a kernel. Here a block has to sum these together.
   int  numPiecesPerThread = numPiecesPerKernel/blockDim.x ; 
   if (blockDim.x*numPiecesPerThread < numPiecesPerKernel)
      numPiecesPerThread++ ; 

   int startKernelIndex = blockIdx.x *  numPiecesPerKernel * kernelWidth ; 
   int startDataIndexForBlock = startKernelIndex + blockIdx.y * numPiecesPerKernel ; 
   int startDataIndexForThread = numPiecesPerThread*threadIdx.x ; 

   extern __shared__ float sData[] ; 

   int numToCopy1 = numPiecesPerThread ; 
   if (startDataIndexForThread + numToCopy1 > numPiecesPerKernel)
      numToCopy1 = numPiecesPerKernel - startDataIndexForThread ;

   float val = 0 ; 
   for (int index = 0 ; index < numToCopy1 ; index++)
   {
      val += dPtrBlockProducts[startDataIndexForBlock+startDataIndexForThread+index] ;
   }
   sData[threadIdx.x] = val ;

   __syncthreads() ; 


   for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
   {
      if (threadIdx.x < s)
          sData[threadIdx.x] += sData[threadIdx.x + s];
      __syncthreads();
   }
   if (threadIdx.x == 0)
   {
      dPtrResults[blockIdx.y+kernelWidth*blockIdx.x] = sData[0] ; 
   }
  
}
__global__ void KernAddSignals(float *signals, float *sumSignals, int signalLength, 
                           int numSignals, int numPtsPerBlock, int numPtsPerThread)
{
   int startIndex = blockIdx.x * numPtsPerBlock + threadIdx.x * numPtsPerThread ;
   for (int ptNum = 0 ; ptNum < numPtsPerThread; ptNum++)
   {
      // index of data point in signal 
      int index = startIndex + ptNum ; 
      if (index >= signalLength)
         break ; 
     
     float val = 0 ;
     for (int signalNum = 0 ; signalNum < numSignals ; signalNum++)
     {
         val += signals[index+signalNum*signalLength] ;
     }
     sumSignals[index] = val ; 
   }
}

// Kernel that executes convolution. Nothing fancy is done. We don't even try to avoid
// block effects here.
__global__ void KernReverseConvolve(float *signals, float *kernels, float *dataOut, int signalLength, int kernelWidth, 
                         int numKernels, int numPtsPerBlock, int numPtsPerThread)
{
   // can probably speed this up well by fetching kernels to local memory or put it in constant memory. 
   int kernelIndex = blockIdx.x ; 
   int signalStartIndex = kernelIndex * signalLength ; 
   int startIndex = blockIdx.y * numPtsPerBlock + threadIdx.x * numPtsPerThread ;
   for (int ptNum = 0 ; ptNum < numPtsPerThread; ptNum++)
   {
      // index of data point in signal 
      int index = startIndex + ptNum ; 
      if (index >= signalLength)
         break ; 
     
     float val = 0 ;
     int startIndexInKernel = kernelIndex*kernelWidth ;
     for (int wtIndex = 0 ; wtIndex < kernelWidth ; wtIndex++)
     {
        if (wtIndex + index >= signalLength + (kernelWidth-1)/2)
            break ;
         if (index+wtIndex < (kernelWidth-1)/2)
            continue ; 
         val += kernels[kernelWidth-1-wtIndex+startIndexInKernel] * 
                       signals[signalStartIndex + index + wtIndex - (kernelWidth-1)/2] ;
      }
      dataOut[signalStartIndex+index] = val ; 
   }
}



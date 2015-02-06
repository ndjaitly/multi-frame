#pragma once 

__global__ void KernConvolve(float *data, 
                             float *kernels, 
                             float *dataOut, 
                             int signalLength, 
                             int kernelWidth, 
                             int numPtsPerBlock) ; 

__global__ void KernPartialConvolve(float *dPtrSignal1, 
                                    float *dPtrSignal2, 
                                    float *dPtrBlockProducts, 
                                    int signalLength, 
                                    int kernelWidth, 
                                    int numPtsPerBlock) ; 

__global__ void KernPartialConvolveSum(float *dPtrBlockProducts, float *dPtrResults,
                         int kernelWidth, int numPiecesPerKernel, 
                         int numKernels) ; 

__global__ void KernAddSignals(float *signals, float *sumSignals, int signalLength, 
                         int numSignals, int numPtsPerBlock, int numPtsPerThread) ; 

__global__ void KernReverseConvolve(float *signals, float *kernels, float *dataOut, 
                         int signalLength, int kernelWidth, int numKernels, 
                         int numPtsPerBlock, int numPtsPerThread) ; 

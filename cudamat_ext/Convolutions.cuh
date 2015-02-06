#pragma once
extern int ShiftedConvolution(cudamat *signal1, cudamat *signal2, int numKernels, cudamat *target, 
                              int kernelWidth, cudamat *scratchPad) ; 
extern int Convolve(cudamat *signal, cudamat *kernels, int kernelWidth, int numKernels, cudamat *target) ; 

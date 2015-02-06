#pragma once
extern int MultinomialSamples(
                              cudamat *unifRandNums, 
                              cudamat *probabilities, 
                              cudamat *samples, 
                              int softMaxWidth
                             ) ; 

extern int ShiftedConvolution(
                              cudamat *signal1, 
                              cudamat *signal2, 
                              int numKernels, 
                              cudamat *target, 
                              int kernelWidth, 
                              cudamat *scratchPad
                             ) ; 

extern int SoftMaxWithOff(
                          cudamat *activations, 
                          cudamat *probabilities, 
                          cudamat *biases, 
                          int softMaxWidth
                         ) ; 

extern int SoftMaxWithOffAndPositionBiases(
                                           cudamat *activations, 
                                           cudamat *probabilities, 
                                           cudamat *featureBiases, 
                                           cudamat *positionBiases, 
                                           int softMaxWidth
                                          ) ; 

extern int SoftMaxStackApproxWithPositionBiases(
                                         cudamat *activations, 
                                         cudamat *probabilities, 
                                         cudamat *stdevs, 
                                         cudamat *featureBiases, 
                                         cudamat *positionBiases, 
                                         int softMaxWidth
                                        ) ; 

extern int  SoftMaxReluWithPositionBiases(
                       cudamat *activations, 
                       cudamat *probabilities, 
                       cudamat *meanValues, 
                       cudamat *featureStdevs, 
                       cudamat *featureBiases, 
                       cudamat *positionBiases,
                       int softMaxWidth,
                       int shift
                       ) ; 


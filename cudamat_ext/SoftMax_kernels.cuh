#pragma once

__global__ void kMultinomialSample(
                                   float *unifRandNums, 
                                   float *probabilities, 
                                   float *samples, 
                                   int softMaxWidth, 
                                   int startShift,
                                   int signalLength
                                  ) ; 

__global__ void kCumulativeSum(
                               float *probabilities, 
                               float *sumProbabilities, 
                               int softMaxWidth, 
                               int signalLength
                              ) ; 

__global__ void kSoftMaxWithOff(
                                float *activations, 
                                float *probabilities, 
                                float *biases,
                                int softMaxWidth,
                                int signalLength, 
                                int numPtsPerThread
                               ) ; 
__global__ void kSoftMaxWithOffAndPositionBiases(
                                                 float *activations, 
                                                 float *probabilities, 
                                                 float *featureBiases,
                                                 float *positionBiases,
                                                 int softMaxWidth, 
                                                 int signalLength
                                                ) ; 
__global__ void kSoftMaxStackApproxWithPositionBiases(
                                               float *activations, 
                                               float *probabilities, 
                                               float *stdevs, 
                                               float *featureBiases,
                                               float *positionBiases,
                                               int softMaxWidth, 
                                               int signalLength
                                               ) ; 

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
                                              ) ; 


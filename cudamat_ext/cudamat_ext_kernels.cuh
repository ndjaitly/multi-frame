__global__ void kMinimum(float* mat1, float* mat2, float* target,
                         unsigned int len);
__global__ void kMaximum(float* mat1, float* mat2, float* target,
                         unsigned int len);
__global__ void kThresholdBelow(float* mat, float threshold, float* target,
                         unsigned int len);
__global__ void kThresholdAbove(float* mat, float threshold, float* target,
                         unsigned int len);
__global__ void kRound(float* mat, float threshold, float* target, 
                         unsigned int len);
__global__ void kReplicateColVector(float* mat, float* vec, unsigned int width,
                          unsigned int height);
__global__ void kGetColumnVectors(float *mat, float *order,
                          float *target, unsigned int start, unsigned int end,
                          unsigned int width, unsigned int height,
                          unsigned int widthSrc) ; 
__global__ void kSetColumnVectors(float *mat, float *order,
                          float *target, unsigned int start, unsigned int end,
                          unsigned int width, unsigned int height,
                          unsigned int widthSrc) ; 
__global__ void kCopyReorderedColumnVectorsToColumnVector(float *mat,
                          float *srcOrder, float *target, float *targetOrder,
                          unsigned int start, unsigned int end, 
                          unsigned int width, unsigned int height,
                        unsigned int widthSrc);
__global__ void kCopySubsequences(float *mat, float *order, float *target,
                        unsigned int start, unsigned int end, 
                        unsigned int length, unsigned int totalLength);
__global__ void kAddMatrixMult(float *orig, float *mat1, float *mat2,
                        float *target, float alpha, unsigned int numEls); 
__global__ void kResampleColumns(float *orig, float *target, int height,
                         float newRate) ; 
__global__ void kResampleColumnsVect(float *orig, float *target, int height,
                         float *newRates) ; 
__global__ void kResampleColumnsVectGrad(float *orig, float *orig_diff, 
                                         float *orig_grad, int height, 
                                         float *rateVect);
__global__ void kResampleColumnsVectGradRate(float *orig, float *orig_diff, 
                                             int height, float *rateVect,
                                             float *rateVectGrad);
__global__ void kSampleSoftMax(float* mat, float* rand,  float* target,
                               unsigned int height);
__global__ void kSoftMax(float* mat, float* target, unsigned int width,
                         unsigned int height);
__global__ void kArgMaxColumnwise(float* mat, float* target, 
                                  unsigned int width,
                                  unsigned int height);
__global__ void kArgMaxIndicatorColumnwise(float* mat, float* target, 
                                  unsigned int width,
                                  unsigned int height);
__global__ void kShiftColumns(float *orig, float *target, int height,
                              float *shiftVect);
__global__ void kColumnwiseDot(float* mat1, float* mat2, 
                              float* target, unsigned int num_dims);
__global__ void kLogisticLogProb(float* outputs, float* targets, 
                              float* log_probs, unsigned int num_dims);
__global__ void kLogisticGrad(float* prob, float* out_grad, float* in_grad,
                             unsigned int len);
__global__ void kSoftmaxGrad(float* prob, float* out_grad, float* in_grad,
                             unsigned int height);
__global__ void kSoftmaxAccuraccy(float* probs, float* targets, float* log_prob,
                             float* correct, unsigned int num_dims);
__global__ void kSoftmaxAccuraccyVect(float* probs, float* targets, float* log_prob,
                             float* correct, unsigned int num_dims);

__global__ void kDropout(unsigned int* rndMults, unsigned long long* rndWords,
                         float* gData, unsigned int numElements,
                         float drop_percent);
__global__ void kNormalizeColumns(float *mat, float *tgt, unsigned int height);
__global__ void kThesholdColumnNorms(float *mat, float *biases, float *tgt, 
                                     unsigned int height, float threshold);
__global__ void kSetDiagonal(float *mat, float *diag, unsigned int side);
__global__ void kReverseColumnEntries(float *mat, float *target, int h);
__global__ void kSumColumns(float *mat, float *target, int h, int num_columns);
__global__ void kApplyLog1PlusAbs(float* mat, float* target, unsigned int len);
__global__ void kApplyLog1PlusAbsGrad(float* act, float *out_grad, 
                                      float* target, unsigned int len);

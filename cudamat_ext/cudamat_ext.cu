#include "common.cuh"
#include "cudamat_ext_kernels.cuh"
#include "cudamat_kernels.cuh"
#include "cudamat.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <curand.h>

extern "C" {
extern curandGenerator_t * create_rand_generator() {
  curandGenerator_t *gen_ptr = new curandGenerator_t();
  curandCreateGenerator(gen_ptr, CURAND_RNG_PSEUDO_DEFAULT);
  return gen_ptr;
}

extern void free_rand_generator(curandGenerator_t *gen_ptr) {
  delete gen_ptr;
}

extern void set_rand_generator_seed(curandGenerator_t *gen_ptr, 
                                    unsigned long long seed) {
  int err_code = curandSetPseudoRandomGeneratorSeed(*gen_ptr, seed);
}

extern int apply_log_1_plus_abs_grad(cudamat* mat, cudamat *grad, 
                                      cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device || !grad->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1] 
        || mat->size[0] != grad->size[0] || mat->size[1] != grad->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplyLog1PlusAbsGrad<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, grad->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}


extern int apply_log_1_plus_abs(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplyLog1PlusAbs<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int fill_with_randn(cudamat* matrix, curandGenerator_t *gen_ptr) {
  unsigned int x = matrix->size[0], y = matrix->size[1]; 
  if (!matrix->on_device)
    return ERROR_NOT_ON_DEVICE;
  int status = curandGenerateNormal(*gen_ptr, matrix->data_device, x*y, 0, 1.0);
  cudaThreadSynchronize();
  return status;
}                        

extern int fill_with_rand(cudamat* matrix, curandGenerator_t *gen_ptr) {
  unsigned int x = matrix->size[0], y = matrix->size[1]; 
  if (!matrix->on_device)
    return ERROR_NOT_ON_DEVICE;
  int status = curandGenerateUniform(*gen_ptr, matrix->data_device, x*y);
  cudaThreadSynchronize();
  return status;
}

extern int replicate_col_vec(cudamat* mat, cudamat* vec) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kReplicateColVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}

extern int column_dot(cudamat* mat1, cudamat* mat2,
                         cudamat* target)
{
  unsigned int h = mat1->size[0],
               w = mat1->size[1];

  if (!mat1->on_device || !mat2->on_device
       || !target->on_device)
    return ERROR_NOT_ON_DEVICE;

  if (target->is_trans)
    return ERROR_TRANSPOSED;

  if (h != mat2->size[0] || w != mat2->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  if (w != target->size[0] * target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  int num_threads = 32;
  int shared_mem =  num_threads * sizeof(float) ; 
  kColumnwiseDot<<<w, num_threads, shared_mem>>>(
                   mat1->data_device,
                   mat2->data_device,
                   target->data_device, 
                   h);

  cudaThreadSynchronize();

  if (checkCUDAError()) {
     return CUDA_ERROR;
  }
  return 0;
}

extern int logistic_log_prob(cudamat* outputs, cudamat* targets,
                         cudamat* log_probs)
{
  unsigned int h = outputs->size[0],
               w = outputs->size[1];

  if (!outputs->on_device || !targets->on_device
       || !log_probs->on_device)
    return ERROR_NOT_ON_DEVICE;

  if (outputs->is_trans)
    return ERROR_TRANSPOSED;

  if (h != targets->size[0] || w != targets->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  if (1 != log_probs->size[0] || w != log_probs->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  int num_threads = 32;
  int shared_mem =  num_threads * sizeof(float) ; 
  kLogisticLogProb<<<w, num_threads, shared_mem>>>(
                   outputs->data_device,
                   targets->data_device,
                   log_probs->data_device, 
                   h);

  cudaThreadSynchronize();

  if (checkCUDAError()) {
     return CUDA_ERROR;
  }
  return 0;
}


extern int logistic_grad(cudamat* probs, cudamat* out_grad,
                         cudamat* in_grad)
{
  unsigned int h = probs->size[0],
               w = probs->size[1];

  if (!probs->on_device || !out_grad->on_device
       || !in_grad->on_device)
    return ERROR_NOT_ON_DEVICE;

  if (probs->is_trans)
    return ERROR_TRANSPOSED;

  if (h != out_grad->size[0] || w != out_grad->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  if (h != in_grad->size[0] || w != in_grad->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  kLogisticGrad<<<NUM_VECTOR_OP_BLOCKS,
             NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(
                   probs->data_device,
                   out_grad->data_device,
                   in_grad->data_device, 
                   w*h);

    cudaThreadSynchronize();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}

extern int softmax_accuraccy(cudamat* prob, cudamat* target,
                 cudamat* log_prob, cudamat* correct)
{
   unsigned int num_dim = prob->size[0],
               num_pts = prob->size[1];

   if (!prob->on_device || !target->on_device
       || !log_prob->on_device || !correct->on_device)
      return ERROR_NOT_ON_DEVICE;

   if (prob->is_trans || target->is_trans)
      return ERROR_TRANSPOSED;

   if (log_prob->size[0] != 1 || correct->size[0] != 1)
      return ERROR_INCOMPATIBLE_DIMENSIONS;

   if (log_prob->size[1] != num_pts || 
       correct->size[1] != num_pts)
      return ERROR_INCOMPATIBLE_DIMENSIONS;

   if (target->size[1] != num_pts)
      return ERROR_INCOMPATIBLE_DIMENSIONS;

   int num_threads_per_block = 32 ;
   
   if (target->size[0] == num_dim) {
      unsigned int shared_mem_size = sizeof(float) + sizeof(int) +
                       num_threads_per_block * sizeof(float);
      kSoftmaxAccuraccy<<<num_pts, num_threads_per_block,
                 shared_mem_size>>>(prob->data_device,
                                    target->data_device,
                                    log_prob->data_device, 
                                    correct->data_device, 
                                    num_dim);
   } else {
      if (target->size[0] != 1)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
      unsigned int shared_mem_size = 
                       num_threads_per_block * sizeof(float);
      kSoftmaxAccuraccyVect<<<num_pts, num_threads_per_block,
                 shared_mem_size>>>(prob->data_device,
                                    target->data_device,
                                    log_prob->data_device, 
                                    correct->data_device, 
                                    num_dim);
   }

  cudaThreadSynchronize();

  if (checkCUDAError()) {
        return CUDA_ERROR;
  }

  return 0;
}


extern int softmax_grad(cudamat* probs, cudamat* out_grad,
                         cudamat* in_grad)
{
  unsigned int h = probs->size[0],
               w = probs->size[1];

  if (!probs->on_device || !out_grad->on_device
       || !in_grad->on_device)
    return ERROR_NOT_ON_DEVICE;

  if (probs->is_trans)
    return ERROR_TRANSPOSED;

  if (h != out_grad->size[0] || w != out_grad->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  if (h != in_grad->size[0] || w != in_grad->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  int num_threads_per_block = 32 ;
  //if (h < num_threads_per_block)
    //num_threads_per_block = h;

  unsigned int shared_mem_size = num_threads_per_block * sizeof(float);
  kSoftmaxGrad<<<w, num_threads_per_block,
                 shared_mem_size>>>(
                   probs->data_device,
                   out_grad->data_device,
                   in_grad->data_device, 
                   h);

    cudaThreadSynchronize();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}
extern int maximum(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMaximum<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}


extern int minimum(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMinimum<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}


extern int get_column_vectors(cudamat* source, cudamat* indices, cudamat* target, 
	unsigned int start, unsigned int end) // DJ March 1 2010.
{

    if (!source->on_device || !indices->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (source->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    int height = source->size[0];
    int width = (end-start);
    int widthSrc = source->size[1] ; 

    // note that widthSrc can be smaller than width because 
    // multiple copying of columns is allowed. Only thing that is required is
    // that order[start..end] be all smaller than widthSrc. Can only check
    // that in a kernel efficiently. 

    if (height != target->size[0] || indices->size[0] * indices->size[1] < end || start >= end
    	|| (indices->size[1] != 1 && indices->size[0] != 1) || target->size[1] < width)
        return ERROR_INCOMPATIBLE_DIMENSIONS;


    kGetColumnVectors<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(source->data_device, 
    	indices->data_device, target->data_device, start, end, width, height, widthSrc);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int set_column_vectors(cudamat* tgt, cudamat* indices, cudamat* src, 
	unsigned int start, unsigned int end) // DJ March 1 2010.
{

    if (!tgt->on_device || !indices->on_device || !src->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (tgt->is_trans != src->is_trans)
        return ERROR_TRANSPOSEDNESS;

    int height = tgt->size[0];
    int width = (end-start);
    int widthSrc = tgt->size[1] ; 

    // note that widthSrc can be smaller than width because 
    // multiple copying of columns is allowed. Only thing that is required is
    // that order[start..end] be all smaller than widthSrc. Can only check
    // that in a kernel efficiently. 

    if (height != src->size[0] || indices->size[0] * indices->size[1] < end || start >= end
    	|| (indices->size[1] != 1 && indices->size[0] != 1) || src->size[1] < width)
        return ERROR_INCOMPATIBLE_DIMENSIONS;


    kSetColumnVectors<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(tgt->data_device, 
    	indices->data_device, src->data_device, start, end, width, height, widthSrc);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

// DJ. Tue April 19th. 2010.
extern int copy_reordered_column_vectors_to_reordered_columns(
                                      cudamat* source, 
                                      cudamat* srcIndices, 
                                      cudamat* target, 
                                      cudamat* targetIndices, 
	                                   unsigned int start, 
                                      unsigned int end) // DJ April 19 2011.
{

    if (!source->on_device || !srcIndices->on_device 
       || !target->on_device || !targetIndices->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (source->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    int height = source->size[0];
    int width = (end-start);
    int widthSrc = source->size[1] ; 

    // note that widthSrc can be smaller than width because 
    // multiple copying of columns is allowed. Only thing that is required is
    // that order[start..end] be all smaller than widthSrc. Can only check
    // that in a kernel efficiently. 

    if (height != target->size[0] 
       || srcIndices->size[0] * srcIndices->size[1] < end 
       || targetIndices->size[0] * targetIndices->size[1] < end 
       || start >= end 
       || (srcIndices->size[1] != 1 && srcIndices->size[0] != 1) 
       || (targetIndices->size[1] != 1 && targetIndices->size[0] != 1) 
       || target->size[1] < width)
        return ERROR_INCOMPATIBLE_DIMENSIONS;


    kCopyReorderedColumnVectorsToColumnVector
        <<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>
               (source->data_device, 
    	          srcIndices->data_device, 
                target->data_device, 
    	          targetIndices->data_device, 
                start, end, width, height, widthSrc);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}
// Deep Jaitly.
extern int copy_subsequences(cudamat* source, cudamat* indices, cudamat* target, 
   unsigned int start, unsigned int end, unsigned int length) // DJ Oct 11 2010.
{

    if (!source->on_device || !indices->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (source->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    int height = source->size[0];
    int width = (end-start);
    int widthSrc = source->size[1] ; 
    int totalLength = height*widthSrc ; 

    if (height != 1 && widthSrc != 1)
        return ERROR_INCOMPATIBLE_DIMENSIONS;


    if (indices->size[1] * indices->size[0] < end || start >= end
      || target->size[1] * target->size[0] < width*length)
        return ERROR_INCOMPATIBLE_DIMENSIONS;


    int numBlocks = width ; 
    kCopySubsequences<<<numBlocks,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(source->data_device, 
      indices->data_device, target->data_device, start, end, length, totalLength);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else 
        return 0;
}

extern int threshold_below(cudamat* mat, float threshold, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kThresholdBelow<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, threshold, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int threshold_above(cudamat* mat, float threshold, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kThresholdAbove<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, threshold, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int apply_round(cudamat* mat, float threshold, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kRound<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, threshold, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

/* Add alpha times elementwise multiplication of 2 matrices to a matrix. Deep Jaitly - Jan 31, 2011*/
extern int add_matrix_mult(cudamat *orig, cudamat* mat1, cudamat* mat2, cudamat* target, float alpha) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device 
         || !orig->on_device)
        return ERROR_NOT_ON_DEVICE;

    if ((mat1->is_trans != mat2->is_trans) || (orig->is_trans != mat2->is_trans)
        || (mat1->is_trans != target->is_trans))
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1] || 
        orig->size[0] != target->size[0] || orig->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddMatrixMult<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(
           orig->data_device, 
           mat1->data_device, 
           mat2->data_device, 
           target->data_device, 
           alpha, 
           len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

/* Resample columns. Deep Jaitly 30th April 2011 */
extern int ResampleColumns(cudamat *orig, cudamat* target, float newRate) {

    if (!orig->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (orig->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (orig->size[0] != target->size[0] || orig->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    int numColumns = orig->size[1] ; 
    kResampleColumns<<<numColumns,32>>>(
           orig->data_device, 
           target->data_device, 
           orig->size[0], 
           newRate) ; 

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
/* Resample columns. Deep Jaitly 30th April 2011 */
extern int ResampleColumnsVect(cudamat *orig, cudamat* target, cudamat* rateVect) {

    if (!orig->on_device || !target->on_device || !rateVect->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (orig->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (orig->size[0] != target->size[0] || orig->size[1] != target->size[1] 
                  || orig->size[1] != rateVect->size[0]*rateVect->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    int numColumns = orig->size[1] ; 
    kResampleColumnsVect<<<numColumns,32>>>(
           orig->data_device, 
           target->data_device, 
           orig->size[0], 
           rateVect->data_device) ; 

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

/* Calculate gradients for Resample columns. Deep Jaitly Nov 6 2011 */
extern int ResampleColumnsVectGrad(cudamat *orig, cudamat* grad_mult, 
                                   cudamat *grads, cudamat* rateVect, 
                                   cudamat *rate_grads) {

    if (!orig->on_device || !rateVect->on_device
        || !grad_mult->on_device || ! grads->on_device || !rate_grads->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (orig->is_trans != grads->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (orig->size[1] != rateVect->size[0]*rateVect->size[1]
        || rate_grads->size[0]*rate_grads->size[1] != orig->size[1] 
        || grads->size[0] != orig->size[0]
        || grads->size[1] != orig->size[1] 
        || grad_mult->size[0] != orig->size[0]
        || grad_mult->size[1] != orig->size[1])
    {
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    }

    int numColumns = orig->size[1] ; 
    unsigned int shared_mem_size = 3*32*4;
    kResampleColumnsVectGrad<<<numColumns,32, shared_mem_size>>>(
           orig->data_device,
           grad_mult->data_device,
           grads->data_device,
           orig->size[0], 
           rateVect->data_device);

    shared_mem_size = 32*4;
    kResampleColumnsVectGradRate<<<numColumns,32, shared_mem_size>>>(
           orig->data_device,
           grad_mult->data_device,
           orig->size[0], 
           rateVect->data_device,
           rate_grads->data_device) ;

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int sample_softmax(cudamat* mat, cudamat* rand, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device || !rand->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    kSampleSoftMax<<<w,32>>>(mat->data_device, rand->data_device, 
                             target->data_device, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int softmax(cudamat* mat, cudamat* target) { 
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (target->size[0] != h || target->size[1] != w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    
    int shared_mem_size = 32 * sizeof(float) ; 
    kSoftMax<<<w,32, shared_mem_size>>>(mat->data_device, 
                                target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}
extern int argmax_by_axis(cudamat* mat, cudamat* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        kArgMaxColumnwise<<<w,32>>>(mat->data_device, target->data_device, w, h);

        cudaThreadSynchronize();
    } else
        return ERROR_UNSUPPORTED;

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int dropout(rnd_struct* rnd_state, cudamat* mat,
                   float drop_percent) {
    int len = mat->size[0] * mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kDropout<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(
                    rnd_state->dev_mults, rnd_state->dev_words,
                    mat->data_device, len, drop_percent);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int argmax_indicator(cudamat* mat, cudamat* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != h || target->size[1] != w)
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        kArgMaxIndicatorColumnwise<<<w,32>>>(mat->data_device, target->data_device, w, h);

        cudaThreadSynchronize();
    } else
        return ERROR_UNSUPPORTED;

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int normalize_columns(cudamat* mat, cudamat *tgt) {
   if (!mat->on_device || !tgt->on_device)
     return ERROR_NOT_ON_DEVICE;

   if (tgt->size[0] != mat->size[0]  ||
         tgt->size[1] != mat->size[1])
      return ERROR_INCOMPATIBLE_DIMENSIONS;
   int num_threads = 32;
   int shared_mem =  num_threads * sizeof(float) ; 
   kNormalizeColumns<<<mat->size[1], num_threads, shared_mem>>>(
                    mat->data_device, tgt->data_device, mat->size[0]);

   if (SYNC_THREADS)
      cudaThreadSynchronize();

   if (checkCUDAError())
      return CUDA_ERROR;
   else
      return 0;
}

extern int threshold_column_norms(cudamat* mat, cudamat *biases,
                                    cudamat *tgt, float threshold) {
   if (!mat->on_device || !tgt->on_device)
     return ERROR_NOT_ON_DEVICE;

   if (tgt->size[0] != mat->size[0]  ||
         tgt->size[1] != mat->size[1] || 
          biases->size[0]*biases->size[1] != mat->size[1] )
      return ERROR_INCOMPATIBLE_DIMENSIONS;
   int num_threads = 32;
   int shared_mem =  num_threads * sizeof(float) ; 
   kThesholdColumnNorms<<<mat->size[1], num_threads, shared_mem>>>(
                    mat->data_device, biases->data_device, 
                    tgt->data_device, mat->size[0],
                    threshold);

   if (SYNC_THREADS)
      cudaThreadSynchronize();

   if (checkCUDAError())
      return CUDA_ERROR;
   else
      return 0;
}

extern int set_diagonal(cudamat* mat, cudamat* diag_vec) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !diag_vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (h != w || mat->size[0] != diag_vec->size[0] * diag_vec->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kSetDiagonal<<<1,32>>>(mat->data_device, diag_vec->data_device, h);

    cudaThreadSynchronize();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}

extern int reverse_column_entries(cudamat* mat, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans || target->is_trans)
        return ERROR_TRANSPOSED;

    if (target->size[0] != h || target->size[1] != w)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kReverseColumnEntries<<<w,32>>>(mat->data_device, target->data_device, h);

    cudaThreadSynchronize();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}

extern int sum_columns(cudamat* mat, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans || target->is_trans)
        return ERROR_TRANSPOSED;

    if (target->size[0] != h || target->size[1] != 1)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    int num_threads = 32;
    int num_blocks = h/num_threads + h % num_threads;

    kSumColumns<<<num_blocks,num_threads>>>(mat->data_device, 
                                            target->data_device, h,
                                            w);

    cudaThreadSynchronize();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}

}

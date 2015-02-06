#include "cudamat_ext_kernels.cuh"
#include "cudamat_kernels.cuh"
#include "float.h"
const int NUM_THREADS = 32;
 __device__ void reduceToSumLocal(float* sdata, unsigned int tid)
{
 
         //Synchronize threads to share shared memory data
         __syncthreads();
 
         float mySum = sdata[tid];
 
         // do reduction in shared mem
         if (NUM_THREADS >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
         if (NUM_THREADS >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
         if (NUM_THREADS >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
 
         if (NUM_THREADS == 32){
                 if (tid < 16)
                 {
                         // now that we are using warp-synchronous programming (below)
                         // we need to declare our shared memory volatile so that the compiler
                         // doesn't reorder stores to it and induce incorrect behavior.
                         volatile float* smem = sdata;
                         if (NUM_THREADS >=  32) { smem[tid] = mySum = mySum + smem[tid + 16];}
                         if (NUM_THREADS >=  16) { smem[tid] = mySum = mySum + smem[tid +  8];}
                         if (NUM_THREADS >=   8) { smem[tid] = mySum = mySum + smem[tid +  4];}
                         if (NUM_THREADS >=   4) { smem[tid] = mySum = mySum + smem[tid +  2];}
                         if (NUM_THREADS >=   2) { smem[tid] = mySum = mySum + smem[tid +  1];}
                 }
         }
         else
         {
                 if (tid < 32)
                 {
                         // now that we are using warp-synchronous programming (below)
                         // we need to declare our shared memory volatile so that the compiler
                         // doesn't reorder stores to it and induce incorrect behavior.
                         volatile float* smem = sdata;
                         if (NUM_THREADS >=  64) { smem[tid] = mySum = mySum + smem[tid + 32];}
                         if (NUM_THREADS >=  32) { smem[tid] = mySum = mySum + smem[tid + 16];}
                         if (NUM_THREADS >=  16) { smem[tid] = mySum = mySum + smem[tid +  8];}
                         if (NUM_THREADS >=   8) { smem[tid] = mySum = mySum + smem[tid +  4];}
                         if (NUM_THREADS >=   4) { smem[tid] = mySum = mySum + smem[tid +  2];}
                         if (NUM_THREADS >=   2) { smem[tid] = mySum = mySum + smem[tid +  1];}
                 }
         }
 }

 __device__ void reduceToMax(float* sdata, unsigned int tid){
  //Synchronize threads to share shared memory data
  __syncthreads();
  float mySum = sdata[tid];

  // do reduction in shared mem
  if (NUM_THREADS >= 512) {
    if (tid < 256) { 
      sdata[tid] = mySum = fmaxf(mySum, sdata[tid + 256]); 
    } 
    __syncthreads();
  }
  if (NUM_THREADS >= 256) { 
    if (tid < 128) { 
      sdata[tid] = mySum = fmaxf(mySum, sdata[tid + 128]);
    } 
    __syncthreads();
  }
  if (NUM_THREADS >= 128) {
    if (tid <  64) {
      sdata[tid] = mySum = fmaxf(mySum, sdata[tid +  64]);
    }
    __syncthreads();
  }

  if (NUM_THREADS == 32){
    if (tid < 16) {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (NUM_THREADS >=  32) { smem[tid] = mySum = fmaxf(mySum, smem[tid + 16]);}
      if (NUM_THREADS >=  16) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  8]);}
      if (NUM_THREADS >=   8) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  4]);}
      if (NUM_THREADS >=   4) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  2]);}
      if (NUM_THREADS >=   2) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  1]);}
    }
  } else {
    if (tid < 32){
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (NUM_THREADS >=  64) { smem[tid] = mySum = fmaxf(mySum, smem[tid + 32]);}
      if (NUM_THREADS >=  32) { smem[tid] = mySum = fmaxf(mySum, smem[tid + 16]);}
      if (NUM_THREADS >=  16) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  8]);}
      if (NUM_THREADS >=   8) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  4]);}
      if (NUM_THREADS >=   4) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  2]);}
      if (NUM_THREADS >=   2) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  1]);}
    }
  }
}

__global__ void kMinimum(float* mat1, float* mat2, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        if (mat2[i] <  mat1[i])
           target[i] = mat2[i];
        else
           target[i] = mat1[i];
    }
}

__global__ void kMaximum(float* mat1, float* mat2, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        if (mat2[i] >  mat1[i])
           target[i] = mat2[i];
        else
           target[i] = mat1[i];
    }
}


/* Added by Deep Jaitly. Feb 19 */
__global__ void kThresholdBelow(float* mat, float threshold, float* target, unsigned int len)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) 
    {
    	if (mat[i] < threshold)
    		target[i] = threshold ; 
    	else
    		target[i] = mat[i] ;
    	
    }
}

__global__ void kThresholdAbove(float* mat, float threshold, float* target, unsigned int len)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) 
    {
    	if (mat[i] > threshold)
    		target[i] = threshold ; 
    	else
    		target[i] = mat[i] ;
    	
    }
}

/* Added by Deep Jaitly. Jan 7, 2011 */
__global__ void kRound(float* mat, float threshold, float* target, unsigned int len)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) 
    {
       int elem = (int) (mat[i]+.5) ; 
    	 target[i] = elem ; 
    }
}
/* Added by Deep Jaitly. Feb 19 */
__global__ void kReplicateColVector(float* mat, float* vec, unsigned int width,
                              unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        mat[i] = vec[i % height];
    }
}

__global__ void kGetColumnVectors(float *mat, float *order, float *target, 
		unsigned int start, unsigned int end, 
		unsigned int width, unsigned int height, 
		unsigned int widthSrc) /* DJ March 1 */
{

    /* We have to copy columns from mat into target, in the order
	defined by vector order, from index start to end...*/

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) 
    {
    	unsigned int srcColumnIdx = i/height ;
    	unsigned int srcColumn = order[start+srcColumnIdx] ;
    	if (srcColumn < widthSrc)
    	{
    		/* 
    		Otherwise calling program has a bug. 
    		Need to report it somehow but do not know yet, how this is done.
    		For now, just trying to prevent a crash and hope that the user sees that
    		his program has horrible results.
    		 */
	    	unsigned int srcRow =  (i%height) ; 
	        target[i] = mat[srcColumn*height+srcRow];
	    }
    }
}

__global__ void kSetColumnVectors(float *mat, float *order, float *src, 
		unsigned int start, unsigned int end, 
		unsigned int width, unsigned int height, 
		unsigned int width_tgt) /* DJ Nov 13, 12. */
{

    /* We have to copy columns from src into mat in the order
	defined by vector order, from index start to end...*/

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) 
    {
    	unsigned int srcColumnIdx = i/height ;
    	unsigned int tgtColumn = order[start+srcColumnIdx] ;
    	if (tgtColumn < width_tgt)
    	{
	    	unsigned int tgtRow =  (i%height) ; 
	      mat[tgtColumn*height+tgtRow] = src[i] ;
	    }
    }
}

__global__ void kCopyReorderedColumnVectorsToColumnVector(float *mat, float *srcOrder, 
      float *target, float *targetOrder,
		unsigned int start, unsigned int end, 
		unsigned int width, unsigned int height, 
		unsigned int widthSrc) /* DJ April 19, 2011 */
{
    /* We have to copy columns from mat into target, in the order
	defined by vector srcOrder, from index start to end, 
        to the order targetOrder ...*/

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) 
    {
    	unsigned int srcColumnIdx = i/height ;
    	unsigned int srcColumn = srcOrder[start+srcColumnIdx] ;
    	unsigned int targetColumn = targetOrder[start+srcColumnIdx] ;
    	if (srcColumn < widthSrc)
    	{
    		/* 
    		Otherwise calling program has a bug. 
    		Need to report it somehow but do not know yet, how this is done.
    		For now, just trying to prevent a crash and hope that the user sees that
    		his program has horrible results.
    		 */
	    	unsigned int srcRow =  (i%height) ; 
	        target[targetColumn*height+srcRow] = mat[srcColumn*height+srcRow];
	}
    }
}

__global__ void kCopySubsequences(float *mat, float *order, float *target,
      unsigned int start, unsigned int end, unsigned int subseqLength, unsigned int totalLength)
      /* DJ Oct 11 2010 */
{

    const unsigned int numThreads = blockDim.x ;
    unsigned int dataIndex = order[start+blockIdx.x] + threadIdx.x ;
    const unsigned int targetIndex = blockIdx.x * subseqLength ;

    for (unsigned int i = threadIdx.x ; i < subseqLength && dataIndex < totalLength ;
                                i += numThreads, dataIndex += numThreads)
    {
           target[targetIndex+i] = mat[dataIndex] ;
    }
}

__global__ void kAddMatrixMult(float *orig, float* a, float* b, float* dest, float alpha, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
      /* DJ Jan 31 2011 */

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = orig[i] + alpha * a[i] * b[i];
    }
}

__global__ void kResampleColumns(float *orig, float *target, int height, float newRate)
      /* DJ April 30 2011 */
{
    /* Resample columns of original matrix into new matrix at a new rate. Use linear interpolation.*/
    /* Not the best algo because sequential fetching is not guaranteed. Should probably prefetch a column
       and resample rather than doing this. But this code took 5 minutes to write and test:) */

    const unsigned int colIdx = blockIdx.x ; 
    const unsigned int rowIdx = threadIdx.x ;
    const unsigned int numThreads = blockDim.x;

    unsigned int colOffset = colIdx * height ; 
    float targetPos = rowIdx * newRate ;
    float movePos = numThreads * newRate ; 
    unsigned int srcRowLow = (unsigned int)targetPos ;
    unsigned int srcRowHigh = srcRowLow+1 ; 

    float lowContr = 0 ; 
    float highContr = 0 ; 
    for (unsigned int i = rowIdx; i < height; i += numThreads)
    {
      lowContr = 0 ; 
      highContr = 0 ; 
      if (srcRowLow < height-1) {
         lowContr = (1.0+srcRowLow-targetPos)*orig[colOffset + srcRowLow] ; 
         highContr = (1.0-srcRowHigh+targetPos)*orig[colOffset + srcRowHigh] ; 
         target[colOffset + i] = lowContr + highContr ; 
      } else {
         target[colOffset+i] = orig[colOffset + height-1] ;
      }
      targetPos += movePos ; 
      srcRowLow = (unsigned int)targetPos ; 
      srcRowHigh = srcRowLow+1 ; 
    }

}

__global__ void kResampleColumnsVect(float *orig, float *target, int height, float *rateVect)
      /* DJ April 30 2011 */
{
    /* Resample columns of original matrix into new matrix at a new rate. Use linear interpolation.*/
    /* Not the best algo because sequential fetching is not guaranteed. Should probably prefetch a column
       and resample rather than doing this. But this code took 5 minutes to write and test:) */

    const unsigned int colIdx = blockIdx.x ; 
    const unsigned int rowIdx = threadIdx.x ;
    const unsigned int numThreads = blockDim.x;

    float newRate = rateVect[colIdx] ; 

    unsigned int colOffset = colIdx * height ; 
    float targetPos = rowIdx * newRate ;
    float movePos = numThreads * newRate ; 
    unsigned int srcRowLow = (unsigned int)targetPos ;
    unsigned int srcRowHigh = srcRowLow+1 ; 

    float lowContr = 0 ; 
    float highContr = 0 ; 
    for (unsigned int i = rowIdx; i < height; i += numThreads)
    {
      lowContr = 0 ; 
      highContr = 0 ; 
      if (srcRowLow < height-1) {
         lowContr = (1.0+srcRowLow-targetPos)*orig[colOffset + srcRowLow] ; 
         highContr = (1.0-srcRowHigh+targetPos)*orig[colOffset + srcRowHigh] ; 
         target[colOffset + i] = lowContr + highContr ; 
      } else {
         target[colOffset+i] = orig[colOffset + height-1] ;
      }
      targetPos += movePos ; 
      srcRowLow = (unsigned int)targetPos ; 
      srcRowHigh = srcRowLow+1 ; 
    }

}

__global__ void kResampleColumnsVectGrad(float *orig, float *orig_diff, 
                                         float *orig_grad, int height, 
                                         float *rateVect)
      /* DJ Nov 6 2011 */
{

    const unsigned int colIdx = blockIdx.x ; 
    const unsigned int rowIdx = threadIdx.x ;
    const unsigned int numThreads = blockDim.x;

    extern __shared__ float thread_gradients[] ; 
    float *thread_indices =  &thread_gradients[2*numThreads];
    thread_indices[threadIdx.x] = -1;

    float newRate = rateVect[colIdx] ; 

    unsigned int colOffset = colIdx * height ; 
    float j_scale = rowIdx * newRate ;
    unsigned int i = (unsigned int)j_scale ;
    float movePos = numThreads * newRate ; 

    float low_grad = 0 ; 
    float high_grad = 0 ; 
    for (unsigned int j = rowIdx; j < height && i < height ; j += numThreads)
    {
      low_grad = 0 ; 
      high_grad = 0 ; 
      thread_indices[threadIdx.x] = i;
      low_grad = (1.0+i-j_scale) * orig_diff[colOffset + j] ; 
      high_grad = (j_scale-i) * orig_diff[colOffset + j] ; 

      thread_gradients[rowIdx*2] = low_grad;
      thread_gradients[rowIdx*2+1] = high_grad;

      // now add thread_gradients. This is really single threaded and hence
      // going to be slow!
      __syncthreads();
      if (threadIdx.x < 2)
      {
         for (unsigned int idx = 0; idx < numThreads ; idx++)
         {
            int cur_i = thread_indices[idx];
            if (cur_i != -1)
            {
               orig_grad[colOffset + cur_i + threadIdx.x] += thread_gradients[2*idx+threadIdx.x];
            }
         }
      }
      __syncthreads();

      j_scale += movePos ; 
      i = (unsigned int)j_scale ; 
      thread_indices[threadIdx.x] = -1;
    }
}

__global__ void kShiftColumns(float *orig, float *target, int height, float *shiftVect)
      /* DJ April 30 2011 */
{
    /* Shift data in the columns by given values. Use linear interpolation for non-integer
       values.
    */

    const unsigned int colIdx = blockIdx.x ; 
    const unsigned int rowIdx = threadIdx.x ;
    const unsigned int numThreads = blockDim.x;

    float shift = shiftVect[colIdx] ; 

    unsigned int colOffset = colIdx * height ; 
    float srcPos = rowIdx - shift;
    unsigned int srcRowLow = (unsigned int)srcPos ;
    unsigned int srcRowHigh = srcRowLow+1 ; 

    float lowContr = shift - (int)shift ; 
    float highContr = 1-lowContr ; 
    for (unsigned int i = rowIdx; i < height; i += numThreads)
    {
      if (srcRowLow < height-1) {
         float lo = lowContr * orig[colOffset + srcRowLow] ; 
         float hi = highContr * orig[colOffset + srcRowHigh] ; 
         target[colOffset + i] = lo + hi ; 
      } else {
         target[colOffset+i] = orig[colOffset + height-1] ;
      }
      srcRowLow += numThreads ; 
      srcRowHigh += numThreads ; 
    }

}



__global__ void kResampleColumnsVectGradRate(float *orig, float *orig_diff, 
                                         int height, float *rateVect,
                                         float *rateVectGrad)
      /* DJ Nov 6 2011 */
{

    const unsigned int colIdx = blockIdx.x ; 
    const unsigned int rowIdx = threadIdx.x ;
    const unsigned int numThreads = blockDim.x;

    extern __shared__ float thread_gradients[] ; 

    float newRate = rateVect[colIdx] ; 

    unsigned int colOffset = colIdx * height ; 

    float grad_sum = 0 ; 
    for (unsigned int j = rowIdx; j < height ; j += numThreads)
    {
      float j_scale = j * newRate ;
      unsigned int i = (unsigned int)j_scale ;
      if (i >= height-1)
         break;
      float diff = -orig[colOffset + i];
      if (i < height-1)
         diff += orig[colOffset + i+1] ; 

      grad_sum += j * orig_diff[colOffset + j] * diff ;
    }
    thread_gradients[rowIdx] = grad_sum;

    // sum it up here. 
    reduceToSumLocal(thread_gradients, rowIdx);
    __syncthreads();
    if (rowIdx == 0)
    {
       rateVectGrad[colIdx] = thread_gradients[0];
    }
}

__global__ void kSampleSoftMaxOld(float* mat, float* rand,  float* target, 
                               unsigned int height) {
  __shared__ float max_vals[32];
  __shared__ unsigned int max_indices[32];
  float cur_max = -FLT_MAX;
  unsigned int cur_max_index = 0;
  float val = 0;

  for (unsigned int i = threadIdx.x; i < height; i += 32) {
    val = mat[blockIdx.x * height + i] - log(-log(rand[blockIdx.x * height + i]));
    if (val > cur_max) {
        cur_max = val;
        cur_max_index = i;
    }
  }

  max_vals[threadIdx.x] = cur_max;
  max_indices[threadIdx.x] = cur_max_index;

  __syncthreads();

  if (threadIdx.x == 0) {
    cur_max = -FLT_MAX;

    for (unsigned int i = 0; i < 32; i++) {
        if (max_vals[i] > cur_max) {
            cur_max = max_vals[i];
            cur_max_index = max_indices[i];
        }
    }
    target[blockIdx.x * height + cur_max_index] = 1;
  }
}

__global__ void kSampleSoftMax(float* mat, float* rand,  float* target,
                               unsigned int height) {
  __shared__ float max_vals[32];
  __shared__ float max_indices[32];
  float cur_max = -FLT_MAX, cur_max_cpy = -FLT_MAX;
  int cur_max_index = -1;
  float val = 0;
 
  for (unsigned int i = threadIdx.x; i < height; i += 32) {
    val = mat[blockIdx.x * height + i] - log(-log(rand[blockIdx.x * height + i]));
    if (val > cur_max) {
      cur_max = val;
      cur_max_index = i;
    }
  }

  cur_max_cpy = cur_max;
  max_vals[threadIdx.x] = cur_max;
  max_indices[threadIdx.x] = cur_max_index;

  reduceToMax(max_vals, threadIdx.x);
  __syncthreads();
  cur_max = max_vals[0] ; 

  // if the value a thread is tracking is not 
  // maximum, lets just set its index to a negative
  // value.
  if (cur_max_cpy != cur_max)
    max_indices[threadIdx.x] = -1;

  // get the maximum index and use that as the sample.
  __syncthreads();
  reduceToMax(max_indices, threadIdx.x);
  __syncthreads();
  if (threadIdx.x == 0) {
     cur_max_index = max_indices[0];
     target[blockIdx.x * height + cur_max_index] = 1;
  }
}

__global__ void kSoftMax(float* mat, float* target, unsigned int width,
                         unsigned int height) {
    extern __shared__ float max_vals[] ;
    float cur_max = -FLT_MAX;
    float val = 0;

    float *cur_data = &mat[blockIdx.x * height] ; 
    max_vals[threadIdx.x]=-FLT_MAX;

    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
        val = cur_data[i];
        if (val > cur_max) {
            cur_max = val;
        }
    }
    max_vals[threadIdx.x] = cur_max;

    reduceToMax(max_vals, threadIdx.x);
    __syncthreads();
    cur_max = max_vals[0] ; 
    __syncthreads();
    val = 0;
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
        val += exp(cur_data[i]-cur_max);
    }

    max_vals[threadIdx.x] = val;
    reduceToSumLocal(max_vals, threadIdx.x);
    __syncthreads();
    float norm = max_vals[0] ;
    float *cur_target = &target[blockIdx.x * height] ; 
    for (unsigned int i = threadIdx.x; i < height; i += blockDim.x) {
        cur_target[i] = exp(cur_data[i]-cur_max) / norm ;
    }
}

__global__ void kArgMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height) {
    __shared__ float max_vals[32];
    __shared__ unsigned int max_indices[32];
    float cur_max = -FLT_MAX;
    unsigned int cur_max_index = 0;
    float val = 0;

    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        max_indices[i]=0;
        max_vals[i]=-FLT_MAX;
    }
 
    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[blockIdx.x * height + i];
        if (val > cur_max) {
            cur_max = val;
            cur_max_index = i;
        }
    }

    max_vals[threadIdx.x] = cur_max;
    max_indices[threadIdx.x] = cur_max_index;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = -FLT_MAX;
        cur_max_index = 0;

        for (unsigned int i = 0; i < 32; i++) {
            if (max_vals[i] > cur_max) {
                cur_max = max_vals[i];
                cur_max_index = max_indices[i];
            }
        }
        target[blockIdx.x] = cur_max_index;
    }
}

__global__ void kArgMaxIndicatorColumnwise(float* mat, float* target, unsigned int width, unsigned int height) {
    __shared__ float max_vals[32];
    __shared__ unsigned int max_indices[32];
    float cur_max = -FLT_MAX;
    unsigned int cur_max_index = 0;
    float val = 0;

    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        max_indices[i]=0;
        max_vals[i]=-FLT_MAX;
    }
 
    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[blockIdx.x * height + i];
        target[blockIdx.x * height + i] = 0;
        if (val > cur_max) {
            cur_max = val;
            cur_max_index = i;
        }
    }

    max_vals[threadIdx.x] = cur_max;
    max_indices[threadIdx.x] = cur_max_index;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = -FLT_MAX;
        cur_max_index = 0;

        for (unsigned int i = 0; i < 32; i++) {
            if (max_vals[i] > cur_max) {
                cur_max = max_vals[i];
                cur_max_index = max_indices[i];
            }
        }
        target[blockIdx.x * height + cur_max_index] = 1;
    }
}

/* Compute the gradients wrt input activations, of elements
   of a matrix of softmaxs, given the outputs and the probabilities
*/
__global__ void kLogisticGrad(float* prob, float *out_grad,
                              float* in_grad, unsigned int len)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        in_grad[i] = out_grad[i] * prob[i] * (1 - prob[i]) ;
    }
}

__global__ void kSoftmaxGrad(float* prob, float* out_grad,
                        float* in_grad, unsigned int height)
{
    extern  __shared__ float grad_prob_sum[] ;
    const unsigned int num_threads = blockDim.x; 
    grad_prob_sum[threadIdx.x]= 0;

    float *probs_cur = &prob[blockIdx.x*height] ;
    float *out_grad_cur = &out_grad[blockIdx.x*height] ;
    float *in_grad_cur = &in_grad[blockIdx.x*height] ;

    for (unsigned int i = threadIdx.x; i < height; i += num_threads) {
        grad_prob_sum[threadIdx.x] += probs_cur[i] * out_grad_cur[i] ;
        in_grad_cur[i] = out_grad_cur[i] ;
    }

    __syncthreads();
   reduceToSumLocal(grad_prob_sum, threadIdx.x);
    __syncthreads();
   float sum_grad_prob = grad_prob_sum[0];

    for (unsigned int i = threadIdx.x; i < height; i += num_threads) {
        in_grad_cur[i] -= sum_grad_prob ;
        in_grad_cur[i] *= probs_cur[i] ;
    }
}

__global__ void kColumnwiseDot(float* mat1, float* mat2, 
                              float* target, unsigned int num_dims)
{
  const unsigned int num_threads = blockDim.x; 
  extern  __shared__ float sum_prod[] ;

  sum_prod[threadIdx.x]= 0;

  float *mat1_cur = &mat1[blockIdx.x*num_dims] ;
  float *mat2_cur = &mat2[blockIdx.x*num_dims] ;

  for (unsigned int i = threadIdx.x; i < num_dims; i += num_threads) {
    sum_prod[threadIdx.x] += mat1_cur[i] * mat2_cur[i] ;
  }

  __syncthreads();
  reduceToSumLocal(sum_prod, threadIdx.x);
  __syncthreads();
  target[blockIdx.x] = sum_prod[0] ;
}

__global__ void kLogisticLogProb(float* outputs, float* targets, 
                              float* log_probs, unsigned int num_dims)
{
  const unsigned int num_threads = blockDim.x; 
  extern  __shared__ float sum_t_log_p[] ;

  sum_t_log_p[threadIdx.x]= 0;

  float *outputs_cur = &outputs[blockIdx.x*num_dims] ;
  float *targets_cur = &targets[blockIdx.x*num_dims] ;

  for (unsigned int i = threadIdx.x; i < num_dims; i += num_threads) {
    float p = outputs_cur[i] ; 
    float t = targets_cur[i] ; 
    sum_t_log_p[threadIdx.x] += t * log(p+1e-8) + (1-t) * log(1-p+ 1e-8) ; 
  }

  __syncthreads();
  reduceToSumLocal(sum_t_log_p, threadIdx.x);
  __syncthreads();
  log_probs[blockIdx.x] = sum_t_log_p[0] ;
}

__global__ void kSoftmaxAccuraccy(float* probs, float* targets,
                                  float* log_prob, float* correct, 
                                  unsigned int num_dims)
{ 
  const unsigned int num_threads = blockDim.x; 
  extern  __shared__ float shared_mem[] ;
  float *sum_t_log_p = &shared_mem[0] ;
  float *correct_class_prob = (float *) &shared_mem[num_threads] ;
  int *correct_class = (int *) &shared_mem[num_threads+1] ;

  sum_t_log_p[threadIdx.x]= 0;

  float *probs_cur = &probs[blockIdx.x*num_dims] ;
  float *targets_cur = &targets[blockIdx.x*num_dims] ;

  for (unsigned int i = threadIdx.x; i < num_dims; i += num_threads) {
    sum_t_log_p[threadIdx.x] += log(probs_cur[i]+1e-35) * targets_cur[i] ;
    // allow for some slack in the probabilities.
    if (targets_cur[i] > .99)
    { 
         // only one thread should get here, if this is really
        // a multinomial target.
        *correct_class = i ; 
        *correct_class_prob = probs_cur[i];
    }
  }

  __syncthreads();
  reduceToSumLocal(sum_t_log_p, threadIdx.x);
  __syncthreads();
  if (threadIdx.x == 0)
     log_prob[blockIdx.x] = sum_t_log_p[0] ;

  // now lets check if the correct_class is the max value. 
  // reusing shared memory. 
  __syncthreads();
  float class_prob = *correct_class_prob; 
  sum_t_log_p[threadIdx.x]= 0;
  for (unsigned int i = threadIdx.x; i < num_dims; i += num_threads) {
    if (probs_cur[i] >= class_prob)
      sum_t_log_p[threadIdx.x]++ ;
  }

  __syncthreads();
  reduceToSumLocal(sum_t_log_p, threadIdx.x);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (sum_t_log_p[0] > 1)
      correct[blockIdx.x] = 0;
    else
      correct[blockIdx.x] = 1;
  }
}

__global__ void kSoftmaxAccuraccyVect(float* probs, float* targets,
                                  float* log_prob, float* correct, 
                                  unsigned int num_dims)
{ 
  const unsigned int num_threads = blockDim.x; 
  extern  __shared__ float shared_mem[] ;

  float *probs_cur = &probs[blockIdx.x*num_dims] ;
  int correct_class = targets[blockIdx.x] ;
  float correct_class_prob = probs_cur[correct_class];

  if (threadIdx.x == 0) {
    log_prob[blockIdx.x] = log(correct_class_prob);
  }
  __syncthreads();

  // now lets check if the correct_class is the max value. 
  // reusing shared memory. 
  shared_mem[threadIdx.x]= 0;
  for (unsigned int i = threadIdx.x; i < num_dims; i += num_threads) {
    if (probs_cur[i] < correct_class_prob)
      shared_mem[threadIdx.x]++ ;
  }
  reduceToSumLocal(shared_mem, threadIdx.x);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (shared_mem[0] == num_dims-1)
      correct[blockIdx.x] = 1;
    else
      correct[blockIdx.x] = 0;
  }
}

__global__ void kDropout(unsigned int* rndMults, unsigned long long* rndWords,
                         float* gData, unsigned int numElements,
                         float drop_percent) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        float val = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        if (val < drop_percent)
           gData[i] = 0 ;
    }
    rndWords[idx] = rndWord;
}

__global__ void kNormalizeColumns(float *mat, float *tgt, 
                                     unsigned int height){
  const unsigned int num_threads = blockDim.x; 
  extern  __shared__ float ss[] ;
  ss[threadIdx.x]= 0;
  unsigned int start_index = height * blockIdx.x;

  for (unsigned int i = threadIdx.x; i < height; i += num_threads) {
    ss[threadIdx.x] += mat[start_index+i] * mat[start_index+i];
  }
  __syncthreads();
  reduceToSumLocal(ss, threadIdx.x);
  __syncthreads();
  float norm = sqrt(ss[0]);
  __syncthreads();
  // renormalize.
  for (unsigned int i = threadIdx.x; i < height; i += num_threads) {
    tgt[start_index+i] = mat[start_index+i] / norm ;
  }
}
__global__ void kThesholdColumnNorms(float *mat, float *biases, float *tgt, 
                                     unsigned int height, float threshold) {
  const unsigned int num_threads = blockDim.x; 
  extern  __shared__ float ss[] ;
  ss[threadIdx.x]= 0;
  unsigned int start_index = height * blockIdx.x;

  for (unsigned int i = threadIdx.x; i < height; i += num_threads) {
    ss[threadIdx.x] += mat[start_index+i] * mat[start_index+i];
  }
  __syncthreads();
  reduceToSumLocal(ss, threadIdx.x);
  __syncthreads();
  float norm = sqrt(ss[0]);
  if (norm <= threshold)
     return ;

  __syncthreads();

  // renormalize.
  biases[blockIdx.x] = biases[blockIdx.x] * threshold / norm ;
  for (unsigned int i = threadIdx.x; i < height; i += num_threads) {
    tgt[start_index+i] = mat[start_index+i] * threshold / norm ;
  }
}

__global__ void kSetDiagonal(float *mat, float *diag, unsigned int side) {
  for (int i = threadIdx.x; i < side; i+= blockDim.x) {
    mat[side*i+i] = diag[i];
  }
}

__global__ void kReverseColumnEntries(float *mat, float *target, int h) {
  float *src = &mat[blockIdx.x*h];
  float *tgt = &target[blockIdx.x*h];
  for (int i = threadIdx.x; i < h/2; i+= blockDim.x) { 
    float first = src[i];
    float last = src[h-i-1];
    tgt[i] = last;
    tgt[h-i-1] = first;
  }
}

__global__ void kSumColumns(float *mat, float *target, int h, int num_columns) {
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= h)
    return;

  float sum = 0;
  for (int col = 0; col < num_columns; col++) { 
    sum += mat[index];
    index += h;
  }
  target[blockIdx.x*blockDim.x + threadIdx.x] = sum;
}

__global__ void kApplyLog1PlusAbs(float* mat, float* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    float mat_i;

    for (unsigned int i = idx; i < len; i += numThreads) {
        mat_i = mat[i];
        if (mat_i > 0)
            target[i] = __logf(1 + mat_i);
        else
            target[i] = __logf(1 - mat_i);
    }
}

__global__ void kApplyLog1PlusAbsGrad(float* mat, float *out_grad, 
                                      float* act_grad, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    float mat_i, out_grad_i;

    for (unsigned int i = idx; i < len; i += numThreads) {
        mat_i = mat[i];
        out_grad_i = out_grad[i];
        if (mat_i > 0)
            act_grad[i] = out_grad_i / (1 + mat_i);
        else
            act_grad[i] = -out_grad_i / (1 - mat_i);
    }
}


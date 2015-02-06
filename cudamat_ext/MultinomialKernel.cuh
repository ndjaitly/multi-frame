/*************************** TAKEN FROM ALEX KRIZHEVSKY ******************************************/
/**************************** Look for CudaConv at http://www.cs.toronto.edu/~kriz/ */

/*
 * The compiler is supposed to convert i / 16 to i >> 4, so
 * why is i >> 4 still so much faster?
 */
#define IDX(i) ((i) + ((i) >> 4))
//#define IDX(i) (i)

/*http://www.cs.toronto.edu/~kriz/
 * Samples from a bunch of multinomial distributions, where each row of data
 * is a different distribution.
 *
 * Uses the scan algorithm from these slides http://www.eecg.toronto.edu/~moshovos/CUDA08/slides/007%20-%20Scans.ppt
 * to compute prefix-sums.
 */
template<int bX>
__global__ void kSampleMultinomial(float* data, float* randoms, float* targets, const int multiSize, const int numMulti)  {
    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    data += multiSize * bidx  + threadIdx.x;
    targets += multiSize * bidx  + threadIdx.x;
    __shared__ float shmem[IDX(bX * 2 + 1)];
    __shared__ float rand;

    if (bidx >= numMulti)
        return;

    shmem[IDX(threadIdx.x)] = 0;
    shmem[IDX(threadIdx.x + bX)] = 0;
    if (threadIdx.x < multiSize) {
        shmem[IDX(threadIdx.x)] = data[0]; // load input into shared memory
        if (threadIdx.x + bX < multiSize) {
            shmem[IDX(threadIdx.x + bX)] = data[bX];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        rand = randoms[bidx];
    }
    /*=============================================================
     * Reduction
     */
    int ai = 2 * threadIdx.x;
    int bi = ai + 1;
    if (bX >= 512) {
        __syncthreads();

        shmem[IDX(bi)] += shmem[IDX(ai)];

        ai = (ai << 1) + 1;
        bi = (bi << 1) + 1;
    }

    if (bX >= 256) {
        __syncthreads();
        if (threadIdx.x < 256) {
            shmem[IDX(bi)] += shmem[IDX(ai)];
        }
        ai = (ai << 1) + 1;
        bi = (bi << 1) + 1;
    }

    if (bX >= 128) {
        __syncthreads();
        if (threadIdx.x < 128) {
            shmem[IDX(bi)] += shmem[IDX(ai)];
        }
        ai = (ai << 1) + 1;
        bi = (bi << 1) + 1;
    }

    if (bX >= 64) {
        __syncthreads();
        if (threadIdx.x < 64) {
            shmem[IDX(bi)] += shmem[IDX(ai)];
        }
        ai = (ai << 1) + 1;
        bi = (bi << 1) + 1;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 16) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 8) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 4) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 2) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 1) {
        shmem[IDX(bi)] += shmem[IDX(ai)];

        /*=============================================================
         * Scan
         */
        shmem[IDX(bX * 2 - 1)] = 0;

        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 2) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 4) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 8) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 16) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 32) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }

    if (bX >= 64) {
        __syncthreads();
        ai >>= 1;
        bi >>= 1;
        if (threadIdx.x < 64) {
            const float t = shmem[IDX(ai)];
            shmem[IDX(ai)] = shmem[IDX(bi)];
            shmem[IDX(bi)] += t;
        }
    }

    if (bX >= 128) {
        __syncthreads();
        ai >>= 1;
        bi >>= 1;
        if (threadIdx.x < 128) {
            const float t = shmem[IDX(ai)];
            shmem[IDX(ai)] = shmem[IDX(bi)];
            shmem[IDX(bi)] += t;
        }
    }

    if (bX >= 256) {
        __syncthreads();
        ai >>= 1;
        bi >>= 1;
        if (threadIdx.x < 256) {
            const float t = shmem[IDX(ai)];
            shmem[IDX(ai)] = shmem[IDX(bi)];
            shmem[IDX(bi)] += t;
        }
    }

    if (bX >= 512) {
        __syncthreads();
        ai >>= 1;
        bi >>= 1;
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }

    __syncthreads();
    if (threadIdx.x < multiSize) {
        shmem[IDX(threadIdx.x)] += data[0]; // load input into shared memory
        if (threadIdx.x + bX < multiSize) {
            shmem[IDX(threadIdx.x + bX)] += data[bX];
        }
    }
    __syncthreads();
    if (threadIdx.x < multiSize) {
        const float prev = threadIdx.x == 0 ? 0 : shmem[IDX(threadIdx.x - 1)];
        targets[0] = rand >= prev && rand < shmem[IDX(threadIdx.x)];
//        targets[0] = shmem[IDX(threadIdx.x)];

        if (threadIdx.x + bX < multiSize) {
            targets[bX] = rand >= shmem[IDX(threadIdx.x - 1 + bX)] && rand < shmem[IDX(threadIdx.x + bX)];
//            targets[bX] = shmem[IDX(threadIdx.x + bX)];
        }
    }
}
#define SSM_THREADS_X   16
#define SSM_THREADS_Y   32
#define SSM_LOOPS_Y     16
/*
 * This routine is just always faster than the fancy tree-based one above...
 * Oh ok, not in all cases. In the cases when the number of distributions
 * that you want to sample from (height) is fairly large.
 *
 * TODO: revisit this routine cause that doWrite statement is too long
 * and it all can probably be simplified if i control the block size at run-time
 */
template <int LOOPS_X, int SUM_WIDTH_UPPERBOUND>
__global__ void kSampleSmallMultinomial(float* multi, float* randoms, float* targets, const int width, const int height) {
    const int shmemX = SSM_THREADS_X + 1;
    __shared__ float shmem[SSM_THREADS_Y*shmemX];

//    const int LOOPS_X = DIVUP(width, AGG_SHORT_ROWS_THREADS_X);

    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int blockRowIdx = bidx * SSM_LOOPS_Y * SSM_THREADS_Y;

    if(blockRowIdx < height) {
        const int tidx = threadIdx.y * SSM_THREADS_X + threadIdx.x;
        int ty = LOOPS_X == 1 ? tidx / width : threadIdx.y;
        const int tx = LOOPS_X == 1 ? tidx % width : threadIdx.x;
        float* shmemWrite = shmem + MUL24(ty, shmemX) + tx;
        //    targets += blockIdx.y * AGG_SHORT_ROWS_LOOPS_Y * AGG_SHORT_ROWS_THREADS_Y + tidx;
        const int dataOffset = width * blockRowIdx + MUL24(ty, width) + tx;
        multi += dataOffset;
        targets += dataOffset;

        float* shmemWriteZeros = &shmem[MUL24(threadIdx.y,shmemX) + threadIdx.x];
//        ty += blockRowIdx;
//#pragma unroll
        for (int y = 0; y < SSM_LOOPS_Y*SSM_THREADS_Y; y += SSM_THREADS_Y) {
//            if (y * AGG_SHORT_ROWS_THREADS_Y + idxY >= height) {
//                return; // we're done here
//            }
            const bool doSum = tidx < SSM_THREADS_Y && tidx + y + blockRowIdx < height;
            float rnd;
            if (doSum) {
                rnd = randoms[tidx + y + blockRowIdx];
            }
            float accum = 0, accumPrev = 0;
//#pragma unroll // this causes > 16 registers to be used in some cases, avoid
            for(int x = 0; x < LOOPS_X * SSM_THREADS_X; x+= SSM_THREADS_X) {
                __syncthreads();
                shmemWriteZeros[0] = 0;
                if (LOOPS_X == 1) { // because the part we zeroed might not be same as one we're writing to
                    __syncthreads();
                }
                const bool doWrite = ty + blockRowIdx + y < height && (LOOPS_X > 1 || ty < SSM_THREADS_Y) && x + tx < width;
                if (doWrite) {
                    shmemWrite[0] = multi[y * width + x];
                }
                __syncthreads();

                if (doSum) {
                    float* shmemRead = shmem + MUL24(tidx, shmemX);

                    // this loops too much if the rows are really short :(
                    for (int i = 0; i < SUM_WIDTH_UPPERBOUND; i++) {
                        accumPrev = accum;
                        accum += shmemRead[0];
                        shmemRead[0] = rnd >= accumPrev && rnd < accum;
                        shmemRead++;
                    }
                }
                __syncthreads();
                if (doWrite) {
                    targets[y * width + x] = shmemWrite[0];
                }
            }
//            multi += width * SSM_THREADS_Y;
//            targets += width * SSM_THREADS_Y;
//            ty += SSM_THREADS_Y;
        }
    }
}

#endif /* CONV_UTIL_CUH_ */


import cudamat
from cudamat import generate_exception
from cudamat import sum, dot, vdot, sigmoid, tanh, \
                    abs, log_1_plus_exp, log, exp,  \
                    sqrt, pow, cuda_sync_threads,  \
                    reformat, cuda_set_device, cublas_init,  \
                    cublas_shutdown

import ctypes as ct

_cudamat = ct.cdll.LoadLibrary('libcudamat.so')
_cudamat_ext = ct.cdll.LoadLibrary('libcudamat_ext.so')
_cudamat_ext.column_dot.restype = ct.c_int
_cudamat_ext.maximum.restype = ct.c_int
_cudamat_ext.minimum.restype = ct.c_int
_cudamat_ext.get_column_vectors.restype = ct.c_int
_cudamat_ext.get_column_vectors.restype = ct.c_int
_cudamat_ext.CumulativeSum.restype = ct.c_int
_cudamat_ext.logistic_grad.restype = ct.c_int
_cudamat_ext.logistic_log_prob.restype = ct.c_int
_cudamat_ext.softmax_grad.restype = ct.c_int
_cudamat_ext.add_matrix_mult.restype = ct.c_int
_cudamat_ext.ResampleColumns.restype = ct.c_int
_cudamat_ext.ResampleColumnsVectGrad.restype = ct.c_int
_cudamat_ext.softmax_accuraccy.restype = ct.c_int

_cudamat_ext.create_rand_generator.restype = ct.c_void_p
_cudamat_ext.create_rand_generator.argtypes = []

_cudamat_ext.set_rand_generator_seed.restype = None
_cudamat_ext.set_rand_generator_seed.argtypes = [ct.c_void_p, ct.c_ulonglong]


_cudamat_ext.fill_with_randn.restype = ct.c_int
_cudamat_ext.fill_with_randn.argtypes = [ct.c_void_p, ct.c_void_p]

_cudamat_ext.fill_with_rand.restype = ct.c_int
_cudamat_ext.fill_with_rand.argtypes = [ct.c_void_p, ct.c_void_p]

class CUDAMatrix(cudamat.CUDAMatrix):
    pass 

    def transpose(self, target = None):
        """
        Return a transposed copy of the matrix.
        """
        if not target:
            target = empty((self.shape[1], self.shape[0]))

        err_code = _cudamat.copy_transpose(self.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    @staticmethod
    def init_random(seed = 0):
        """
        Initialize and seed the random number generator.
        """
        print "In cudamat_ext.init_random()"
        try:
            CUDAMatrix.rnd_state_p
            _cudamat_ext.free_device_memory(CUDAMatrix.rnd_state_p)
        except AttributeError:
            pass

        CUDAMatrix.rnd_state_p = _cudamat_ext.create_rand_generator()
        _cudamat_ext.set_rand_generator_seed(CUDAMatrix.rnd_state_p, seed)

       
    def fill_with_rand(self):
        """
        Fill matrix on the GPU with random numbers drawn from the uniform
        distribution over the (0,1) interval.
        """
        err_code = _cudamat_ext.fill_with_rand(self.p_mat, 
                                               CUDAMatrix.rnd_state_p)
        if err_code:
            raise generate_exception(err_code)
        return self

    def fill_with_randn(self):
        """
        Fill matrix on the GPU with random numbers drawn from the standard normal
        distribution.
        """
        err_code = _cudamat_ext.fill_with_randn(self.p_mat, 
                                            CUDAMatrix.rnd_state_p)
        if err_code:
            raise generate_exception(err_code)
        return self


    def load_matrix(self, array):
        """
        For a cudamat array that already exists, copy over new data from
        a numpy.ndarray. Must be of right size
        """
        assert(self.shape == array.shape)
        array = reformat(array)
        self.numpy_array = array
        self.free_device_memory()
        _cudamat.init_from_array(self.p_mat,
                           array.ctypes.data_as(ct.POINTER(ct.c_float)),
                           ct.c_int(array.shape[0]), ct.c_int(array.shape[1]))
        err_code = _cudamat.copy_to_device(self.p_mat)
        if err_code:
            raise generate_exception(err_code)
        self.T = cudamat.TransposedCUDAMatrix(self.mat)

        return self

    def minimum(self, val, target = None):
        """
        Perform the operation target = min(self, val)
        """

        if not target:
            target = self

        err_code = _cudamat_ext.minimum(self.p_mat, val.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def maximum(self, val, target = None):
        """
        Perform the operation target = max(self, val)
        """

        if not target:
            target = self

        err_code = _cudamat_ext.maximum(self.p_mat, val.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target


    def argmax(self, axis, target = None):
        """
        Implemented by Deep Jaitly
        Find the argmax value along the given dimension, where 0 represents the
        leading dimension and 1 represents the non-leading dimension. 
        """

        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))
 
        elif axis == 1:
            if not target:
                target = empty((m, 1))

        err_code =  _cudamat_ext.argmax_by_axis(self.p_mat, target.p_mat, ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def argmax_indicator(self, axis, target = None):
        """
        Find the argmax value along the given dimension, where 
        0 represents the leading dimension and 1 represents 
        the non-leading dimension. Set the corresponding element
        of target to 1.
        """

        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))
 
        elif axis == 1:
            if not target:
                target = empty((m, 1))

        err_code =  _cudamat_ext.argmax_indicator(self.p_mat, target.p_mat, ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target


    #  Changed name copy_reordered_column_vectors to get_column_vectors
    def get_column_vectors(self, order, start, end, target = None):
        """
        Copies columns from self into target. The columns are copied in the order
        specified by indices=order[start .. end] DJ March 1. 2010.
        """
        if target is None:
            target = empty((self.shape[0], end-start))
    
        if isinstance(order, cudamat.CUDAMatrix):
            orderMat = order
        else:
            orderMat = cudamat.CUDAMatrix(cudamat.reformat(order))
    
        err_code = _cudamat_ext.get_column_vectors(self.p_mat, \
                                    orderMat.p_mat, target.p_mat, \
                                    ct.c_int(start), ct.c_int(end))
    
        if err_code:
            raise generate_exception(err_code)
    
        
        return target

    def set_column_vectors(self, order, start, end, src):
        """
        Copies columns from src in self. The columns are copied in the order
        specified by indices=order[start .. end] DJ Nov 13. 2012.
        """
    
        if isinstance(order, cudamat.CUDAMatrix):
            orderMat = order
        else:
            orderMat = cudamat.CUDAMatrix(cudamat.reformat(order))
    
        err_code = _cudamat_ext.set_column_vectors(self.p_mat, \
                                    orderMat.p_mat, src.p_mat, \
                                    ct.c_int(start), ct.c_int(end))
    
        if err_code:
            raise generate_exception(err_code)
    
        
        return self


    def copy_reordered_column_vectors_to_reordered_columns(self, orderIn, orderOut, start, end, target):
        """
        Copies columns from self into target. The source columns are copied in the order
        specified by indices=orderIn[start .. end]. The target column orders are similarly
        specified by indices=orderOut[start .. end].  DJ April 19. 2011.
        """
        if not target:
            raise CUDAMatException("target not specified. target cannot be null")
    
        if isinstance(orderIn, CUDAMatrix):
            orderInMat = orderIn
        else:
            orderInMat = cudamat.CUDAMatrix(cudamat.reformat(orderIn))

        if isinstance(orderOut, CUDAMatrix):
            orderOutMat = orderOut
        else:
            orderOutMat = cudamat.CUDAMatrix(cudamat.reformat(orderOut))
    
        err_code = _cudamat_ext.copy_reordered_column_vectors_to_reordered_columns(\
                               self.p_mat,   orderInMat.p_mat,\
                               target.p_mat, orderOutMat.p_mat, \
                               ct.c_int(start), ct.c_int(end))
    
        if err_code:
            raise generate_exception(err_code)
    
        
        return self
    
    def copy_subsequences(self, order, start, end, seqLength, target):
        """
        Copies subsequences starting at indices specfied in order.
        i.e. specified by indices=order[start .. end]. Each of length = seqLength.
        """
        if not target:
            raise CUDAMatException("target not specified. target cannot be null")
    
        if isinstance(order, CUDAMatrix):
            orderMat = order
        else:
            orderMat = cudamat.CUDAMatrix(cudamat.reformat(order))
    
        err_code = _cudamat_ext.copy_subsequences(self.p_mat, \
                                    orderMat.p_mat, target.p_mat, \
                                    ct.c_int(start), ct.c_int(end),\
                                    ct.c_int(seqLength))
    
        if err_code:
            raise generate_exception(err_code)
    
    
        return self
    
    def replicate_col_vec(self, vec):
        """
        Copy vector vec to every column of the matrix. 
        """
    
        err_code = _cudamat_ext.replicate_col_vec(self.p_mat, vec.p_mat)
        if err_code:
            raise generate_exception(err_code)
    
        return self
    
    def threshold_below(self, threshold, target = None):
        """
        Perform the operation target = val>threshold ? val: threshold , where threshold is a scalar.
        """
    
        if not target:
            target = self
    
        if isinstance(threshold, (int, float)):
            err_code = _cudamat_ext.threshold_below(self.p_mat, ct.c_float(threshold), target.p_mat)
        else:
            raise TypeError("Incorrect type to function truncate_below: integet or float accepted only")
    
        if err_code:
            raise generate_exception(err_code)
    
        return target

    def threshold_above(self, threshold, target = None):
        """
        Perform the operation target = val<threshold ? val: threshold , where threshold is a scalar.
        """
    
        if not target:
            target = self
    
        if isinstance(threshold, (int, float)):
            err_code = _cudamat_ext.threshold_above(self.p_mat, ct.c_float(threshold), target.p_mat)
        else:
            raise TypeError("Incorrect type to function truncate_below: integet or float accepted only")
    
        if err_code:
            raise generate_exception(err_code)
    
        return target
   
 
    def round(mat, target = None):
        """
        Round each element of the matrix mat.
        """

        if not target:
            target = mat

        err_code = _cudamat_ext.apply_round(mat.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target


   
    def SampleSoftMax(self, unifRands, samples):
        ''' Takes softmax log(probabilities) along columns. A matrix of uniform random
            variable (0,1) samples in unifRands and generates multinomail samples..
            WARNING!!!! REMEMBER ITS LOG(PROBABILITIES) i.e. ACTIVATIONS that are
            input (in the self matrix).
        '''
    
        softMaxWidth, numCases = self.shape
    
    
        if samples.shape[0] != softMaxWidth or samples.shape[1] != numCases: 
           raise ValueError, "samples must be of same shape as input."

        samples.assign(0)
    
        if unifRands.shape[0] != softMaxWidth or unifRands.shape[1] != numCases: 
           raise ValueError, "Number of random numbers provided is not enough"

        err_code =  _cudamat_ext.sample_softmax(self.p_mat, unifRands.p_mat, samples.p_mat)

        if err_code:
            raise generate_exception(err_code)
    
        return samples

    #def sum(self, axis, target = None):
        #"""
        #Sum the matrix along the given dimension, where 0 represents the leading
        #dimension and 1 represents the non-leading dimension. If a target is
        #not prvided, a new vector is created for storing the result.
        #"""
        #return sum(self, axis, target)
            

    def compute_softmax(self, target = None):
        ''' Compute softmax probabilities along columns.
        '''
 
        if target is not None: 
            if target.shape != self.shape:
               raise ValueError, "target must be of same shape as input."
            err_code =  _cudamat_ext.softmax(self.p_mat, target.p_mat)
        else:
            err_code =  _cudamat_ext.softmax(self.p_mat, self.p_mat)

        if err_code:
            raise generate_exception(err_code)
    
        return target
    
    
    def CumulativeSum(self, cumulativeSums, sumWidth=-1):
        """
        Compute cumulative sums along columns of self, with the cumulative sum over
        blocks of width = sumWidth
        """
    
        signalLength, numFeatures = self.shape
    
        if (sumWidth == -1):
           sumWidth = signalLength
    
        if sumWidth > 1024:
           raise ValueError, "Kernel does not support more than 1024 points width in cumulative sum"
    
        if cumulativeSums.shape[0] != signalLength or cumulativeSums.shape[1] != numFeatures: 
           raise ValueError, "cumulativeSums must be of same shape as input."
    
        err_code =  _cudamat_ext.CumulativeSum(self.p_mat, cumulativeSums.p_mat, \
                                     ct.c_int(sumWidth))
    
        if err_code:
            raise generate_exception(err_code)
    
        return cumulativeSums
    
    
   
    def ResampleColumns(self, target, rate):

       if isinstance(rate, CUDAMatrix):
          err_code = _cudamat_ext.ResampleColumnsVect(self.p_mat, \
                                            target.p_mat, \
                                            rate.p_mat)
       elif isinstance(p, (int, float)):
          err_code = _cudamat_ext.ResampleColumns(self.p_mat, \
                                            target.p_mat, \
                                            ct.c_float(rate))
       else:
          raise ValueError, "Value must be of type CUDAMatrix, int, or float."


       if err_code:
          raise generate_exception(err_code)
        
       return self

    def ResampleColumnsGrad(self, grad_mult, grads, rate, rate_grad):

       err_code = _cudamat_ext.ResampleColumnsVectGrad(self.p_mat,
                                            grad_mult.p_mat,
                                            grads.p_mat,
                                            rate.p_mat,
                                            rate_grad.p_mat)


       if err_code:
          raise generate_exception(err_code)
        
       return self


    def add_matrix_mult(self, mat1, mat2, alpha=1.0, target = None):
        """
        Add mat1*mat2 to the matrix.
        """
        if target == None:
           target = self

        if not(mat1.shape[0] == mat2.shape[0] == self.shape[0] == target.shape[0]):
           raise ValueError, "Dimension error in add_matrix_mult"
        if not(mat1.shape[1] == mat2.shape[1] == self.shape[1] == target.shape[1]):
           raise ValueError, "Dimension error in add_matrix_mult"

        err_code = _cudamat_ext.add_matrix_mult(self.p_mat, mat1.p_mat, mat2.p_mat, target.p_mat, ct.c_float(alpha))
        if err_code:
            raise generate_exception(err_code)

        return self

    def dropout(self, dropout_percent):
        """
        Fill matrix on the GPU with random numbers drawn from the uniform
        distribution over the (0,1) interval.
        """

        err_code = _cudamat_ext.dropout(cudamat.CUDAMatrix.rnd_state_p, self.p_mat, 
                                        ct.c_float(dropout_percent))
        if err_code:
            raise generate_exception(err_code)

        return self

    def normalize_columns(self, target = None):
        """
        Normalize each column
        """
        if target == None:
           target = self
        
        err_code = _cudamat_ext.normalize_columns(self.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return self

    def threshold_column_norms(self, max_val, biases, target = None):
        """
        Normalize each column
        """
        if target == None:
           target = self
        
        err_code = _cudamat_ext.threshold_column_norms(self.p_mat, 
                                                       biases.p_mat, 
                                                       target.p_mat, 
                                                        ct.c_float(max_val))
        if err_code:
            raise generate_exception(err_code)

        return self


    def set_diagonal(self, diagonal):
        ''' Compute softmax probabilities along columns.
        '''
        if diagonal.shape[0] != 1 and diagonal.shape[1] != 1:
            raise ValueError, "Diagonal must be a column or row vector"

        if self.shape[0] != self.shape[1] or \
             self.shape[0] != (diagonal.shape[0]*diagonal.shape[1]):
               raise ValueError, "Matrix needs to be square, and diagonal \
                                    should be of appropriate size"

        err_code =  _cudamat_ext.set_diagonal(self.p_mat, diagonal.p_mat)

        if err_code:
            raise generate_exception(err_code)
    
        return self

    def reverse_column_entries(self, target = None):
        """
        reverse the elements of column vectors, swapping first for last
        etc
        """
        if target == None:
           target = self
        
        err_code = _cudamat_ext.reverse_column_entries(self.p_mat, 
                                                       target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return self

def log_1_plus_abs(mat, target = None):
    """
    Apply log(1+abs(x)) to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat_ext.apply_log_1_plus_abs(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def log_1_plus_abs_grad(act, out_grad, target = None):
    """
    backprop gradient through log(1+abs(x))
    """

    if not target:
        target = act

    err_code = _cudamat_ext.apply_log_1_plus_abs_grad(act.p_mat, 
                                                  out_grad.p_mat,
                                                  target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def columnwise_dot(mat1, mat2, target):
    """
       compute sum(mat1 * mat2, axis = 0)
    """

    err_code =  _cudamat_ext.column_dot(mat1.p_mat,
                                        mat2.p_mat,
                                        target.p_mat)

    if err_code:
        raise generate_exception(err_code)


def compute_logistic_log_prob(outputs, targets, log_probs):
    """
       compute targets * log(outputs) + (1-targets) * log (1-outputs)
    """
    err_code =  _cudamat_ext.logistic_log_prob(outputs.p_mat,
                                               targets.p_mat,
                                               log_probs.p_mat)

    if err_code:
        raise generate_exception(err_code)


def compute_logistic_grad(outputs, output_grad, target=None):
    """
       compute output_grad * outputs * (1-outputs)
    """
    if target != None:
        err_code =  _cudamat_ext.logistic_grad(outputs.p_mat,
                                                     output_grad.p_mat,
                                                     target.p_mat)
    else:
        err_code =  _cudamat_ext.logistic_grad(outputs.p_mat,
                                                     output_grad.p_mat,
                                                     output_grad.p_mat)

    if err_code:
        raise generate_exception(err_code)

    return target

def compute_softmax_grad(outputs, output_grad, target=None):
    """
       compute softmax grad:
        outputs * (output_grad - sum(output_grad * outputs,
                   axis=0).reshape(1,-1))
    """
    if target != None:
        err_code =  _cudamat_ext.softmax_grad(outputs.p_mat,
                                                     output_grad.p_mat,
                                                     target.p_mat)
    else:
        err_code =  _cudamat_ext.softmax_grad(outputs.p_mat,
                                                     output_grad.p_mat,
                                                     output_grad.p_mat)

    if err_code:
        raise generate_exception(err_code)

    return target

def compute_softmax_accuraccy(cm_prob, cm_target, cm_log_prob, 
                              cm_correct):
    """
       Computes accuraccy and log probs for targets, given 
       softmax predictions in cm_probs. Return log probs 
       of individual points and correct (=1) / incorrect (=0)
       status, in row vectors, cm_log_prob and cm_correct.
    """
    err_code =  _cudamat_ext.softmax_accuraccy(cm_prob.p_mat,
                                               cm_target.p_mat,
                                               cm_log_prob.p_mat,
                                               cm_correct.p_mat)

    if err_code:
        raise generate_exception(err_code)

def compute_mixture_of_softmax(cm_input, cm_expert_weights, 
                               cm_wts, cm_biases, cm_num_inputs_to_experts,
                               cm_activation_softmax, cm_prob_softmax,
                               cm_output):
	 ''' Compute mixture of softmax probabilities. Here
	 '''

	 err_code =  _cudamat_ext.mixture_of_softmax(\
                                          cm_input.p_mat,
                                          cm_expert_wts.p_mat,
                                          cm_wts.p_mat,
                                          cm_biases.p_mat,
                                          cm_num_inputs_to_experts.p_mat,
                                          cm_activation_softmax.p_mat,
                                          cm_prob_softmax.p_mat,
                                          cm_output.p_mat)

	 if err_code:
		  raise generate_exception(err_code)

	 return target


def empty(shape):
    mat = cudamat.cudamat()
    err_code = _cudamat.init_empty(ct.pointer(mat), ct.c_int(shape[0]), ct.c_int(shape[1]))
    if err_code:
        raise generate_exception(err_code)

    return CUDAMatrix(mat)

#def sum(mat, axis, target = None):
    #"""
    #Sum the matrix along the given dimension, where 0 represents the leading
    #dimension and 1 represents the non-leading dimension. If a target is
    #not prvided, a new vector is created for storing the result.
    #"""
    #
    #if axis == 0:
        #return cudamat.sum(mat, axis, target)
    #elif axis == 1:
        #if target is None:
            #target = empty((mat.shape[0], 1))
        #else:
            #assert(target.shape[0] == mat.shape[0] and
                   #target.shape[1] == 1)
        #err_code = _cudamat_ext.sum_columns(mat.p_mat, target.p_mat)
    #else:
        #raise Exception, "Incorrect value for axis"

    #if err_code:
        #raise generate_exception(err_code)

    #return target


def cublas_shutdown():
    """
    Shut down Cublas.
    """

    CUDAMatrix.ones = 0
    _cudamat_ext.free_rand_generator(CUDAMatrix.rnd_state_p)
    _cudamat.cublas_shutdown()

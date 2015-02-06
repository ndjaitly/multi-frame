import GPULock
GPULock.GetGPULock()
import cudamat_ext as cm
cm.cublas_init()
cm.CUDAMatrix.init_random(42)
import numpy as np
from pylab import log, sum

m = 299
n = 128

target = np.random.rand(m, n) > 0.5
prob = np.random.rand(m, n)

cm_target = cm.CUDAMatrix(cm.reformat(target))
cm_prob = cm.CUDAMatrix(cm.reformat(prob))

cm_log_prob = cm.empty((1,n))
cm.compute_logistic_log_prob(cm_prob, cm_target, cm_log_prob)

lg_prob = sum(target*log(1e-8+prob) + (1-target)*log(1-prob+1e-8), axis=0).reshape((1,-1))
error =  np.sum((cm_log_prob.asarray() - lg_prob)**2)
print "Error = ", error, " sum_lg_prob = ", str(sum(lg_prob))

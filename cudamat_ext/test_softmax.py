import cudamat_ext as cm
import GPULock
GPULock.GetGPULock()
cm.cublas_init()
cm.CUDAMatrix.init_random()

import numpy as np
import datetime, time

def test_softmax():
    m = 2000
    n = 128

    data = np.random.randn(m, n)
    prob = data - data.max(axis=0).reshape(1,-1)
    prob = np.exp(prob) / np.exp(prob).sum(axis=0).reshape(1,-1)
    
    cm_data = cm.CUDAMatrix(cm.reformat(data))
    cm_prob = cm.CUDAMatrix(cm.reformat(np.zeros(data.shape)))

    cm_data.compute_softmax(cm_prob)

    error =  np.sum((cm_prob.asarray() - prob)**2)
    print "Error = ", error
    assert error < 10**-2, "Error in CUDAMatrix.compute_softmax exceeded threshold"


def test_softmax_sample():
    dim, num_pts = 160, 128
    num_draws = 10000

    probs = rand(dim, num_pts)
    for i in range(min(dim, num_pts)):
        probs[i,i] = 2.0
 
    probs = probs / probs.sum(axis=0).reshape(1,-1)

    cm_prob = cm.CUDAMatrix(log(probs))
    cm_data = cm.empty(probs.shape)
    cm_rands = cm.empty(probs.shape)
    cm_counts = cm.empty(probs.shape).assign(0)

    s = datetime.datetime.now()
    for draw in range(num_draws):
        cm_rands.fill_with_rand()
        cm_prob.SampleSoftMax(cm_rands, cm_data)
        cm_counts.add(cm_data)
        cm_data.assign(0)
    e = datetime.datetime.now()
    diff= e-s
    cm_counts.divide(num_draws)
    est_probs = cm_counts.asarray().copy()

    print "Total time for %d draws = %d microseconds\n"%(num_draws, diff.microseconds)
    print "Average case error = %.5f \n"%(np.mean(abs(est_probs-probs)))
  
    from matplotlib.pyplot import subplot, imshow, draw
    subplot(311), imshow(probs, aspect='auto', interpolation='nearest')
    subplot(312), imshow(est_probs, aspect='auto', interpolation='nearest')
    subplot(313), plot(est_probs[:,0])
    subplot(313), plot(probs[:,0])
    draw(), time.sleep(0.2)
    raw_input('enter to finish')
    return est_probs, probs


if __name__ == "__main__":
    test_softmax()
    est_probs, probs = test_softmax_sample()

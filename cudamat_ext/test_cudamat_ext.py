import pdb
import numpy as np
import nose
import cudamat_ext as cm
from pylab import permutation

def setup():
    cm.cublas_init()

def teardown():
    cm.cublas_shutdown()

def test_columnwise_dot():
    m = 64
    n = 64
    a = np.array(np.random.randn(m, n), dtype=np.float32, order='F')
    b = np.array(np.random.randn(m, n), dtype=np.float32, order='F')

    res = np.sum(a*b, axis=0).reshape(1,-1)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    cm_res = cm.CUDAMatrix(cm.reformat(np.zeros(res.shape)))

    cm.columnwise_dot(m1, m2, cm_res)

    err = np.sum(np.abs(res - cm_res.asarray()))
    assert err < 10**-2, "Error in cudamat_ext.columnwise_dot exceeded threshold"

def test_softmax():
    m = 256
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


def test_column_set_get():
    m = 256
    n = 128

    data = np.random.randn(m, n)
    cm_data = cm.CUDAMatrix(cm.reformat(data))

    indices = permutation(n)
    cm_indices = cm.CUDAMatrix(cm.reformat(indices.reshape(1,-1)))

    start = 0
    end = 10
    cm_columns = cm_data.get_column_vectors(cm_indices, start, end)
    
    get_error =  np.sum((cm_columns.asarray() - 
                             data[:,indices[start:end]])**2)


    data_set = np.random.randn(m, end-start)
    cm_columns.free_device_memory()
    cm_columns = cm.CUDAMatrix(cm.reformat(data_set))
    cm_data.set_column_vectors(cm_indices, start, end, cm_columns)

    data[:,indices[start:end]] = data_set
    set_error =  np.sum((cm_data.asarray() - 
                             data)**2)

    print "Get Error = ", get_error
    print "Set Error = ", set_error
    assert get_error < 10**-2 or set_error < 10**-2,  \
             "Error in CUDAMatrix.get_column_vectors exceeded threshold"


if __name__ == '__main__':
    #nose.run()
    #test_softmax()
    #test_columnwise_dot()
    test_column_set_get()

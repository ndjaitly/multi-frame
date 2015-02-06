
import numpy, sys, util
import cudamat as cm
cm.cuda_set_device(6)
cm.cublas_init()
cm.CUDAMatrix.init_random(42)

cmMat = cm.empty((20, 128))
for batch in range(10000):
    cmMat.fill_with_randn()
    if numpy.isnan(cmMat.euclid_norm()):
        util.save('test.dat', 'a', {'a':cmMat.asarray()})
        print "nan error in batch: ", batch
        sys.stdout.flush()
        sys.exit(1)

print "Ran without a problem"

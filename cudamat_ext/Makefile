cudamat_ext:
	nvcc -O --ptxas-options=-v --compiler-options '-fPIC' -o libcudamat_ext.so --shared cudamat_ext.cu cudamat_ext_kernels.cu Convolutions.cu Convolutions_kernels.cu SoftMax.cu SoftMax_kernels.cu -I../cudamat/ -lcublas  -lcurand


all: cudamat_ext
clean:
	rm -f *.pyc *.so

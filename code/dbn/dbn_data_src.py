from pylab import *
import cudamat_ext as CM
import numpy
import logging
logger = logging.getLogger('GaussianBinaryRBM')
logger.setLevel(logging.INFO)
# create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


class data_src:
    def __init__(self, file_name, data_dim):
        self.data_file = file_name
        self.f = open(file_name, 'wb')
        self.num_pts = 0
        self._data_dim = data_dim

    def finish_adding(self):
        self.f.flush()
        self.f.close()
        self.f = None

    def add_data(self, data):
        self.num_pts += data.shape[1]
        self.f.write(data.reshape(-1, order='F').tostring())

    def get_data_dim(self):
        return self._data_dim

    def get_iterator(self, batch_size, return_labels=False):
        self.f = open(self.data_file, 'rb')
        cm_data = CM.empty((self._data_dim, batch_size))
        batch_num = 0
        data_dim = self._data_dim
        num_batches = self.num_pts/batch_size
        num_batches_per_load = 1000
        num_bytes_per_batch = 4*data_dim*batch_size
        num_bytes_per_load = num_batches_per_load*num_bytes_per_batch
        batch_num_since_last_load = 0
        num_batches_loaded = 0

        while batch_num < num_batches:
            if batch_num_since_last_load == num_batches_loaded:
                cur_data_str = self.f.read(num_bytes_per_load)
                num_batches_loaded = len(cur_data_str)/num_bytes_per_batch
                num_pts_read = num_batches_loaded * batch_size
                cur_data = zeros((data_dim, num_pts_read), 'float32')
                for b in arange(0,num_pts_read, batch_size):
                    str_s = b*4*data_dim
                    str_e = str_s + 4*data_dim*batch_size
                    data_arr =  numpy.fromstring(cur_data_str[str_s:str_e],
                                                  dtype='float32')
                    cur_data[:,b:(b+batch_size)] = data_arr.reshape(\
                                     (data_dim, batch_size), order='F')

                try:
                    cm_data_big.free_device_memory()
                    cm_indices.free_device_memory()
                    cm_data_big, cm_indices = None, None
                except NameError:
                    pass
                cm_data_big = CM.CUDAMatrix(cur_data)
                cm_indices = CM.CUDAMatrix(permutation(num_pts_read).reshape(1,-1))
                batch_num_since_last_load = 0
                cur_data_str = None

            start = batch_num_since_last_load*batch_size
            cm_data_big.select_columns(cm_indices.slice(start,
                                             start+batch_size), cm_data)
            batch_num_since_last_load += 1
            batch_num += 1
            yield cm_data


        cm_data.free_device_memory()
        cm_data_big.free_device_memory()
        cm_indices.free_device_memory()
        cm_data, cm_data_big, cm_indices = None, None, None

        self.f.close()
        self.f = None


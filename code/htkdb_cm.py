import cudamat_ext as cm
import util, sys, scipy, numpy, htkdb, pdb
from numpy import zeros, arange, uint32, tile, array, int32, diag
from numpy import  ones, log, concatenate, eye, floor, flatnonzero, unique, sort
from numpy.random import permutation
from scipy.io import loadmat

class htkdb_cm:
    def __init__(self, db_name, db_path, num_frames_per_pt,
                     use_deltas_accs = True, dropout_rate=0, 
                     skip_borders = 0):
        self._data_src = htkdb.htkdb(db_name, db_path)
        self._data_src.load(use_deltas_accs)
        self._num_files = self._data_src.get_num_files()
        self._num_frames_per_pt = num_frames_per_pt
        self.__frame_dim = self._data_src.data_dim
        self.__data_dim = num_frames_per_pt * self.__frame_dim
        self.dropout_rate = dropout_rate
        self._skip_borders = skip_borders
        self._label_map = None
        self._num_mapped_labels = -1

    def set_label_map(self, label_map):
        assert(label_map.size == self._data_src.get_label_dim())
        assert(len(label_map.shape) == 1)
        self._label_map = label_map.copy()
        self._num_mapped_labels = unique(sort(label_map)).size
 
    def get_label_dim(self):
        if self._label_map is None:
            return self._data_src.get_label_dim()
        return self._num_mapped_labels

    def get_num_frames_per_pt(self):
        return self._num_frames_per_pt

    def get_data_dim(self):
        return self.__data_dim

    def get_frame_dim(self):
        return self.__frame_dim

    def get_num_files(self):
        return self._num_files


    def get_data_for_file(self, file_num, return_labels=False):
        data, labels = self._data_src.get_spectrogram_and_labels(\
                                   file_num, self._speaker_cmn,\
                                   self._speaker_cmvn, self._normalize)
        if return_labels:
            if self._label_map is None:
                return data, labels
            else:
                return data, self._label_map[labels]
        return data
 
    def load_next_data(self):
        last_file = min(self.__start_file_num + self._num_files_per_load,
                          self._num_files)
        data_lst = [] 
        label_lst = [] 
        indices_lst = [] 
        num_frames = 0
        num_indices = 0
        for file_index in range(self.__start_file_num, last_file):
            file_num = self._file_indices[file_index]
            data, cur_labels = self._data_src.get_spectrogram_and_labels(\
                                   file_num, self._speaker_cmn,\
                                   self._speaker_cmvn, self._normalize)
            if self._label_map is not None:  cur_labels = self._label_map[cur_labels]
            if self._skip_borders != 0:
                data = data[:, self._skip_borders:(-self._skip_borders)]
                cur_labels = cur_labels[self._skip_borders:(-self._skip_borders)]

            indices = arange(max(0, data.shape[1]-self._num_frames_per_pt+1))
            data_lst.append(data)
            label_lst.append(cur_labels.copy())
            indices_lst.append(indices.copy())

            num_frames += data.shape[1]
            num_indices += indices.size

        self._num_frames = 0
        self._num_frames_for_training = num_indices

        self._data_matrix = zeros((self.__frame_dim, num_frames))
        self._label_matrix = zeros((1, num_frames))
        self._data_indices = zeros((1, num_indices))

        num_frames_so_far = 0
        num_indices_so_far  = 0

        for (cur_data, cur_labels, cur_indices) in zip(data_lst, label_lst, 
                                                       indices_lst):
            num_frames_cur = cur_data.shape[1]
            self._data_matrix[:, 
               num_frames_so_far:(num_frames_so_far+num_frames_cur)] = cur_data.copy()
            self._label_matrix[0, 
               num_frames_so_far:(num_frames_so_far+num_frames_cur)] = cur_labels.copy()

            num_indices_cur = cur_indices.size
            self._data_indices[0, 
               num_indices_so_far:(num_indices_so_far+num_indices_cur)] = cur_indices + num_frames_so_far

            num_frames_so_far += num_frames_cur
            num_indices_so_far += num_indices_cur

        assert(num_indices_so_far == num_indices)
        assert(num_frames_so_far == num_frames)


        try:
            self.__cm_data_matrix.free_device_memory()
            self.__cm_targets_matrix.free_device_memory()
            self.__cm_indices_matrix.free_device_memory()
        except AttributeError:
            pass

        self.__cm_data_matrix = cm.CUDAMatrix(self._data_matrix)
        self.__cm_targets_matrix = cm.CUDAMatrix(self._label_matrix)
        self.__start_file_num = last_file
        self.permute_indices_for_loaded_data()


    def permute_indices_for_loaded_data(self):
        ''' Repermutes indices for currently loaded data. Can be used if 
            we load all the data at one, and don't want to reload it.
        '''
        data_permutation = permutation(self._data_indices.size)
        self._data_indices = self._data_indices[0,\
                                  data_permutation].reshape((1,-1))
        self.__cm_indices_matrix = cm.CUDAMatrix(\
                                    cm.reformat(self._data_indices))
        self.__batch_index = 0
        self._is_setup = True

    def setup_data(self, num_files_per_load=-1, speaker_cmn=False, 
                                speaker_cmvn=False, normalize=True):
        if num_files_per_load == -1:
            self._num_files_per_load = self._num_files
        else:
            self._num_files_per_load = num_files_per_load
        self._speaker_cmn = speaker_cmn
        self._speaker_cmvn = speaker_cmvn
        self._normalize = normalize
        self.permute_file_indices_for_loading()

    def permute_file_indices_for_loading(self):
        self._file_indices = permutation(self._num_files)
        self.__start_file_num = 0
        self.label_offset = floor(self._num_frames_per_pt/2)
        self.load_next_data()
        self._is_setup = True

    def get_iterator(self, batch_size, return_labels=True):
        if not hasattr(self, '_is_setup'):
            raise Exception, "Call setup_data or permute_indices first"
        if  not self._is_setup:
            self.permute_file_indices_for_loading()
      
        self.__cm_data_for_batch = cm.empty((self.__data_dim, batch_size))
        self.__cm_targets_for_batch = cm.empty((self.get_label_dim(),
                                                batch_size))
        self.__cm_data_indices_for_batch = cm.empty((1, batch_size))
        self.__cm_target_indices_for_batch = cm.empty((1, batch_size))
        self.__cm_data_indices_with_frames = cm.empty((self._num_frames_per_pt, 
                                                      batch_size))
        self.__cm_range_frames = cm.CUDAMatrix(cm.reformat(arange(\
                                  self._num_frames_per_pt).reshape((-1,1))))
        self.__cm_target_vectors_matrix = cm.CUDAMatrix(\
                                                  eye(self.get_label_dim()))
 
        while True:
            if self.__batch_index + batch_size > self._num_frames_for_training:
                if self.__start_file_num >= self._num_files:
                    break
                self.load_next_data()

            self.__cm_indices_matrix.get_col_slice(self.__batch_index, 
                                      self.__batch_index + batch_size, 
                                      self.__cm_data_indices_for_batch)

            self.__cm_data_indices_with_frames.reshape((self._num_frames_per_pt,
                                                        batch_size))
            self.__cm_data_indices_with_frames.assign(0)
            self.__cm_data_indices_with_frames.add_col_vec(\
                                            self.__cm_range_frames)
            self.__cm_data_indices_with_frames.add_row_vec(\
                                          self.__cm_data_indices_for_batch)
            self.__cm_data_indices_with_frames.reshape((1, 
                              self._num_frames_per_pt * batch_size))

            self.__cm_data_matrix.select_columns(\
                                       self.__cm_data_indices_with_frames, 
                                                 self.__cm_data_for_batch)
            self.__cm_data_for_batch.reshape((self.__data_dim,
                                              batch_size))
            if self.dropout_rate != 0:
                self.__cm_data_for_batch.dropout(self.dropout_rate)
                self.__cm_data_for_batch.mult(1./(1-self.dropout_rate))

            self.__batch_index += batch_size

            if return_labels:
                self.__cm_data_indices_for_batch.add(self.label_offset)
                self.__cm_targets_matrix.select_columns(\
                                   self.__cm_data_indices_for_batch, 
                                   self.__cm_target_indices_for_batch)

                self.__cm_target_vectors_matrix.select_columns(\
                                       self.__cm_target_indices_for_batch,
                                       self.__cm_targets_for_batch)
                
                yield self.__cm_data_for_batch, self.__cm_targets_for_batch
            else:
                yield self.__cm_data_for_batch

        self._is_setup=False


import cudamat_ext as cm
import util, sys, scipy, numpy, htkdb, pdb
from numpy import zeros, arange, uint32, tile, array, int32, diag
from numpy import  ones, log, concatenate, eye, floor, flatnonzero, unique, sort
from numpy.random import permutation
from scipy.io import loadmat

class htkdb_cm:
    def __init__(self, db_name, db_path, num_frames_per_pt,
                     num_outputs_per_pt, use_deltas_accs = True, 
                     dropout_rate=0, skip_borders = 0):
        self._data_src = htkdb.htkdb(db_name, db_path)
        self._data_src.load(use_deltas_accs)
        self._num_files = self._data_src.get_num_files()
        self._num_frames_per_pt = num_frames_per_pt
        self._frame_dim = self._data_src.data_dim
        self._num_outputs_per_pt = num_outputs_per_pt
        self._data_dim = num_frames_per_pt * self._frame_dim
        self.dropout_rate = dropout_rate
        self._skip_borders = skip_borders
        self._borders_only = False

    def get_label_dim(self):
        return self._data_src.get_label_dim()

    def get_num_frames_per_pt(self):
        return self._num_frames_per_pt

    def get_num_outputs_frames_per_pt(self):
        return self._num_outputs_per_pt

    def get_data_dim(self):
        return self._data_dim

    def get_num_files(self):
        return self._num_files


    def get_data_for_file(self, file_num, return_labels=False):
        data, labels = self._data_src.get_spectrogram_and_labels(\
                                   file_num, self._speaker_cmn,\
                                   self._speaker_cmvn, self._normalize)
        if return_labels:
            return data, labels
        return data
 
    def load_next_data(self):
        last_file = min(self._start_file_num + self._num_files_per_load,
                          self._num_files)
        data_lst = [] 
        label_lst = [] 
        indices_lst = [] 
        num_frames = 0
        num_indices = 0
        for file_index in range(self._start_file_num, last_file):
            file_num = self._file_indices[file_index]
            data, cur_labels = self._data_src.get_spectrogram_and_labels(\
                                   file_num, self._speaker_cmn,\
                                   self._speaker_cmvn, self._normalize)
            if self._skip_borders != 0:
                data = data[:, self._skip_borders:(-self._skip_borders)]
                cur_labels = cur_labels[self._skip_borders:(-self._skip_borders)]

            if self._borders_only:
                I = flatnonzero(cur_labels[1:] != cur_labels[:-1])
                if I[0] != 0:
                    indices = concatenate(([0], I, I+1))
                else:
                    indices = concatenate((I, I+1))
                indices.sort()
                indices = unique(indices)
                indices -= int(self.label_offset)
                indices = indices[indices < (data.shape[1] - self._num_frames_per_pt + 1)]
                indices = indices[indices >= 0]
            else:
                indices = arange(max(0, data.shape[1]-self._num_frames_per_pt+1))

            data_lst.append(data)
            label_lst.append(cur_labels.copy())
            indices_lst.append(indices.copy())

            num_frames += data.shape[1]
            num_indices += indices.size

        self._num_frames = 0
        self._num_frames_for_training = num_indices

        self._data_matrix = zeros((self._frame_dim, num_frames))
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
            self._cm_data_matrix.free_device_memory()
            self._cm_targets_matrix.free_device_memory()
            self._cm_indices_matrix.free_device_memory()
        except AttributeError:
            pass

        self._cm_data_matrix = cm.CUDAMatrix(self._data_matrix)
        self._cm_targets_matrix = cm.CUDAMatrix(self._label_matrix)
        self._start_file_num = last_file
        self.permute_indices_for_loaded_data()


    def permute_indices_for_loaded_data(self):
        ''' Repermutes indices for currently loaded data. Can be used if 
            we load all the data at one, and don't want to reload it.
        '''
        data_permutation = permutation(self._data_indices.size)
        self._data_indices = self._data_indices[0,\
                                  data_permutation].reshape((1,-1))
        self._cm_indices_matrix = cm.CUDAMatrix(\
                                    cm.reformat(self._data_indices))
        self._batch_index = 0
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
        self._start_file_num = 0
        self.label_offset = floor((self._num_frames_per_pt-self._num_outputs_per_pt+1)/2)
        self.load_next_data()
        self._is_setup = True

    def get_iterator(self, batch_size, return_labels=True):
        if not hasattr(self, '_is_setup'):
            raise Exception, "Call setup_data or permute_indices first"
        if  not self._is_setup:
            self.permute_file_indices_for_loading()
      
        self._cm_data_for_batch = cm.empty((self._data_dim, batch_size))

        target_shape = ((self.get_label_dim(),
                         self._num_outputs_per_pt*batch_size))
        multi_target_shape = ((self.get_label_dim()*self._num_outputs_per_pt, 
                               batch_size))

        self._cm_targets_for_batch = cm.empty(target_shape)
        self._cm_data_indices_for_batch = cm.empty((1, batch_size))
        self._cm_data_indices_with_frames = cm.empty((self._num_frames_per_pt, 
                                                      batch_size))
        self._cm_target_indices_with_frames = cm.empty((self._num_outputs_per_pt, 
                                                         batch_size))
        self._cm_target_indices_for_batch = cm.empty((1, 
                                           self._num_outputs_per_pt*batch_size))

        self._cm_range_frames = cm.CUDAMatrix(cm.reformat(arange(\
                                  self._num_frames_per_pt).reshape((-1,1))))
        self._cm_range_target_frames = cm.CUDAMatrix(cm.reformat(arange(\
                                             self._num_outputs_per_pt).reshape((-1,1))))
        self._cm_target_vectors_matrix = cm.CUDAMatrix(\
                                                  eye(self.get_label_dim()))
 
        while True:
            if self._batch_index + batch_size > self._num_frames_for_training:
                if self._start_file_num >= self._num_files:
                    break
                self.load_next_data()

            self._cm_indices_matrix.get_col_slice(self._batch_index, 
                                      self._batch_index + batch_size, 
                                      self._cm_data_indices_for_batch)

            self._cm_data_indices_with_frames.reshape((self._num_frames_per_pt,
                                                        batch_size))
            self._cm_data_indices_with_frames.assign(0)
            self._cm_data_indices_with_frames.add_col_vec(\
                                            self._cm_range_frames)
            self._cm_data_indices_with_frames.add_row_vec(\
                                          self._cm_data_indices_for_batch)
            self._cm_data_indices_with_frames.reshape((1, 
                              self._num_frames_per_pt * batch_size))

            self._cm_target_indices_with_frames.reshape((self._num_outputs_per_pt,
                                                        batch_size))
            self._cm_target_indices_with_frames.assign(0)
            self._cm_target_indices_with_frames.add_col_vec(\
                                            self._cm_range_target_frames)
            self._cm_target_indices_with_frames.add_row_vec(\
                                          self._cm_data_indices_for_batch)
            self._cm_target_indices_with_frames.add(self.label_offset)
            self._cm_target_indices_with_frames.reshape((1, 
                              self._num_outputs_per_pt * batch_size))

            self._cm_data_matrix.select_columns(\
                                       self._cm_data_indices_with_frames, 
                                                 self._cm_data_for_batch)
            self._cm_data_for_batch.reshape((self._data_dim,
                                              batch_size))
            if self.dropout_rate != 0:
                self._cm_data_for_batch.dropout(self.dropout_rate)
                self._cm_data_for_batch.mult(1./(1-self.dropout_rate))

            self._batch_index += batch_size

            if return_labels:
                self._cm_targets_matrix.select_columns(\
                                   self._cm_target_indices_with_frames, 
                                   self._cm_target_indices_for_batch)

                self._cm_targets_for_batch.reshape(target_shape)
                self._cm_target_vectors_matrix.select_columns(\
                                       self._cm_target_indices_for_batch,
                                       self._cm_targets_for_batch)
                self._cm_targets_for_batch.reshape(multi_target_shape)
                
                yield self._cm_data_for_batch, self._cm_targets_for_batch
            else:
                yield self._cm_data_for_batch

        self._is_setup=False


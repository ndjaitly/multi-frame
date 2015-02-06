import os
import util, HTK, pdb
from pylab import zeros, sqrt, sys, sort, logical_and, sum, linspace,\
                  ones, eye, array, concatenate, isnan, isinf, find
class htkdb:
    def __init__(self, db_name, db_path, smooth_alignments=False):
        ''' Create the barebones htkdb object. After this, either
            call load() if database has already been created or
            call create_db() to create it.
        '''
        self.db_path = db_path
        self.DBName = db_name
        self.db_path =  os.path.join(db_path, db_name)
        self.SummaryFilePath = os.path.join(self.db_path, db_name+".dat")
        self.AliFile = os.path.join(db_path, db_name, "ali")
        self._smooth_alignments = smooth_alignments

       
    @staticmethod
    def create_speaker_maps(utt2spk_file):
        f_utt2spk = open(utt2spk_file)
        utt2spk_map = {}
        spk2utt_map = {}
        for line in f_utt2spk:
            pieces =  line.replace('\n', '').split(" ")
            utt2spk_map[pieces[0]] = pieces[1]
            if not spk2utt_map.has_key(pieces[1]):
                spk2utt_map[pieces[1]] = pieces[0]
            else:
                spk2utt_map[pieces[1]] += " " + pieces[0]
        f_utt2spk.close()
        return utt2spk_map, spk2utt_map

    def create_db(self, utt2spk_file=None):
        self.Utt2Speaker, self.Speaker2Utt = None, None
        if utt2spk_file is not None:
            self.Utt2Speaker, self.Speaker2Utt = \
                        htkdb.create_speaker_maps(utt2spk_file)

        sys.stderr.write("Creating DB in file: " + self.SummaryFilePath + "\n")
        if utt2spk_file:
            sys.stderr.write("Using utterance to speaker file: " +\
                                     utt2spk_file + "\n")
        else:
            sys.stderr.write("No utterance to speaker file provided\n")

        sys.stderr.flush()
        self.CreateFromAliFile()
        self.SaveDB()

    def load(self, use_deltas_accs):
        self.use_deltas_accs = use_deltas_accs
        if os.path.isfile(self.SummaryFilePath) == True:
            print "Loading DB indices from file: ", self.SummaryFilePath

        params_dict = {}
        util.load(self.SummaryFilePath, params_dict, verbose=False)

        try:
            self.label_dim = int(params_dict['label_dim'])
        except AttributeError:
            print "No label_dim in file"

        try:
            self.UtteranceIds = params_dict['UtteranceIds']
        except KeyError:
            print "No UtteranceIds in index file."

        self.RawFileList = params_dict['RawFileList']

        self.data_dim = params_dict['data_dim']

        try:
            self.NumFrames = params_dict['NumFrames']
        except KeyError:
            self.NumFrames = 0
        self.DataMeanVect = params_dict['DataMeanVect'].reshape(-1,1)
        self.DataStdVect = params_dict['DataStdVect'].reshape(-1,1)

        self.Utt2Speaker = params_dict['Utt2Speaker']
        self.Speaker2Utt = params_dict['Speaker2Utt']
        self.SpeakerMeans = params_dict['SpeakerMeans']
        self.SpeakerStds = params_dict['SpeakerStds']

        self._lst_ignored_files = params_dict['lst_ignored_files']

        if not self.use_deltas_accs:
            self.data_dim /= 3
            self.DataMeanVect = self.DataMeanVect[:self.data_dim,]
            self.DataStdVect = self.DataStdVect[:self.data_dim,]

            if self.SpeakerMeans is not None:
                for speaker in self.SpeakerMeans.keys():
                    self.SpeakerMeans[speaker] = \
                       self.SpeakerMeans[speaker][:self.data_dim,]
                    self.SpeakerStds[speaker] = \
                       self.SpeakerStds[speaker][:self.data_dim,]

        self.LoadAligments(self.AliFile)

    def CreateCache(self, skip=0, speaker_cmn=False, speaker_cmvn=False,
                         normalize=False):
        numFiles = self.get_num_files()

        print "Caching database. Loading file #: ",
        self.DataMatrixCache = list()
        self.LabelsCache = list()

        totalPoints = 0 

        printStr = ''
        for fileNum in range(0,numFiles):
            if fileNum % 100 == 0 or fileNum == numFiles-1:
                printStrNew = '\b' * (len(printStr)+1)
                printStr = str(fileNum)
                printString = printStrNew + printStr
                print printString,
                sys.stdout.flush()
            if fileNum in self._lst_ignored_files:
                continue
            spectrograms, labels = self.get_spectrogram_and_labels(fileNum, 
                                      speaker_cmn, speaker_cmvn, normalize)

            if self.Utt2Speaker != None:
                if speaker_cmvn:
                    speaker = self.Utt2Speaker[self.UtteranceIds[fileNum]]
                    spectrograms = (spectrograms - 
                               self.SpeakerMeans[speaker])/self.SpeakerStds[speaker]
                elif speaker_cmn:
                    speaker = self.Utt2Speaker[self.UtteranceIds[fileNum]]
                    spectrograms = (spectrograms - self.SpeakerMeans[speaker])

            self.LabelsCache.append(labels.copy())
 
            if spectrograms.shape[0] != self.data_dim:
                print "spectrogram.shape = ", spectrograms.shape
                raise ValueError, "Dimension of spectrogram does not match"

            self.DataMatrixCache.append(spectrograms)
            totalPoints = totalPoints + spectrograms.shape[1]
        self.TotalPoints = totalPoints
        print " Done. " 
        self.blnDataCached = True

    def RescaleData(self, scaleVector):
        for i in range(len(self.DataMatrixCache)):
            self.DataMatrixCache[i] = self.DataMatrixCache[i] / scaleVector

    def get_num_files(self): 
        return len(self.RawFileList)

    def get_label_dim(self): 
        return self.label_dim

    def SaveDB(self):
        util.save(self.SummaryFilePath, 'DataMeanVect DataStdVect \
                                RawFileList data_dim NumFrames \
                                Utt2Speaker Speaker2Utt SpeakerMeans SpeakerStds \
                                label_dim UtteranceIds lst_ignored_files',
                               {'DataMeanVect': self.DataMeanVect, 
                                'DataStdVect': self.DataStdVect, 
                                'lst_ignored_files': self._lst_ignored_files, 
                                'RawFileList': self.RawFileList, 
                                'data_dim': self.data_dim, 
                                'label_dim': self.label_dim, 
                                'NumFrames': self.NumFrames, 
                                'Utt2Speaker': self.Utt2Speaker, 
                                'Speaker2Utt': self.Speaker2Utt, 
                                'SpeakerMeans': self.SpeakerMeans, 
                                'SpeakerStds': self.SpeakerStds, 
                                'UtteranceIds': self.UtteranceIds})


    def get_labels(self, fileNum):
        ''' Gets labels from the label file
        '''
        return self._lst_labels[fileNum]

    def get_spectrogram(self, fileNum, speaker_cmn=False,
                          speaker_cmvn=False, bln_normalize=False):
        ''' Reads projections from file
        '''
        fName = os.path.join(self.db_path, self.RawFileList[fileNum])
        if self.use_deltas_accs: 
            data = HTK.ReadHTKWithDeltas(fName)
        else: 
            data = HTK.ReadHTK(fName)

        if speaker_cmvn:
            speaker = self.Utt2Speaker[self.UtteranceIds[fileNum]]
            speakerMean = self.SpeakerMeans[speaker]
            speakerStd = self.SpeakerStds[speaker]
            data = (data - speakerMean)/speakerStd
        elif speaker_cmn:
            speaker = self.Utt2Speaker[self.UtteranceIds[fileNum]]
            speakerMean = self.SpeakerMeans[speaker]
            data = data - speakerMean
        elif bln_normalize:
            data = (data - self.DataMeanVect)/self.DataStdVect
        return data

    def get_spectrogram_and_labels(self, fileNum, speaker_cmn=False,
                                   speaker_cmvn=False, bln_normalize=False):
        ''' Reads projections from file
        '''
        data = self.get_spectrogram(fileNum, speaker_cmn, speaker_cmvn, 
                                       bln_normalize)
        labels = self.get_labels(fileNum)
        assert(abs(labels.size- data.shape[1]) <= 3)
        if labels.size > data.shape[1]:
            labels = labels[:data.shape[1]]
        else:
            data = data[:, :labels.size]
        return data, labels

    def LoadAligments(self, ali_file):
        f = open(ali_file, 'r')
        lines = f.readlines()
        f.close()

        self._lst_labels = []
        self.UtteranceIds = []
        self.RawFileList = []


        self.Utt2Index = {}
        label_dim = -1
        for line in lines:
            parts = line.rstrip('\n').split()
            utterance_id = parts[0]
            self._lst_labels.append(array([int(x) for x in parts[1:]]))
            label_dim = max(label_dim, self._lst_labels[-1].max()+1)
            self.UtteranceIds.append(utterance_id)
            self.RawFileList.append(os.path.join(self.db_path,
                                          utterance_id + ".htk"))
            self.Utt2Index[utterance_id] = len(self.RawFileList)-1

        try:
            self.label_dim = max(label_dim, self.label_dim)
        except AttributeError:
            self.label_dim = label_dim
            print "No label_dim in file"
        if self._smooth_alignments: self.SmoothAlignments()

    def SmoothAlignments(self):
        ali_file = self.AliFile
        f = open(ali_file, 'r')
        lines = f.readlines()
        f.close()

        self._lst_labels = []
        self.UtteranceIds = []
        self.RawFileList = []


        self.Utt2Index = {}
        label_dim = -1
        for line in lines:
            parts = line.rstrip('\n').split()
            utterance_id = parts[0]
            cur_labels = array([int(x) for x in parts[1:]])
            I = find(cur_labels % 3 ==0)
            I2 = find(I[1:] != (I[:-1]+1))
            starts = concatenate(([0], I[I2+1]))
            ends = concatenate((I[I2+1], [cur_labels.size]))
            smoothed_labels = zeros(cur_labels.size, 'int')
            for (s,e) in zip(starts, ends):
               a = array(linspace(s,e,4), 'int')
               smoothed_labels[a[0]:a[1]] = cur_labels[s]
               smoothed_labels[a[1]:a[2]] = cur_labels[s]+1
               smoothed_labels[a[2]:a[3]] = cur_labels[s]+2

            #self._lst_labels.append(array([int(x) for x in parts[1:]]))
            self._lst_labels.append(smoothed_labels)
            label_dim = max(label_dim, self._lst_labels[-1].max()+1)
            self.UtteranceIds.append(utterance_id)
            self.RawFileList.append(os.path.join(self.db_path,
                                          utterance_id + ".htk"))
            self.Utt2Index[utterance_id] = len(self.RawFileList)-1

        try:
            self.label_dim = max(label_dim, self.label_dim)
        except AttributeError:
            self.label_dim = label_dim
            print "No label_dim in file"



    def __CreateMeansAndStdevs(self):
        print "Creating means and stdevs"
        self.DataSumSq = zeros((self.data_dim,1))
        self.DataSum = zeros((self.data_dim,1))

        if self.Utt2Speaker != None:
            self.SpeakerMeans = {}
            self.SpeakerStds = {}
            self.SpeakerNumFrames = {}
            if not hasattr(self, "Speaker2Utt"):
                raise Exception, "HTKDb needs to have attribute Speaker2Utt"

            for speaker in self.Speaker2Utt.keys():
                self.SpeakerMeans[speaker] = zeros((self.data_dim,1))
                self.SpeakerStds[speaker] = zeros((self.data_dim,1))
                self.SpeakerNumFrames[speaker] = zeros((self.data_dim,1))
        else:
            print "utt2speaker information does not exist"

    def CreateFromAliFile(self):
        self.LoadAligments(self.AliFile)
        printStr = ''
        self._lst_ignored_files = []
        self.NumFrames = 0
        created_means = False
        for index, (file_name, utterance_id) in \
                enumerate(zip(self.RawFileList, self.UtteranceIds)):
            printStrNew = '\b' * (len(printStr)+1)
            printStr = "Loading data for utterance #: " + str(index+1)
            printString = printStrNew + printStr
            print printString,
            sys.stdout.flush()

            data = HTK.ReadHTKWithDeltas(file_name)
            if sum(isnan(data)) != 0 or sum(isinf(data)) != 0:
                self._lst_ignored_files.append(index)
                continue

            if not created_means:
                created_means = True
                self.data_dim = data.shape[0]
                self.__CreateMeansAndStdevs()

            self.DataSumSq += (data**2).sum(axis=1).reshape(-1,1)
            self.DataSum += data.sum(axis=1).reshape(-1,1)
            self.NumFrames += data.shape[1]

            if self.Utt2Speaker != None:
                speaker = self.Utt2Speaker[utterance_id]
                self.SpeakerMeans[speaker] += data.sum(axis=1).reshape(-1,1)
                self.SpeakerStds[speaker] += (data**2).sum(axis=1).reshape(-1,1)
                self.SpeakerNumFrames[speaker] += data.shape[1]

        sys.stdout.write("\n")
        for file_num in self._lst_ignored_files:
            sys.stdout.write("File # " + str(file_num) + " was ignored \
                                   because of errors\n")

        if self.Utt2Speaker != None:
            for speaker in self.Speaker2Utt.keys():
                self.SpeakerMeans[speaker] /= (1.0 *self.SpeakerNumFrames[speaker])
                self.SpeakerStds[speaker] -= self.SpeakerNumFrames[speaker] * \
                                        (self.SpeakerMeans[speaker]**2)
                self.SpeakerStds[speaker] /= (1.0 *self.SpeakerNumFrames[speaker]-1)
                self.SpeakerStds[speaker][self.SpeakerStds[speaker] < 1e-8] = 1e-8
                self.SpeakerStds[speaker] = sqrt(self.SpeakerStds[speaker])

        self.DataMeanVect = self.DataSum/self.NumFrames
        variances = (self.DataSumSq - self.NumFrames*(self.DataMeanVect**2))/(self.NumFrames-1)
        variances[variances < 1e-8] = 1e-8
        self.DataStdVect = sqrt(variances)


    def ComputePriors(self):
        numFiles = self.get_num_files()
        priors = zeros(self.label_dim)
        printStr = ''
        for fileNum, labels in enumerate(self._lst_labels):
            printStrNew = '\b' * (len(printStr)+1)
            printStr = "Loading states for sentence #: " + str(fileNum)
            printString = printStrNew + printStr
            print printString,
            sys.stdout.flush()

            for label in labels:
                priors[label] = priors[label] + 1

        return priors


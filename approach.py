
from DataProcessing import Learner, Filter
from processing_utils import loadBiosig, nanCleaner
import math
from DataManipulation import loadDataAsMatrix, readEvents, extractEpochs

import pickle
import numpy as np

class Approach:
    def __init__(self):
        pass

    def defineApproach(self, sample_rate, f_low, f_high, f_order, csp_nei, class_ids, epoch_start, epoch_end):

        self.class_ids = class_ids

        # FEATURE EXTRACTION:
        self.smin = int(math.floor(epoch_start * sample_rate))
        self.smax = int(math.floor(epoch_end * sample_rate))

        self.filter = Filter(f_low, f_high, sample_rate, f_order, filt_type = 'iir', band_type = 'band')
        self.learner = Learner()

        self.learner.DesignLDA()
        self.learner.DesignCSP(csp_nei)
        self.learner.AssembleLearner()

        self.balance_epochs = False

    def trainModel(self):

        data, ev = self.loadData(self.data_cal_path, self.events_cal_path)
        epochs, labels = self.loadEpochs(data, ev)

        epochs_f = self.preProcess(epochs)
        self.learner.Learn(epochs_f, labels)
        self.learner.EvaluateSet(epochs_f, labels)
        score = self.learner.GetResults()

        return score

    def validateModel(self):

        data, ev = self.loadData(self.data_val_path, self.events_val_path)
        epochs, labels = self.loadEpochs(data, ev)

        epochs_f = self.preProcess(epochs)
        self.learner.EvaluateSet(epochs_f, labels)
        score = self.learner.GetResults()

        return score

    def cross_validate_model(self, n_iter, test_perc):

        data, ev = self.loadData(self.data_cal_path, self.events_cal_path)
        epochs, labels = self.loadEpochs(data, ev)

        epochs_f = self.preProcess(epochs)
        score = self.learner.cross_evaluate_set(epochs_f, labels, \
                                    n_iter, test_perc)

        return score

    def applyModelOnDataSet(self, epochs, labels):

        self.learner.EvaluateSet(epochs, labels)
        score = self.learner.GetResults()
        return score

    def applyModelOnEpoch(self, epoch, out_param = 'label'):
        #TODO

        epoch_f = self.preProcess(epoch)

        guess = self.learner.EvaluateEpoch(epoch_f, out_param = out_param)

        return guess

    def setPathToCal(self, dpath, evpath):

        self.data_cal_path = dpath
        self.events_cal_path = evpath

    def setPathToVal(self, dpath, evpath):

        self.data_val_path = dpath
        self.events_val_path = evpath

    def loadData(self, dpath, evpath):
        if self.channels == [-1]:
            data = loadDataAsMatrix(dpath).T
        else:
            data = loadDataAsMatrix(dpath).T[self.channels]

        data = nanCleaner(data)
        events = readEvents(evpath)

        return data, events

    def loadEpochs(self, data, events):

        epochs, labels = extractEpochs(data, events, 
                                                self.smin, 
                                                self.smax, 
                                                self.class_ids)

        idx_1 = np.where(labels == self.class_ids[0])[0]
        idx_2 = np.where(labels == self.class_ids[1])[0]

        if self.balance_epochs:
            nepochs = min([len(idx_1), len(idx_2)])
            idx_1 = idx_1[:nepochs]
            idx_2 = idx_2[:nepochs]
            idx = np.concatenate([idx_1,idx_2])

            return epochs[idx],labels[idx]
        else:
            return epochs, labels


    def preProcess(self, data_in):

        data_f = self.filter.ApplyFilter(data_in)
        
        return data_f

    def setValidChannels(self, channels):
        self.channels = channels

    def set_balance_epochs(self, balance_epochs):
        ''' Set balance epochs: the number of epochs loaded from one class will
        be the same as the number of epochs loaded for the opposite class. The 
        number of epochs is defined as the minimum from both classes. This avoid
        model bias when training.
        '''
        self.balance_epochs = balance_epochs
        
    def saveToPkl(self, path):
        path += '/approach_info.pkl'
        
        with open(path, 'w') as file_name:
            pickle.dump(self.__dict__, file_name)


    def loadFromPkl(self, path):
        path += '/approach_info.pkl'
        
        with open(path, 'r') as file_name:
            load_obj = pickle.load(file_name)

        self.__dict__.update(load_obj) 


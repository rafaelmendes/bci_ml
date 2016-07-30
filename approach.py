
from DataProcessing import Learner, Filter
from processing_utils import loadBiosig, nanCleaner
import math
from DataManipulation import loadDataAsMatrix, readEvents, extractEpochs

import pickle

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

        data = loadDataAsMatrix(dpath).T[self.channels]
        data = nanCleaner(data)
        events = readEvents(evpath)

        return data, events

    def loadEpochs(self, data, events):

        epochs_cal, labels = extractEpochs(data, events, 
                                                self.smin, 
                                                self.smax, 
                                                self.class_ids)
        return epochs_cal, labels


    def preProcess(self, data_in):

        data_f = self.filter.ApplyFilter(data_in)
        
        return data_f

    def setValidChannels(self, channels):
        self.channels = channels
        
    def saveToPkl(self, path):
        path += '/approach_info.pkl'
        
        with open(path, 'w') as file_name:
            pickle.dump(self.__dict__, file_name)


    def loadFromPkl(self, path):
        path += '/approach_info.pkl'
        
        with open(path, 'r') as file_name:
            load_obj = pickle.load(file_name)

        self.__dict__.update(load_obj) 


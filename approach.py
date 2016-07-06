
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

        epochs_cal_f = self.preProcess(self.epochs_cal)
        self.learner.Learn(epochs_cal_f, self.labels_cal)
        self.learner.EvaluateSet(epochs_cal_f, self.labels_cal)
        auto_score = self.learner.GetResults()

        return auto_score

    def validateModel(self):

        epochs_val_f = self.preProcess(self.epochs_val)
        self.learner.EvaluateSet(epochs_val_f, self.labels_val)
        val_score = self.learner.GetResults()

        return val_score


    def applyModelOnDataSet(self, epochs, labels):

        self.learner.EvaluateSet(epochs, labels)
        score = self.learner.GetResults()
        return score

    def applyModelOnEpoch(self, epoch, out_param = 'label'):
        #TODO

        epoch_f = self.preProcess(epoch)

        guess = self.learner.EvaluateEpoch(epoch_f, out_param = out_param)

        return guess

    def loadCalData(self, data_path, ev_path):

        data = loadDataAsMatrix(data_path).T
        events = readEvents(ev_path)

        self.epochs_cal, self.labels_cal = extractEpochs(data, events, 
                                                        self.smin, 
                                                        self.smax, 
                                                        self.class_ids)
        self.data_cal = data


    def loadValData(self, data_path, ev_path):

        data = loadDataAsMatrix(data_path).T
        events = readEvents(ev_path)

        self.epochs_val, self.labels_val = extractEpochs(data, events, 
                                                        self.smin, 
                                                        self.smax, 
                                                        self.class_ids)
        self.data_val = data

    def preProcess(self, data_in):

        data_f = self.filter.ApplyFilter(data_in)

        return data_f
        
    def saveToPkl(self, path):
        path += '/approach_info.pkl'
        
        with open(path, 'w') as file_name:
            pickle.dump(self.__dict__, file_name)


    def loadFromPkl(self, path):
        path += '/approach_info.pkl'
        
        with open(path, 'r') as file_name:
            load_obj = pickle.load(file_name)

        self.__dict__.update(load_obj) 


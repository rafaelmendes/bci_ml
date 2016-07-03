# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:33:00 2016

@author: rafael
"""

import os, sys, inspect

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split \
    (inspect.getfile( inspect.currentframe() ))[0],'algorithms')))

if cmd_subfolder not in sys.path:
            sys.path.insert(0, cmd_subfolder)

import numpy as np # numpy - used for array and matrices operations
import math as math # used for basic mathematical operations

import scipy.signal as sp
import scipy.linalg as lg

from scipy.fftpack import fft

from pylab import plot, show, pi

# from mne import Epochs, pick_types, find_events

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# from mne.decoding import CSP # Import Common Spatial Patterns
from sklearn.pipeline import Pipeline

from CommonSpatialPatterns import CSP

class Learner:
    def __init__(self, model = None):
        # Loads a previous model if existent
        self.clf = model

    def DesignLDA(self):
        self.svc = LinearDiscriminantAnalysis()

    def DesignCSP(self, n_comp):
        self.csp = CSP(n_components=n_comp, reg=None, log=True, cov_est='epoch')

    def AssembleLearner(self):
        self.clf = Pipeline([('CSP', self.csp), ('SVC', self.svc)])

    def Learn(self, train_epochs, train_labels):

        self.clf.fit(train_epochs, train_labels)

    def EvaluateSet(self, eval_epochs, eval_labels):

        self.score = self.clf.score(eval_epochs, eval_labels)

    def EvaluateEpoch(self, epoch, out_param = 'prob'):

        if out_param == 'prob':

            guess = self.clf.predict_proba(epoch)

        elif out_param == 'label':

            guess = self.clf.predict(epoch)

        return guess

    def PrintResults(self):
        # class_balance = np.mean(labels == labels[0])
        # class_balance = max(class_balance, 1. - class_balance)
        class_balance = 0.5
        print("Classification accuracy: %f / Chance level: %f" % (self.score,
                                                                  class_balance))
    def GetResults(self):
        return self.score

    def GetModel(self):
        return self.clf


class Filter:
    def __init__(self, fl, fh, srate, forder, filt_type = 'iir', band_type = 'band'):
        
        nyq = 0.5 * srate
        low = fl / nyq
        high = fh / nyq

        if filt_type == 'iir':
            # self.b, self.a = sp.butter(self.filter_order, [low, high], btype='band')
            self.b, self.a = sp.iirfilter(forder, [low, high], btype=band_type)

        elif filt_type == 'fir':
            self.b = sp.firwin(forder, [low, high], window = 'hamming',pass_zero=False)
            self.a = [1]

    def ApplyFilter(self, data_in):
    
        data_out = sp.filtfilt(self.b, self.a, data_in)

        return data_out

    def ComputeEnergy(self, data_in):

        data_squared = data_in ** 2
        # energy in each channel [e(ch1) e(ch2) ...]
        energy = np.mean(data_squared, axis = 0)

        return energy

    def GenerateWindow(self, win_len, n_seg, w_type = 'black'):
        ov = 0.5 # windows overlap

        seg_len = int(win_len / math.floor((n_seg * ov) + 1))

        print seg_len

        if w_type == 'han':
            win_seg = np.hanning(seg_len)

        if w_type == 'ham':
            win_seg = np.hamming(seg_len)

        if w_type == 'black':
            win_seg = np.blackman(seg_len)

        self.window = np.zeros(win_len)

        idx = np.array(range(seg_len))
        for i in range(n_seg):
            new_idx = idx + seg_len*ov*i
            new_idx = new_idx.astype(int)
            self.window[new_idx] = self.window[new_idx] +  win_seg

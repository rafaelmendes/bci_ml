# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:33:00 2016

@author: rafael
"""

import numpy as np # numpy - used for array and matrices operations
import math as math # used for basic mathematical operations

import scipy.signal as sp
import scipy.linalg as lg

from scipy.fftpack import fft

from pylab import plot, show, pi
import mne

from mne import Epochs, pick_types, find_events

from sklearn.lda import LDA
from mne.decoding import CSP # Import Common Spatial Patterns
from sklearn.pipeline import Pipeline
    
class DataProcessing:
    def __init__(self):
        pass

    def DesignFilter(self, fl, fh, srate, forder, filt_type = 'iir'):
        
        nyq = 0.5 * srate
        low = fl / nyq
        high = fh / nyq

        if filt_type == 'iir':
            # self.b, self.a = sp.butter(self.filter_order, [low, high], btype='band')
            self.b, self.a = sp.iirfilter(forder, [low, high], btype='band')

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

def nanCleaner(data_in):
    """Removes NaN from data by interpolation
    Parameters
    ----------
    data_in : input data - np matrix channels x samples

    Returns
    -------
    data_out : clean dataset with no NaN samples

    Examples
    --------
    >>> data_path = "/PATH/TO/DATASET/dataset.gdf"
    >>> EEGdata_withNaN = loadBiosig(data_path)
    >>> EEGdata_clean = nanCleaner(EEGdata_withNaN)
    """
    for i in range(data_in.shape[0]):
        
        bad_idx = np.isnan(data_in[i, ...])
        data_in[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], data_in[i, ~bad_idx])
    
    return data_in

def MNEFilter(data_in, f_low, f_high, f_order):
    # Apply band-pass filter
    data_out = data_in.filter(f_low, f_high, picks = None, filter_length=f_order, method='iir')
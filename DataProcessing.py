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
# from mne.decoding import CSP # Import Common Spatial Patterns
from sklearn.pipeline import Pipeline
    
class DataFiltering:
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


class DataLearner:
    def __init__(self):
        pass

    def DesignLDA(self):
        self.svc = LDA()

    def DesignCSP(self, n_comp):
        self.csp = CSP(n_components=n_comp, reg=None, log=True, cov_est='epoch')

    def AssembleLearner(self):
        self.clf = Pipeline([('CSP', self.csp), ('SVC', self.svc)])

    def Learn(self, train_epochs, train_labels):

        self.clf.fit(train_epochs, train_labels)

    def Evaluate(self, eval_epochs, eval_labels):

        self.score = self.clf.score(eval_epochs, eval_labels)

    def PrintResults(self):
        # class_balance = np.mean(labels == labels[0])
        # class_balance = max(class_balance, 1. - class_balance)
        class_balance = 0.5
        print("Classification accuracy: %f / Chance level: %f" % (self.score,
                                                                  class_balance))
    def GetResults(self):
        return self.score

class CSP:
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP).
    This object can be used as a supervised decomposition to estimate
    spatial filters for feature extraction in a 2 class decoding problem.
    CSP in the context of EEG was first described in [1]; a comprehensive
    tutorial on CSP can be found in [2].
    Parameters
    ----------
    n_components : int (default 4)
        The number of components to decompose M/EEG signals.
        This number should be set by cross-validation.
    reg : float | str | None (default None)
        if not None, allow regularization for covariance estimation
        if float, shrinkage covariance is used (0 <= shrinkage <= 1).
        if str, optimal shrinkage using Ledoit-Wolf Shrinkage ('ledoit_wolf')
        or Oracle Approximating Shrinkage ('oas').
    log : bool (default True)
        If true, apply log to standardize the features.
        If false, features are just z-scored.
    cov_est : str (default 'concat')
        If 'concat', covariance matrices are estimated on concatenated epochs
        for each class.
        If 'epoch', covariance matrices are estimated on each epoch separately
        and then averaged over each class.
    Attributes
    ----------
    filters_ : ndarray, shape (n_channels, n_channels)
        If fit, the CSP components used to decompose the data, else None.
    patterns_ : ndarray, shape (n_channels, n_channels)
        If fit, the CSP patterns used to restore M/EEG signals, else None.
    mean_ : ndarray, shape (n_channels,)
        If fit, the mean squared power for each component.
    std_ : ndarray, shape (n_channels,)
        If fit, the std squared power for each component.
    """

    def __init__(self, n_components=4, reg=None, log=True, cov_est="concat"):
        """Init of CSP."""
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.cov_est = cov_est
        self.filters_ = None
        self.patterns_ = None
        self.mean_ = None
        self.std_ = None

    def get_params(self, deep=True):
        """Return all parameters (mimics sklearn API).
        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        """
        params = {"n_components": self.n_components,
                  "reg": self.reg,
                  "log": self.log}
        return params

    def fit(self, epochs_data, y):
        """Estimate the CSP decomposition on epochs.
        Parameters
        ----------
        epochs_data : ndarray, shape (n_epochs, n_channels, n_times)
            The data to estimate the CSP on.
        y : array, shape (n_epochs,)
            The class for each epoch.
        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """

        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))
        epochs_data = np.atleast_3d(epochs_data)
        e, c, t = epochs_data.shape
        # check number of epochs
        if e != len(y):
            raise ValueError("n_epochs must be the same for epochs_data and y")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("More than two different classes in the data.")
        if not (self.cov_est == "concat" or self.cov_est == "epoch"):
            raise ValueError("unknown covariance estimation method")

        if self.cov_est == "concat":  # concatenate epochs
            class_1 = np.transpose(epochs_data[y == classes[0]],
                                   [1, 0, 2]).reshape(c, -1)
            class_2 = np.transpose(epochs_data[y == classes[1]],
                                   [1, 0, 2]).reshape(c, -1)
            cov_1 = _regularized_covariance(class_1, reg=self.reg)
            cov_2 = _regularized_covariance(class_2, reg=self.reg)
        elif self.cov_est == "epoch":
            class_1 = epochs_data[y == classes[0]]
            class_2 = epochs_data[y == classes[1]]
            cov_1 = np.zeros((c, c))
            for t in class_1:
                cov_1 += _regularized_covariance(t, reg=self.reg)
            cov_1 /= class_1.shape[0]
            cov_2 = np.zeros((c, c))
            for t in class_2:
                cov_2 += _regularized_covariance(t, reg=self.reg)
            cov_2 /= class_2.shape[0]

        # normalize by trace
        cov_1 /= np.trace(cov_1)
        cov_2 /= np.trace(cov_2)

        e, w = lg.eigh(cov_1, cov_1 + cov_2)
        n_vals = len(e)
        # Rearrange vectors
        ind = np.empty(n_vals, dtype=int)
        ind[::2] = np.arange(n_vals - 1, n_vals // 2 - 1, -1)
        ind[1::2] = np.arange(0, n_vals // 2)
        w = w[:, ind]  # first, last, second, second last, third, ...
        self.filters_ = w.T
        self.patterns_ = lg.pinv(w)

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in epochs_data])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

def _regularized_covariance(data, reg=None):

    if reg is None:
        cov = np.cov(data)
    
    return cov

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


def computeAvgFFT(epochs, ch, fs, epoch_idx):
    
    n_samples = epochs.shape[2]
    
    N = 512
    
    T = 1.0 / fs

    n_epochs = epochs.shape[0]
    
    ft = np.zeros(N)
    A = np.zeros(N/2)
 
    for i in epoch_idx:
        epoch = epochs[i,ch,:]      
        ft = fft(epoch, N)
        A += 2.0/N * np.abs(ft[0:N/2])
    
    A = A / n_epochs        
    freq = np.linspace(0.0, 1.0/(2.0*T), N/2)
    
    return freq, A
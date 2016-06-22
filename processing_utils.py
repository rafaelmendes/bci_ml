import biosig # used to import gdf data files
import numpy as np # numpy - used for array and matrices operations
import math as math # used for basic mathematical operations

# Filter design Libs: http://docs.scipy.org/doc/scipy/reference/signal.html
# from scipy.signal import firwin # Design a fir filter
# from scipy.signal import freqz # Plots filter frequency response
# from scipy.signal import lfilter # Applies designed filter to data
# from scipy.signal import filtfilt # Applies designed filter to data
# from scipy.signal import butter # Applies designed filter to data

import scipy.signal as sp
import scipy.linalg as lg

from scipy.fftpack import fft

from pylab import plot, show, pi

def loadBiosig(fname):
    """Loads biosig compatible datasets.
    Parameters
    ----------
    fname : path to dataset

    Returns
    -------
    data : dataset as a numpy matrix
    sample_rate : dataset sample rate

    Examples
    --------
    >>> data_path = "/PATH/TO/PATH/dataset.gdf"
    >>> EEGdata = loadBiosig(data_path)
    """

    # Loads GDF competition data
    HDR = biosig.constructHDR(0, 0)
    HDR = biosig.sopen(fname, 'r', HDR)

    sample_rate = HDR.SampleRate
    data = biosig.sread(0, HDR.NRec, HDR)

    biosig.sclose(HDR)
    biosig.destructHDR(HDR)

    return data, sample_rate

def readEvents(csv_path):
    
    t_events = np.loadtxt(open(csv_path,"rb"), skiprows=1, usecols=(2,0,1))

    return t_events

def selectChannels(data_in, channels):
    
    # Select channels
    data_out = data_in[channels,:]

    return data_out


def extractEpoch(data_in, labels, code, epoch_start, epoch_end):

    n_channels = data_in.shape[0]

    index = np.array(np.where(labels[:,0] == code))
    # sample = np.floor (1e-6 * labels[index,1] * sample_rate)

    sample = labels[index,1] # extract the sample position which corresponds to the beggining of each trial

    q = (epoch_end - epoch_start) # 3 seconds of samples per epoch 

    data_out = np.zeros((index.size, n_channels, q))

    # data_out = np.array([])
    j = 0
    for i in sample.flat:
        data_out[j,:,:] = data_in[:,(i + epoch_start):(i + epoch_end)]
        j+=1

    # data_out has dimensions = [epoch, channels, samples]

    return data_out


def Filter(data_in, f1, f2, sample_rate, filter_order):

    
    nyq_rate = sample_rate / 2
    w1, w2 = f1 / nyq_rate, f2 / nyq_rate


    b, a = sp.butter(filter_order, [f2 / nyq_rate], btype='low')
    data_out = sp.filtfilt(b, a, data_in)
    
    b, a = sp.butter(filter_order, [f1 / nyq_rate], btype='high')
    data_out = sp.filtfilt(b, a, data_out)

    return data_out


def designCSP(dataA, dataB, nb):

    # return v, a, d
    n_channels = dataA.shape[0]
    q = dataA.shape[1]

    cA = np.zeros([dataA.shape[0], n_channels, n_channels])
    cB = np.zeros([dataB.shape[0], n_channels, n_channels])

    # Compute the covariance matrix of each epoch of the same class (A and B)
    for i in range(dataA.shape[0]):
        # cA[i,...] = np.cov(dataA[i,:,:])
        c = np.dot(dataA[i,:,:], dataA[i,:,:].transpose())
        cA[i,...] = c / (np.trace(c) * q) 
        # cA[i,...] = c


    cA_mean = cA.mean(0) # compute the mean of the covariance matrices of each epoch

    for i in range(dataB.shape[0]):
        # cB[i,...] = np.cov(dataB[i,:,:])
        c = np.dot(dataB[i,:,:], dataB[i,:,:].transpose())
        cB[i,...] = c / (np.trace(c) * q) 
        # cB[i,...] = c

    cB_mean = cB.mean(0) # compute the mean of the covariance matrices of each epoch

    lamb, v = lg.eig(cA_mean + cB_mean) # eigvalue and eigvector decomposition

    lamb = lamb.real # return only real part of eigen vector

    index = np.argsort(lamb) # returns the index of array lamb in crescent order

    index = index[::-1] # reverse the order, now index has the positon of lamb in descendent order

    lamb = lamb[index] # sort the eingenvalues in descendent order

    v = v.take(index, axis=1) # the same goes for the eigenvectors along axis y

    Q = np.dot(np.diag(1 / np.sqrt(lamb)), v.transpose()) # whitening matrix computation

    D, V = lg.eig(np.dot(Q, np.dot(cA_mean, Q.transpose()))) # eig decomposition of whiten cov matrix

    W_full = np.dot(V.transpose(), Q)

    W = W_full[:nb,:] # select only the neighbours defined in NB; get the first 3 eigenvectors
    W = np.vstack((W,W_full[-nb:,:])) # get the three last eigenvectors

    return W

def applyCSP(W, data_in):

    q = data_in.shape[1]
    c = np.dot(data_in, data_in.transpose())
    cIN = c / (np.trace(c) * q) 
    cY = np.dot(W, np.dot(cIN, W.transpose()))

    return cY


def featExtract(data_in):

    feat = np.log(np.diag(data_in))
    return feat

def designLDA(dataA, dataB):

    biasA = np.ones([dataA.shape[0],1])
    biasB = np.ones([dataB.shape[0],1])

    dataA = np.concatenate((biasA, dataA), axis = 1)
    dataB = np.concatenate((biasB, dataB), axis = 1)

    dataA_mean = dataA.mean(axis = 0)
    dataB_mean = dataB.mean(axis = 0)

    Sa = np.dot(dataA.T,dataA) - np.dot(dataA_mean,dataA_mean.transpose())
    Sb = np.dot(dataB.T,dataB) - np.dot(dataB_mean,dataB_mean.transpose())

    W = np.dot(lg.inv(Sa + Sb), (dataA_mean - dataB_mean))

    b = 0.5 * np.dot(W.transpose(), (dataA_mean + dataB_mean))

    return W, b

def applyLDA(W, data_in):

    bias = np.ones([data_in.shape[0],1])
    data_in = np.concatenate((bias, data_in), axis = 1)

    L = np.dot(W, data_in.T) # compute linear scores

    return L

def classifyIF(score, b):

    result = np.zeros(score.shape[0])

    for i in range(score.shape[0]):
        if score[i] < b:
            result[i] = 0
        else:
            result[i] = 1

    return result

def computeAcc(resultsA, resultsB):
    count = 0
    count += resultsA.size - sum(resultsA)
    count += sum(resultsB) 

    acc = count / (resultsA.size + resultsB.size)

    return acc * 100

def nanCleaner(data):
    for i in range(data.shape[0]):
        bad_idx = np.isnan(data[i, ...])
        data[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], data[i, ~bad_idx])

    return data

def computeAvgFFT(data, ch):
    T = 1 / sample_rate

    n = data.shape[1]

    ft = fft(data[:,ch,:]) / n 
    freq = np.linspace(0.0, 1.0/(2.0*T), n/2)

    plot.plot(freq, 2.0/N * np.abs(ft[0:N/2]))
    plot.grid()
    plot.show()

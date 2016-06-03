import numpy as np
import json

from DataProcessing import DataFiltering, DataLearner

import mne

def loadDataAsMatrix(path, cols=[]):
    """Loads text file content as numpy matrix
    Parameters
    ----------
    path : path to text file
    
    cols : order of columns to be read

    Returns
    -------
    matrix : numpy matrix, shape as written in txt

    Examples
    --------
    >>> data_path = "/PATH/TO/FILE/somematrix.txt"
    >>> matrix_data = loadAsMatrix(data_path)
    """
    
    if cols == []:
        matrix = np.loadtxt(open(path,"rb"), skiprows=1)
        
    else:
        matrix = np.loadtxt(open(path,"rb"), skiprows=1, usecols=cols)

    # return np.fliplr(matrix.T).T
    return matrix

def extractEpochs(data, events_id, tmin, tmax):
    """Extracts the epochs from data based on event information
    Parameters
    ----------
    data : raw data in mne format
    
    event_id : labels of each class
    
    tmin: time in seconds at which the epoch starts (event as reference) 
    
    tmax: time in seconds at which the epoch ends (event as reference) 

    Returns
    -------
    epochs: epochs in mne format
    
    labels: labels of each extracted epoch

    Examples
    --------
    >>> data, sfreq = loadBiosig(data_eval_path)
    >>> raw = mne.io.RawArray(data, info)
    >>> csv_path = "/PATH/TO/CSVFILE/events.csv"
    >>> raw = addEvents(raw, eval_events_path)
    >>> event_id = dict(LH=769, RH=770)
    >>> tmin, tmax = 1, 3 # epoch starts 1 sec after event and ends 3 sec after
    >>> epochs_train, labels_train = extractEpochs(raw, event_id, tmin, tmax)
    
    """

    events_list = mne.find_events(data, stim_channel='stim_clean')

    picks = mne.pick_types(data.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = mne.Epochs(data, events_list, events_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True, add_eeg_ref=False, verbose=False)
    labels = epochs.events[:, -1]
    
    return epochs, labels

def saveMatrixAsTxt(data_in, path, mode = 'a'):

    with open(path, mode) as data_file:    
        np.savetxt(data_file, data_in)

def loadChannelLabels(path):
    # if os.path.exists("data/rafael/precal_config"):
    with open(path, "r") as data_file:    
        data = json.load(data_file)

    return data["ch_labels"].split(' ')

def readEvents(events_path):

    e = np.loadtxt(events_path, skiprows=0)
    # insert dummy column to fit mne event list format
    t_events = np.insert(e, 1, values=0, axis=1)
    t_events = t_events.astype(int) # convert to integer

    return t_events

def loadDataForMNE(pathToConfig, pathToData, sfreq):

    dp = DataFiltering()

    dp.DesignFilter(8, 30, sfreq, 7)

    data = loadDataAsMatrix(pathToData).T
    # data = nanCleaner(data)

    ch_names = loadChannelLabels(pathToConfig) 
    
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eog']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    print data.shape    
    
    mne_data = mne.io.RawArray(data, info)

    return mne_data

def addEventsToMNEData(data, pathToEvents):

    events_list = readEvents(pathToEvents)
    info_stim = mne.create_info(ch_names=['stim_clean'], sfreq=data.info['sfreq'], ch_types=['stim'])
    info_stim['buffer_size_sec'] = data.info['buffer_size_sec']
    data_dum = np.zeros([1, data._data.shape[1]])
    raw_stim = mne.io.RawArray(data_dum, info=info_stim)
    data.add_channels([raw_stim])
    data.add_events(events_list, stim_channel = 'stim_clean')

    return data



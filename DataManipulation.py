import numpy as np

def LoadDataAsMatrix(path, cols=[]):
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

def extractEpochs(data, events_list, events_id, tmin, tmax):
    """Extracts the epochs from data based on event information
    Parameters
    ----------
    data : raw data in mne format

    events_list: list of events in mne format,
    shape(time stamp (in samples), offset (can be a range arr), label)
    
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

    picks = pick_types(data.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(data, events_list, events_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True, add_eeg_ref=False, verbose=False)
    labels = epochs.events[:, -1]
    
    return epochs, labels

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

def saveMatrixAsTxt(data_in, path, mode = 'a'):

    with open(path, mode) as data_file:    
        np.savetxt(data_file, data_in)

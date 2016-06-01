
"""CSP + LDA approach.
Implements the CSP + LDA approach using a data from the V BCI competition
"""
import mne

from processing import *

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.decoding import CSP

import numpy as np

# DATASETS PATH
# data_train_path = "/arquivos/Documents/eeg_data/doutorado_cleison/data_set/A01T.set"
# data_eval_path = "/arquivos/Documents/eeg_data/doutorado_cleison/data_set/A01E.set"

subj = 9

data_train_path = "/arquivos/Documents/eeg_data/doutorado_cleison/A0" + str(subj) + "T.gdf"
data_eval_path = "/arquivos/Documents/eeg_data/doutorado_cleison/A0" + str(subj) + "E.gdf"
# filename = "/arquivos/downloads/testpport_1to100.bdf"

# EVENTS INFO PATH
train_events_path = "/arquivos/Documents/eeg_data/doutorado_cleison/train_events/A0" + str(subj) + "T.csv"
eval_events_path = "/arquivos/Documents/eeg_data/doutorado_cleison/true_labels/A0" + str(subj) + "E.csv"

# raw = mne.io.read_raw_eeglab(data_train_path)

data, fs = loadBiosig(data_train_path)
data = nanCleaner(data)

ch_names = ['Fz', 'EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG-C3', 'EEG7', 'EEG-Cz', 'EEG8', 
'EEG-C4', 'EEG9', 'EEG10', 'EEG11', 'EEG12', 'EEG13', 'EEG14', 'EEG15', 'EEG-Pz', 'EEG16', 'EEG17', 'EOG1', 'EOG2', 'EOG3' ]

sfreq = fs

ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eog', 'eog', 'eog' ]

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

raw = mne.io.RawArray(data, info)

events_list = readEvents(train_events_path)

info_stim = mne.create_info(ch_names=['stim_clean'], sfreq=raw.info['sfreq'], ch_types=['stim'])
info_stim['buffer_size_sec'] = raw.info['buffer_size_sec']
data_dum = np.zeros([1, data.shape[1]])
raw_stim = mne.io.RawArray(data_dum, info=info_stim)
raw.add_channels([raw_stim])

raw.add_events(events_list, stim_channel = None)

# Processing beggining:
tmin, tmax = 1, 3 # time before event, time after event
event_id = dict(LH=769, RH=770)

# Apply band-pass filter
raw.filter(8., 35., method='iir', filter_length=10)

events = find_events(raw, stim_channel='stim_clean')

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs_train = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True, add_eeg_ref=False, verbose=False)
labels_train = epochs_train.events[:, -1]

###############################################################################
# Repeat the same steps for validation data:

data, fs = loadBiosig(data_eval_path)
data = nanCleaner(data)

sfreq = fs

raw = mne.io.RawArray(data, info)

events_list = readEvents(eval_events_path)
events_list[:,0] = np.floor(events_list[:,0] * fs)

data_dum = np.zeros([1, data.shape[1]])
raw_stim = mne.io.RawArray(data_dum, info=info_stim)
raw.add_channels([raw_stim])

raw.add_events(events_list, stim_channel = None)

# Processing beggining:

# Apply band-pass filter
raw.filter(8., 35., method='iir', filter_length=10)

events = find_events(raw, stim_channel='stim_clean')

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs_eval = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True, add_eeg_ref=False, verbose=False)
labels_eval = epochs_evals.events[:, -1]


###############################################################################
# Classification with linear discrimant analysis

from sklearn.lda import LDA  # noqa

# Assemble a classifier
svc = LDA()
csp = CSP(n_components=3, reg=None, log=True, cov_est='epoch')

epochs_data_train = epochs_train.get_data()
epochs_data_eval = epochs_eval.get_data()

# Use scikit-learn Pipeline with cross_val_score function
from sklearn.pipeline import Pipeline  # noqa
clf = Pipeline([('CSP', csp), ('SVC', svc)])

clf.fit(epochs_data_train, labels_train)

score = clf.score(epochs_data_eval, labels_eval)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (score,
                                                          class_balance))
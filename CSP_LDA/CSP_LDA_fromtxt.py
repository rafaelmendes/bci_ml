
"""CSP + LDA approach.
Implements the CSP + LDA approach using a data from the V BCI competition
"""
import mne

from processing import *

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.decoding import CSP

import numpy as np

data_folder_path = "/home/rafael/codes/repo/bci_training_platform/data/session/rafael3_long/"

data_train_path = data_folder_path + "data_cal.txt"
data_train_path = data_folder_path + "data_val.txt"

# EVENTS INFO PATH
train_events_path = data_folder_path + "events_cal.txt"
eval_events_path = data_folder_path + "events_val.txt"
# raw = mne.io.read_raw_eeglab(data_train_path)

data = np.loadtxt(data_train_path).T
fs = 250
data = nanCleaner(data)

ch_names = ['c4', 'cz', 'c3', 'p4', 'pz', 'p3', 'oz', 'fpz']

sfreq = fs

ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eog' ]

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

raw = mne.io.RawArray(data, info)

events_list = np.loadtxt(train_events_path)
dum = np.ones(events_list.shape)
events_list = np.vstack((events_list, dum[:,0]))

info_stim = mne.create_info(ch_names=['stim_clean'], sfreq=raw.info['sfreq'], ch_types=['stim'])
info_stim['buffer_size_sec'] = raw.info['buffer_size_sec']
data_dum = np.zeros([1, data.shape[1]])
raw_stim = mne.io.RawArray(data_dum, info=info_stim)
raw.add_channels([raw_stim])

raw.add_events(events_list, stim_channel = None)

# Processing beggining:
tmin, tmax = 0, 2 # time before event, time after event
event_id = dict(LH=1, RH=2)

# Apply band-pass filter
raw.filter(8., 30., method='iir', filter_length=7)

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
data = np.loadtxt(data_eval_path)
data = nanCleaner(data)

sfreq = fs

raw = mne.io.RawArray(data, info)

events_list = np.loadtxt(eval_events_path)
events_list[:,0] = np.floor(events_list[:,0] * fs)

data_dum = np.zeros([1, data.shape[1]])
raw_stim = mne.io.RawArray(data_dum, info=info_stim)
raw.add_channels([raw_stim])

raw.add_events(events_list, stim_channel = None)

# Processing beggining:

# Apply band-pass filter
raw.filter(8., 30., method='iir', filter_length=7)

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
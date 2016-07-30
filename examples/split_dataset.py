import sys

sys.path.insert(1,'../')

from approach import Approach
import numpy as np

from random import randint

from DataManipulation import saveMatrixAsTxt 

DATA_FOLDER_PATH = "/arquivos/Documents/eeg_data/bci_comp_IV/standard_data/"
EVENTS_FOLDER_PATH = "/arquivos/Documents/eeg_data/bci_comp_IV/standard_events/"

DATA_PATH = DATA_FOLDER_PATH + "A05E.npy"

# EVENTS INFO PATH
EVENTS_PATH = EVENTS_FOLDER_PATH + "A05E.npy"

SAMPLING_FREQ = 250.0

# FILTER SPEC
LOWER_CUTOFF = 8.
UPPER_CUTOFF = 30.
FILT_ORDER = 7

# EPOCH EXTRACTION CONFIG:
EVENT_IDS = [1, 2, 3, 4]

T_MIN, T_MAX = 0,6  # time before event, time after event

CSP_N = 8

ap = Approach()
ap.defineApproach(SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF, FILT_ORDER, CSP_N, EVENT_IDS, T_MIN, T_MAX)

ap.setValidChannels(range(22))

data, ev = ap.loadData(DATA_PATH, EVENTS_PATH)

epochs, labels = ap.loadEpochs(data, ev)

epochs = ap.preProcess(epochs)


idx_1 = np.where(labels == 1)[0]
idx_2 = np.where(labels == 2)[0]
idx_3 = np.where(labels == 3)[0]
idx_4 = np.where(labels == 4)[0]


new_data = np.zeros([1,epochs.shape[1]])
new_events = np.zeros([1,2])


for j in range(4):
	for i in range(2):
		k = randint(0,len(idx_1))
		new_events = np.vstack([new_events, [1, new_data.shape[0]]])
		new_data = np.vstack([new_data, epochs[idx_1[i]].T])

	for i in range(2):
		k = randint(0,len(idx_2))
		new_events = np.vstack([new_events, [2, new_data.shape[0]]])
		new_data = np.vstack([new_data, epochs[idx_2[i]].T])

	for i in range(2):
		k = randint(0,len(idx_4))
		new_events = np.vstack([new_events, [4, new_data.shape[0]]])
		new_data = np.vstack([new_data, epochs[idx_4[i]].T])

new_data = np.delete(new_data, 0, axis= 0)
new_events = np.delete(new_events, 0, axis= 0)

SAVE_PATH = '/arquivos/Documents/eeg_data/bci_comp_IV/split_datasets/split7.npy'
NEW_EVENTS_PATH = '/arquivos/Documents/eeg_data/bci_comp_IV/split_datasets/split7_events.npy'

saveMatrixAsTxt(new_data, SAVE_PATH)
saveMatrixAsTxt(new_events, NEW_EVENTS_PATH)
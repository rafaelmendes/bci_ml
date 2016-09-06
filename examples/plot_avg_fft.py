import sys
sys.path.insert(0, '../')

from approach import Approach

from processing_utils import computeAvgFFT

import matplotlib.pyplot as plt

import numpy as np

DATA_FOLDER_PATH = "/home/rafael/repo/bci_training_platform/data/session/cleison_handvfeet/"

DATA_CAL_PATH = DATA_FOLDER_PATH + "data_cal.npy"
DATA_VAL_PATH = DATA_FOLDER_PATH + "data_val.npy"

# EVENTS INFO PATH
CAL_EVENTS_PATH = DATA_FOLDER_PATH + "events_cal.npy"
VAL_EVENTS_PATH = DATA_FOLDER_PATH + "events_val.npy"

SAMPLING_FREQ = 125.0

# FILTER SPEC
LOWER_CUTOFF = 2.
UPPER_CUTOFF = 30.
FILT_ORDER = 5

# EPOCH EXTRACTION CONFIG:
EVENT_IDS = [1,2] 

T_MIN, T_MAX = 2.5,4.5  # time before event, time after event

CSP_N = 12

ap = Approach()

ap.defineApproach(SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF, FILT_ORDER, CSP_N, EVENT_IDS, T_MIN, T_MAX)

ap.setPathToCal(DATA_CAL_PATH, CAL_EVENTS_PATH)
ap.setPathToVal(DATA_VAL_PATH, VAL_EVENTS_PATH)

ap.setValidChannels([-1])

data, events = ap.loadData(DATA_CAL_PATH, CAL_EVENTS_PATH)

data = ap.preProcess(data)

epochs, labels = ap.loadEpochs(data,events)

idx_1 = np.where(labels == 1)[0]
idx_2 = np.where(labels == 2)[0]

f, A1 = computeAvgFFT(epochs,0,125, idx_1)
f, A2 = computeAvgFFT(epochs,0,125, idx_2)

plt.plot(f, A1)
plt.plot(f, A2)

plt.show()
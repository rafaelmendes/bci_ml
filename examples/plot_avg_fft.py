import sys
sys.path.insert(0, '../')

from approach import Approach

from processing_utils import computeAvgFFT, computeAvgFFTWelch

import matplotlib.pyplot as plt

import numpy as np

DATA_FOLDER_PATH = "/home/rafael/repo/bci_training_platform/data/session/mario/"

DATA_CAL_PATH = DATA_FOLDER_PATH + "data_cal.npy"

# EVENTS INFO PATH
CAL_EVENTS_PATH = DATA_FOLDER_PATH + "events_cal.npy"

SAMPLING_FREQ = 125.0

# FILTER SPEC
LOWER_CUTOFF = 8.
UPPER_CUTOFF = 30.
FILT_ORDER = 5

# EPOCH EXTRACTION CONFIG:
EVENT_IDS = [1,2] 

T_MIN, T_MAX = -3,-1  # time before event, time after event

CSP_N = 12

ap = Approach()

ap.defineApproach(SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF, FILT_ORDER, CSP_N, EVENT_IDS, T_MIN, T_MAX)

ap.setPathToCal(DATA_CAL_PATH, CAL_EVENTS_PATH)

ap.setValidChannels(range(16))

data, events = ap.loadData(DATA_CAL_PATH, CAL_EVENTS_PATH)

data = ap.preProcess(data)

# nch = data.shape[0]
# Id = np.identity(nch)
# W = Id - (1.0 / nch) * np.dot(Id, Id.T)
# data = np.dot(W, data)

epochs, labels = ap.loadEpochs(data,events)

idx_1 = np.where(labels == 1)[0]
idx_2 = np.where(labels == 2)[0]

c3_idx = 0
c4_idx = 2

f, A1_c3 = computeAvgFFTWelch(epochs,c3_idx,SAMPLING_FREQ, idx_1)
f, A2_c3 = computeAvgFFTWelch(epochs,c3_idx,SAMPLING_FREQ, idx_2)

f, A1_c4 = computeAvgFFTWelch(epochs,c4_idx,SAMPLING_FREQ, idx_1)
f, A2_c4 = computeAvgFFTWelch(epochs,c4_idx,SAMPLING_FREQ, idx_2)

f = f * SAMPLING_FREQ

plt.plot(f, A1_c3, '-bo')
# plt.plot(f, A2_c3, '-go')
# plt.plot(f, A2_c3, '-gs')
# plt.plot(f, A2_c4, '-gs')
plt.grid(True)

# plt.legend(('Foot-C3', 'Foot-C4'), 
# 	loc='upper right', shadow=True)

plt.legend(('Lhand-C4', 'Rhand-C4'), 
	loc='upper right', shadow=True)

plt.title('6 a 8 s')

# plt.legend(('Left-C3', 'Right-C3','Left-C4', 'Right-C4'), 
# 	loc='upper right', shadow=True)

plt.show()
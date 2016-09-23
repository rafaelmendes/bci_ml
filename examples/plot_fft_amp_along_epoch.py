import sys
sys.path.insert(0, '../')

from approach import Approach

from processing_utils import computeAvgFFT, computeAvgFFTWelch

import matplotlib.pyplot as plt

import numpy as np

DATA_FOLDER_PATH = "/home/rafael/repo/bci_training_platform/data/session/cleison_hh_1/"

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


start_t = -3
end_t = 13

f_min = 9
f_max = 13

A1h = []
A2h = []

t = []
increment = 0.2

T_MIN = start_t
T_MAX = T_MIN + 2

while T_MAX < end_t:

	T_MIN += increment
	T_MAX += increment

	t.extend([T_MIN])

	CSP_N = 12

	ap = Approach()

	ap.defineApproach(SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF, FILT_ORDER, CSP_N, EVENT_IDS, T_MIN, T_MAX)

	ap.setPathToCal(DATA_CAL_PATH, CAL_EVENTS_PATH)

	ap.setValidChannels(range(16))
	ap.define_bad_epochs(50, None)


	data, events = ap.loadData(DATA_CAL_PATH, CAL_EVENTS_PATH)

	ref_channel = 8 # fcz

	data = ap.preProcess(data)

	data = data[:,:] - data[ref_channel]

	# nch = data.shape[0]
	# Id = np.identity(nch)
	# W = Id - (1.0 / nch) * np.dot(Id, Id.T)
	# data = np.dot(W, data)

	epochs, labels = ap.loadEpochs(data,events)

	idx_1 = np.where(labels == 1)[0]
	idx_2 = np.where(labels == 2)[0]

	ch_idx = 9
	ch2_idx = 13

	f, A1 = computeAvgFFTWelch(epochs,ch_idx,SAMPLING_FREQ, idx_1)
	f, A2 = computeAvgFFTWelch(epochs,ch2_idx,SAMPLING_FREQ, idx_1)

	f = f * SAMPLING_FREQ

	idx_f = np.where((f > f_min) * (f < f_max))

	A_mean = np.mean(A1[idx_f])
	A1h.extend([A_mean])

	A_mean = np.mean(A2[idx_f])
	A2h.extend([A_mean])




plt.plot(t, A1h, '-bo')
plt.plot(t, A2h, '-gs')
plt.grid(True)

plt.legend(('Lhand-Left', 'Lhand-Right'), 
	loc='upper right', shadow=True)

# plt.title('6 a 8 s')

# # plt.legend(('Left-C3', 'Right-C3','Left-C4', 'Right-C4'), 
# # 	loc='upper right', shadow=True)

plt.show()
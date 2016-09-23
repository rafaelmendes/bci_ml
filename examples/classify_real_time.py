import sys
sys.path.insert(0, '../')

from approach import Approach


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

T_MIN, T_MAX = 3,5  # time before event, time after event

CSP_N = 12

ap = Approach()

ap.defineApproach(SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF, FILT_ORDER, CSP_N, EVENT_IDS, T_MIN, T_MAX)

ap.setPathToCal(DATA_CAL_PATH, CAL_EVENTS_PATH)

ap.setValidChannels([-1])
ap.define_bad_epochs(100)

autoscore = ap.trainModel()

crossvalscore = ap.cross_validate_model(10, 0.2)


print autoscore
print crossvalscore




## test on single epoch
import numpy as np

data, events = ap.loadData(DATA_CAL_PATH, CAL_EVENTS_PATH)

buf = np.array([data.shape[0], 250])

increment = 50

prob1h = []
prob2h = []
labelh = []

i = 0
tinit, tend = 0, 250

while tend < data.shape[1]: 

	idx = range(tinit,tend)

	buf = data[:,idx] 

	p = ap.applyModelOnEpoch(buf, out_param = 'prob')[0]
	g = ap.applyModelOnEpoch(buf, out_param = 'label')

	prob1h.extend([p[0]])
	prob2h.extend([p[1]])
	labelh.extend([g])

	tinit += increment
	tend += increment

smooth_window = 30

smooth_prob1h = np.convolve(prob1h, np.ones((smooth_window,))/smooth_window, mode='valid')
smooth_prob2h = np.convolve(prob2h, np.ones((smooth_window,))/smooth_window, mode='valid')


# PLOTS
import matplotlib.pyplot as plt

labels_pos = events[0]
labels = events[1]

n_samples = smooth_prob1h.shape[0]

samples = range(n_samples)

plt.plot(samples, smooth_prob1h)
# plt.plot(samples, smooth_prob2h)
# plt.plot(labelh)

plt.grid(True)

# plt.fill_between([0,10000],[0.5,1])

plt.show()


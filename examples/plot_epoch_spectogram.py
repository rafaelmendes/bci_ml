import sys
sys.path.insert(0, '../')

from approach import Approach
from processing_utils import computeAvgFFT, plot_spectogram, compute_time_avg
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
LOWER_CUTOFF = 8.
UPPER_CUTOFF = 15.
FILT_ORDER = 5

# EPOCH EXTRACTION CONFIG:
EVENT_IDS = [1,2] 

T_MIN, T_MAX = -4,4  # time before event, time after event

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

avg1 = compute_time_avg(epochs,0, idx_1)
avg2 = compute_time_avg(epochs,1, idx_1)


plot_spectogram(avg1, SAMPLING_FREQ)
plot_spectogram(avg2, SAMPLING_FREQ)


# test
# f = [10, 20, 30]
# sample = len(avg1)
# t = np.arange(sample)
# y1 = np.sin(2 * np.pi * f[0] * t / SAMPLING_FREQ)
# y2 = 0.2*np.sin(2 * np.pi * f[1] * t / SAMPLING_FREQ)
# y3 = 0.1*np.sin(2 * np.pi * f[2] * t / SAMPLING_FREQ)

# y = y1+y2+y3

# plot_spectogram(y, SAMPLING_FREQ)

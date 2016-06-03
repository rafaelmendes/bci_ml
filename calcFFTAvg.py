# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:36:44 2016

@author: rafael
"""

"""CSP + LDA approach.
Implements the CSP + LDA approach using a data from the V BCI competition
"""
from DataManipulation import *
from DataProcessing import *

import numpy as np

import matplotlib.pyplot as plt

DATA_FOLDER_PATH = "/home/rafael/codes/repo/bci_training_platform/data/session/rafael3_long/"

DATA_CAL_PATH = DATA_FOLDER_PATH + "data_cal.txt"
DATA_VAL_PATH = DATA_FOLDER_PATH + "data_val.txt"

ACQ_CONFIG_PATH = DATA_FOLDER_PATH + "openbci_config.txt"

# EVENTS INFO PATH
CAL_EVENTS_PATH = DATA_FOLDER_PATH + "events_cal.txt"
VAL_EVENTS_PATH = DATA_FOLDER_PATH + "events_val.txt"

SAMPLING_FREQ = 250

# FILTER SPEC
LOWER_CUTOFF = 8.
UPPER_CUTTOF = 30.
FILT_ORDER = 10

# EPOCH EXTRACTION CONFIG:
EVENT_IDS = dict(LH=1, RH=2)

T_MIN, T_MAX = 0, 2 # time before event, time after event

### MAIN ###

cal_data = loadDataForMNE(ACQ_CONFIG_PATH, DATA_CAL_PATH, SAMPLING_FREQ)

cal_data = addEventsToMNEData(cal_data, CAL_EVENTS_PATH)

# FEATURE EXTRACTION:
cal_epochs, cal_labels = extractEpochs(cal_data, EVENT_IDS, T_MIN, T_MAX)

cal_epochs_d = cal_epochs.get_data()

# LOAD VALIDATION DATA:
val_data = loadDataForMNE(ACQ_CONFIG_PATH, DATA_VAL_PATH, SAMPLING_FREQ)

val_data = addEventsToMNEData(val_data, VAL_EVENTS_PATH)

# FEATURE EXTRACTION:
val_epochs, val_labels = extractEpochs(val_data, EVENT_IDS, T_MIN, T_MAX)

val_epochs_d = val_epochs.get_data()

## PLOT DATA
#plt.plot(cal_epochs_d[0,:,:].T)
#plt.show()

class_1_idx = np.where(cal_labels == 1)[0] # left hand
class_2_idx = np.where(cal_labels == 2)[0] # right hand

f, mean1 = computeAvgFFT(cal_epochs_d, 2, SAMPLING_FREQ, class_1_idx)
f, mean2 = computeAvgFFT(cal_epochs_d, 2, SAMPLING_FREQ, class_2_idx)

plt.plot(f,mean1, color = [0,0,1])
plt.plot(f,mean2, color = [0,1,0])
plt.show()





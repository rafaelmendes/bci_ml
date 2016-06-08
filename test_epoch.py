"""CSP + LDA approach.
Implements the CSP + LDA approach using a data from the V BCI competition
"""
from DataManipulation import *
from DataProcessing import *

import numpy as np

DATA_FOLDER_PATH = "/home/rafaelmd/codes/repo/bci_training_platform/data/session/rafael3_long/"

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
FILT_ORDER = 4

# EPOCH EXTRACTION CONFIG:
EVENT_IDS = dict(LH=1, RH=2)

TMIN, TMAX = 0.5, 2.5 # time before event, time after event
SMIN = TMIN * SAMPLING_FREQ
SMAX = TMAX * SAMPLING_FREQ

### MAIN ###

dp = DataFiltering()

dp.DesignFilter(LOWER_CUTOFF, UPPER_CUTTOF, SAMPLING_FREQ, FILT_ORDER)

# LOAD CALIBRATION DATA:
data_cal = loadDataAsMatrix(DATA_CAL_PATH).T
events_list_cal = readEvents(CAL_EVENTS_PATH)

data_cal = dp.ApplyFilter(data_cal)


# FEATURE EXTRACTION:
epochs_cal, labels_cal = extractEpochs(data_cal, events_list_cal, SMIN, SMAX)

dl = DataLearner()

dl.DesignLDA()
dl.DesignCSP(6)
dl.AssembleLearner()
dl.Learn(epochs_cal, labels_cal)

# LOAD VALIDATION DATA:
data_val = loadDataAsMatrix(DATA_VAL_PATH).T
events_list_val = readEvents(VAL_EVENTS_PATH)

data_val = dp.ApplyFilter(data_val)

# FEATURE EXTRACTION:
epochs_val, labels_val = extractEpochs(data_val, events_list_val, SMIN, SMAX)

dl.Evaluate(epochs_val, labels_val)

dl.PrintResults()
















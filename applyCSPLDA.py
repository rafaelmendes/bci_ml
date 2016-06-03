"""CSP + LDA approach.
Implements the CSP + LDA approach using a data from the V BCI competition
"""
from DataManipulation import *
from DataProcessing import *

import numpy as np

DATA_FOLDER_PATH = "/home/rafaelmd/codes/repo/bci_training_platform/data/session/rafael4_long/"

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

T_MIN, T_MAX = 2, 4 # time before event, time after event

### MAIN ###

cal_data = loadDataForMNE(ACQ_CONFIG_PATH, DATA_CAL_PATH, SAMPLING_FREQ)

cal_data = addEventsToMNEData(cal_data, CAL_EVENTS_PATH)

# FEATURE EXTRACTION:
cal_epochs, cal_labels = extractEpochs(cal_data, EVENT_IDS, T_MIN, T_MAX)

dl = DataLearner()

dl.DesignLDA()
dl.DesignCSP(6)
dl.AssembleLearner()
dl.Learn(cal_epochs.get_data(), cal_labels)

# LOAD VALIDATION DATA:
val_data = loadDataForMNE(ACQ_CONFIG_PATH, DATA_VAL_PATH, SAMPLING_FREQ)

val_data = addEventsToMNEData(val_data, VAL_EVENTS_PATH)

# FEATURE EXTRACTION:
val_epochs, val_labels = extractEpochs(val_data, EVENT_IDS, T_MIN, T_MAX)

dl.Evaluate(val_epochs.get_data(), val_labels)

dl.PrintResults()







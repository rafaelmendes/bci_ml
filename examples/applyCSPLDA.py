"""CSP + LDA approach.
Implements the CSP + LDA approach using a data from the V BCI competition
"""
import os, sys, inspect

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split \
	(inspect.getfile( inspect.currentframe() ))[0],'../approaches')))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from CSPLDAapproach import *

DATA_FOLDER_PATH = "/home/rafael/codes/repo/bci_training_platform/data/session/old/cleison_17_jun_16ch/"

DATA_CAL_PATH = DATA_FOLDER_PATH + "data_cal.txt"
DATA_VAL_PATH = DATA_FOLDER_PATH + "data_val.txt"

# EVENTS INFO PATH
CAL_EVENTS_PATH = DATA_FOLDER_PATH + "events_cal.txt"
VAL_EVENTS_PATH = DATA_FOLDER_PATH + "events_val.txt"

SAMPLING_FREQ = 125.0

# FILTER SPEC
LOWER_CUTOFF = 8.
UPPER_CUTOFF = 30.
FILT_ORDER = 7

# EPOCH EXTRACTION CONFIG:
EVENT_IDS = [1,2] 

T_MIN, T_MAX = 0.5, 4 # time before event, time after event

CSP_N = 8

### MAIN ###

results = apply_ml(DATA_CAL_PATH, CAL_EVENTS_PATH, DATA_VAL_PATH, VAL_EVENTS_PATH, 
		SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF, FILT_ORDER, CSP_N, EVENT_IDS, T_MIN, T_MAX)

print results







"""CSP + LDA approach.
Implements the CSP + LDA approach using a data from the V BCI competition
"""
import os, sys, inspect

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split \
	(inspect.getfile( inspect.currentframe() ))[0],'../')))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from DataManipulation import *
from DataProcessing import *

import numpy as np


### MAIN ###
def apply_ml(DATA_CAL_PATH, CAL_EVENTS_PATH, DATA_VAL_PATH, VAL_EVENTS_PATH, 
		SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF, FILT_ORDER, CSP_N, EVENT_IDS, T_MIN, T_MAX):

	SMIN = T_MIN * SAMPLING_FREQ
	SMAX = T_MAX * SAMPLING_FREQ

	### MAIN ###

	dp = DataFiltering()

	dp.DesignFilter(LOWER_CUTOFF, UPPER_CUTOFF, SAMPLING_FREQ, FILT_ORDER)

	# LOAD CALIBRATION DATA:
	data_cal = loadDataAsMatrix(DATA_CAL_PATH).T
	events_list_cal = readEvents(CAL_EVENTS_PATH)

	data_cal = dp.ApplyFilter(data_cal)


	# FEATURE EXTRACTION:
	epochs_cal, labels_cal = extractEpochs(data_cal, events_list_cal, SMIN, SMAX)

	dl = DataLearner()

	dl.DesignLDA()
	dl.DesignCSP(CSP_N)
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

	return dl.GetResults()







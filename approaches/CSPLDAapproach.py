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

from processing_utils import loadBiosig, nanCleaner

import numpy as np

import math

### MAIN ###
def apply_ml(DATA_CAL_PATH, CAL_EVENTS_PATH, DATA_VAL_PATH, VAL_EVENTS_PATH, 
		SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF, FILT_ORDER, CSP_N, EVENT_IDS, T_MIN, T_MAX):

	### MAIN ###

	dp = Filter(LOWER_CUTOFF, UPPER_CUTOFF, SAMPLING_FREQ, FILT_ORDER)

	# # LOAD CALIBRATION DATA:
	# if DATA_CAL_PATH[-3:] == 'gdf':
	# 	data_cal, SAMPLING_FREQ = loadBiosig(DATA_CAL_PATH).T
	# 	data_cal = nanCleaner(data_cal)

	# else:
	data_cal = loadDataAsMatrix(DATA_CAL_PATH).T
	
	events_list_cal = readEvents(CAL_EVENTS_PATH)

	data_cal = dp.ApplyFilter(data_cal)

	# FEATURE EXTRACTION:
	SMIN = int(math.floor(T_MIN * SAMPLING_FREQ))
	SMAX = int(math.floor(T_MAX * SAMPLING_FREQ))

	epochs_cal, labels_cal = extractEpochs(data_cal, events_list_cal, SMIN, SMAX)

	dl = Learner()

	dl.DesignLDA()
	dl.DesignCSP(CSP_N)
	dl.AssembleLearner()
	dl.Learn(epochs_cal, labels_cal)

	data_val = loadDataAsMatrix(DATA_VAL_PATH).T

	events_list_val = readEvents(VAL_EVENTS_PATH)

	data_val = dp.ApplyFilter(data_val)

	# FEATURE EXTRACTION:
	epochs_val, labels_val = extractEpochs(data_val, events_list_val, SMIN, SMAX)

	dl.EvaluateSet(epochs_val, labels_val)

	dl.PrintResults()

	return dl.GetResults()






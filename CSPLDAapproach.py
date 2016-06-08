"""CSP + LDA approach.
Implements the CSP + LDA approach using a data from the V BCI competition
"""
from DataManipulation import *
from DataProcessing import *

import numpy as np

### MAIN ###
def apply_ml(DATA_CAL_PATH, CAL_EVENTS_PATH, DATA_VAL_PATH, VAL_EVENTS_PATH, SAMPLING_FREQ, EVENT_IDS, T_MIN, T_MAX):

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

	return dl.GetResults()







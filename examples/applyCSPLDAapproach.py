

import sys
sys.path.insert(0, '../')

from approach import Approach


DATA_FOLDER_PATH = "/arquivos/mestrado/repo/bci_training_platform/data/session/cleison_24_jun_16_3_0_mao/"

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

T_MIN, T_MAX = 4,6  # time before event, time after event

CSP_N = 12


ap = Approach(SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF, FILT_ORDER, CSP_N, EVENT_IDS, T_MIN, T_MAX)

ap.loadCalData(DATA_CAL_PATH, CAL_EVENTS_PATH)
ap.loadValData(DATA_VAL_PATH, VAL_EVENTS_PATH)

autoscore = ap.trainModel()

valscore = ap.validateModel()


print autoscore
print valscore


## test on single epoch

i = 0
while i < len(ap.labels_cal):
	epoch_number = i

	e = ap.epochs_cal[epoch_number,:,:] #get epoch_number th epoch from calibration dataset

	p = ap.applyModelOnEpoch(e, out_param = 'prob')
	g = ap.applyModelOnEpoch(e, out_param = 'label')

	right = ap.labels_cal[epoch_number]

	print 'Mister M says probabilities: ', p
	print 'Mister M says: ', g
	print 'The right class was: ', right
	print '-------------------------'
	print '-------------------------'


	i += 1
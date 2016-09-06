import sys
sys.path.insert(0, '../')

from approach import Approach


DATA_FOLDER_PATH = "/home/rafael/repo/bci_training_platform/data/session/cleison_handvfeet/"

DATA_CAL_PATH = DATA_FOLDER_PATH + "data_cal.npy"
DATA_VAL_PATH = DATA_FOLDER_PATH + "data_val.npy"

# EVENTS INFO PATH
CAL_EVENTS_PATH = DATA_FOLDER_PATH + "events_cal.npy"
VAL_EVENTS_PATH = DATA_FOLDER_PATH + "events_val.npy"

SAMPLING_FREQ = 125.0

# FILTER SPEC
LOWER_CUTOFF = 8.
UPPER_CUTOFF = 30.
FILT_ORDER = 5

# EPOCH EXTRACTION CONFIG:
EVENT_IDS = [1,2] 

T_MIN, T_MAX = 2.5,4.5  # time before event, time after event

CSP_N = 12

ap = Approach()

ap.defineApproach(SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF, FILT_ORDER, CSP_N, EVENT_IDS, T_MIN, T_MAX)

ap.setPathToCal(DATA_CAL_PATH, CAL_EVENTS_PATH)
ap.setPathToVal(DATA_VAL_PATH, VAL_EVENTS_PATH)

ap.setValidChannels([-1])
ap.set_balance_epochs(False)

crossvalscore = ap.cross_validate_model()

print 'Cross Validation result: ', crossvalscore

autoscore = ap.trainModel()

valscore = ap.validateModel()

print 'SelfValidation result: ', autoscore
print 'Validation result: ', valscore
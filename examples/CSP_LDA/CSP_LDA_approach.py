
"""CSP + LDA approach.
Implements the CSP + LDA approach using a data from the V BCI competition
"""

from processing import *

# DATASETS PATH
data_train_path = "/arquivos/Documents/eeg_data/doutorado_cleison/A01T.gdf"
data_eval_path = "/arquivos/Documents/eeg_data/doutorado_cleison/A01E.gdf"
# filename = "/arquivos/downloads/testpport_1to100.bdf"

# EVENTS INFO PATH
train_events_path = "/arquivos/Documents/eeg_data/doutorado_cleison/train_events/A01T.csv"
eval_events_path = "/arquivos/Documents/eeg_data/doutorado_cleison/true_labels/A01E.csv"

# Main ------------------------------------------------------------------
# -----------------------------------------------------------------------

# DATASET VARIABLES:
codeA = 769 # Class 1 label = left hand mov
codeB = 770 # Class 2 label = right hand mov

channels = range(22) # select only EEG channels (exclude EOG)
CSP_N = 3 # number of CSP neighbours

# Training Stage:
# -----------------------------------------------------------------------

# EXTRACT EVENTS AND SAVE IT IN A NP ARRAY. COL 1 = TYPE; COL 2 = LATENCY
t_events = np.loadtxt(open(train_events_path,"rb"), skiprows=1, usecols=(1,2))

# LOAD DATASETS
data_train_raw, sample_rate = loadBiosig(data_train_path)
data_train_raw = selectChannels(data_train_raw, channels)

# Data Filter
filter_order = 5
f1, f2 = 8.0, 30.0

data_train = nanCleaner(data_train_raw)
data_train = Filter(data_train, f1, f2, sample_rate, filter_order)

epoch_start = 2.5 * sample_rate
epoch_end = 4.5 * sample_rate

epoch_trainA = extractEpoch(data_train, t_events, codeA, epoch_start, epoch_end)
epoch_trainB = extractEpoch(data_train, t_events, codeB, epoch_start, epoch_end)

# Train CSP model
W_CSP = designCSP(epoch_trainA, epoch_trainB, CSP_N)

# Apply CSP and extract Features
featA = np.array([0,0,0,0,0,0])

for i in range(epoch_trainA.shape[0]):
    data_spatialF = applyCSP(W_CSP, epoch_trainA[i,...])
    featA = np.vstack((featA,featExtract(data_spatialF)))

featB = np.array([0,0,0,0,0,0])

for i in range(epoch_trainB.shape[0]):
    data_spatialF = applyCSP(W_CSP, epoch_trainB[i,...])
    featB = np.vstack((featB,featExtract(data_spatialF)))

# Remove initilization row (gambiarra, arrumar depois)
featA = np.delete(featA, 0 , 0)
featB = np.delete(featB, 0 , 0)

# Train LDA model
W_LDA, th = designLDA(featA, featB)

# Apply LDA classifier
LA = applyLDA(W_LDA, featA) # for features of class A
LB = applyLDA(W_LDA, featB) # for features of class A

resultsA = classifyIF(LA, th)
resultsB = classifyIF(LB, th)


# Training Performance Evaluation:
# -----------------------------------------------------------------------

Accuracy = computeAcc(resultsA, resultsB)

print "Accuracy: ", Accuracy

# Eval Stage:
# -----------------------------------------------------------------------

e_events = np.loadtxt(open(eval_events_path,"rb"), skiprows=1, usecols=(1,2))
#     data_out = np.dot(W, data_in)
data_eval_raw = load(data_eval_path)

data_eval_raw = selectChannels(data_eval_raw, channels)

data_eval = nanCleaner(data_eval_raw)
data_eval = Filter(data_eval, f1, f2, sample_rate, filter_order)
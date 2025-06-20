import numpy as np
import pandas as pd
import wfdb
import ast

sampling_rate = 100
path = "data/physionet.org/files/ptb-xl/1.0.1/"


def load_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + filename) for filename in df.filename_lr]
    elif sampling_rate == 500:
        data = [wfdb.rdsamp(path + filename) for filename in df.filename_hr]
    
    data = [signal for signal, meta in data]
    
    return data


sampling_rate = 100
Y = pd.read_csv(path + "ptbxl_database.csv", index_col = 'ecg_id')
X = load_data(Y, sampling_rate, path)

Y.scp_codes = Y.scp_codes.apply(lambda X: ast.literal_eval(Y.scp_codes))

valid_fold = 9
test_fold  = 10

X_train = X[np.where(Y.strat_fold != valid_fold and Y.strat_fold != test_fold)]
Y_train = Y[np.where(Y.strat_fold != valid_fold and Y.strat_fold != test_fold)]

X_valid = X[np.where(Y.strat_fold == valid_fold)]
Y_valid = Y[np.where(Y.strat_fold == valid_fold)]

X_test  = X[np.where(Y.strat_fold == test_fold)]
Y_test  = Y[np.where(Y.strat_fold == test_fold)]

n_batch = 32
n_epoch = 10
learning_rate = 0.1
# criterion = 
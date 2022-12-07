

import random
import time

import numpy as np
import scipy.linalg as la
from scipy.stats import pearsonr as p
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.svm import SVR

import pct.pipeline.feature_ext as f
import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline

from pct.eval.eval_regression import eval_reg
from pct.eval.pretty_table import print_pretty_table


def main():
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels_all = parsing.parse_all_labels()
    filename_df = pipeline.get_filename_df(labels_all, raw_data_path, force=True)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_all)
    
    X_full = pipeline.get_X_full(filename_df, labels_all, Y_full_reg, force=True)

    epsilon = 0.05
    C = 5
    model = SVR(C=C, epsilon=epsilon, kernel="rbf")
    model.fit(X_full, Y_full_reg)
    N = 500
    
    pred_t = model.predict(X_full)

    description, test_eval = eval_reg(Y_full_reg, pred_t, labels_all)
    scores = np.zeros((N, len(test_eval)))
    
    # train svr
    for i in range(N):
        train_inds, val_inds = pipeline.train_val_patient_split(labels_all)
        # train_inds, val_inds = pipeline.get_stratified_Kfold_split(labels_all, Y_full_reg)
        X_train, m, s = f.normalize_features(X_full[train_inds, :])
        Y_train = Y_full_reg[train_inds]
        X_val, _, _ = f.normalize_features(X_full[val_inds, :], m, s)
        Y_val = Y_full_reg[val_inds]
        model = SVR(C=C, epsilon=epsilon, kernel="rbf")
        model.fit(X_train, Y_train)

        pred = model.predict(X_val)
        _, e = eval_reg(pred, Y_val, labels_all.iloc[val_inds,:])
        scores[i,:] = e
    
    print_pretty_table(scores, description)
    

if __name__ == '__main__':
    main()


    
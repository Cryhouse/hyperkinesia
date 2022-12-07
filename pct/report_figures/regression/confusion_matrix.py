

import random
import time

import numpy as np
import scipy.linalg as la
from scipy.stats import pearsonr as p
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix

import pct.pipeline.feature_ext as f
import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline
import pct.pipeline.binary_classifier as bc
import pct.pipeline.regressor as reg
from pct.eval.pretty_table import print_pretty_table
from pct.eval.eval_regression import eval_reg


def main():
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels_all = parsing.parse_all_labels()
    filename_df = pipeline.get_filename_df(labels_all, raw_data_path, force=True)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_all)
    
    X_full = pipeline.get_X_full(filename_df, labels_all, Y_full_per, force=True)
    # change this line to test against per's labels instead
    Y = Y_full_per
    epsilon = 0.05
    C = 5

    model = SVR(C=C, epsilon=epsilon, kernel="rbf")
    model.fit(X_full, Y_full_reg)
    N = 500
    
    pred_t = model.predict(X_full)

    description, test_eval = eval_reg(Y_full_per, pred_t, labels_all)
    scores = np.zeros((N, len(test_eval)))
    # train svr
    a_s = np.zeros((N,))
    im = np.zeros((4,4))
    for i in range(N):
        train_inds, val_inds = pipeline.train_val_patient_split2(labels_all)
        X_train, m, s = f.normalize_features(X_full[train_inds, :])
        Y_train = Y[train_inds]
        X_val, _, _ = f.normalize_features(X_full[val_inds, :], m, s)
        Y_val = Y[val_inds]
        model = SVR(C=C, epsilon=epsilon, kernel="rbf")
        model.fit(X_train, Y[train_inds])

        pred = model.predict(X_val)
        _, e = eval_reg(pred, Y_val, labels_all.iloc[val_inds,:])
        scores[i,:] = e
        c = np.round(pred)
        c[c<0] = 0
        a = np.sum(c == Y_val)/ len(pred)
        a_s[i] = a
        im_loc = confusion_matrix(Y_val, c)
        # if im_loc.shape[0] != 8: continue
        if im_loc.shape[0] != 4:
            im_loc = np.append(im_loc, np.zeros((3,1)), axis=1)
            im_loc = np.append(im_loc, np.zeros((1,4)), axis=0)
        assert im_loc.shape == (4,4)
        im += im_loc
        a += a
    
    a_tot = np.mean(a_s)
    im_tot = 100 * im / np.sum(im[:,:])
    print(f"mean accuracy: {a_tot}")
    print(np.round(im_tot))
    
    print_pretty_table(scores, description)

    # eriks results
    conf = np.array(
        [
            [3, 143, 5, 0],
            [0, 150, 70, 13],
            [0, 28, 87, 29],
            [0, 3, 57, 39]
        ]
    )
    n_corr = 0
    for i in range(4):
        n_corr += conf[i,i]
    print(f"eric acc: {n_corr/np.sum(conf[:,:])}")
    print(np.round(conf*100 / np.sum(conf[:,:])))

if __name__ == '__main__':
    main()


    
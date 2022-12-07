

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



def get_final_model(test=False):
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels = parsing.parse_labels_minus_test() if test else parsing.parse_all_labels()
    t = "train_val" if test else "all"
    filename_df = pipeline.get_filename_df(labels, raw_data_path, force=True, t=t)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels)
    X_full = pipeline.get_X_full(filename_df, labels, Y_full_reg, force=True, t="regressor")
    X_full_norm, m, s = f.normalize_features(X_full)
    # train svr
    epsilon = 0.05
    C = 2
    model = SVR(C=C, epsilon=epsilon, kernel="rbf")
    model.fit(X_full_norm, Y_full_reg)
    return model, m, s


def get_final_model2():
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels = parsing.parse_all_labels()
    train_inds, val_inds = pipeline.train_val_patient_split2(labels)

    labels_train = labels.iloc[train_inds,:].reset_index(drop=True)
    labels_val = labels.iloc[val_inds,:].reset_index(drop=True)
    filename_df = pipeline.get_filename_df(labels, raw_data_path, force=False)
    
    Y_train_class, Y_train_reg, Y_train_per, Y_train_jon = pipeline.get_Ys(labels_train)
    X_train = pipeline.get_X_full(filename_df, labels_train, Y_train_reg, force=True, t="regressor")
    X_train_norm, m, s = f.normalize_features(X_train)

    C = 2
    epsilon = 0.05

    model = SVR(C=C, epsilon=epsilon, kernel="rbf")
    model.fit(X_train_norm, Y_train_reg)
    return model, m, s, train_inds, val_inds, filename_df, labels


def main():
    pass
    

if __name__ == '__main__':
    main()


    
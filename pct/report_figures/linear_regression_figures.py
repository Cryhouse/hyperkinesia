import random

import numpy as np
import scipy.linalg as la
from scipy.stats import pearsonr as p
import matplotlib.pyplot as plt
from prettytable import PrettyTable

import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline
from pct.config import RAW_DATA_DIR, LABELS_PATH, TRAIN_PATIENTS, TEST_PATIENTS
from pct.eval.pretty_table import print_pretty_table



def train_linear_regressor(X_full, Y_full_reg, train_inds, val_inds):
    # train linear regressor
    train_inds = train_inds.astype("int")
    val_inds = val_inds.astype("int")
    X_train = X_full[train_inds, :]
    mean_train = np.mean(X_train, axis = 0)
    std_train = np.std(X_train, axis=0)

    Z_train = (X_train - mean_train)/std_train

    Y_mean = np.mean(Y_full_reg[train_inds])

    Y_hat_full = Y_full_reg - Y_mean

    Y_hat_train = Y_hat_full[train_inds]
    w = la.solve(Z_train.T.dot(Z_train), Z_train.T.dot(Y_hat_train))

    train_preds = Z_train.dot(w) + Y_mean

    X_val = X_full[val_inds,:]
    Z_val = (X_val - mean_train)/ std_train

    val_preds = Z_val.dot(w) + Y_mean

    Z_full = (X_full - mean_train)/std_train
    all_preds = Z_full.dot(w) + Y_mean
    # train errors and validation errors
    train_loss = np.mean((Y_full_reg[train_inds] - train_preds)**2)
    val_loss = np.mean((Y_full_reg[val_inds] - val_preds)**2)
    loss_all = np.mean((Y_full_reg - all_preds)**2)

    p_train, _ = p(Y_full_reg[train_inds], train_preds)
    p_val, _ = p(Y_full_reg[val_inds], val_preds)
    

    p_all, _ = p(Y_full_reg, all_preds)


    mae_val = np.mean(abs(Y_full_reg[val_inds] - val_preds))
    max_abs_error_val = np.max(abs(Y_full_reg[val_inds] - val_preds))

    description = ["Train MSE", "Validation MSE", r"$\rho$ train", r"$\rho$ val", "MAE", "Max AE"]
    return description, np.array((train_loss, val_loss, p_train, p_val, mae_val, max_abs_error_val))
    # return correlation, loss, mean absolute error (val) and max absolute error (val)


def main():
    labels_minus_test = parsing.parse_labels_minus_test()
    
    filename_df = pipeline.get_filename_df(labels_minus_test, RAW_DATA_DIR, force=False)

    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_minus_test)
    X_full = pipeline.get_X_full(filename_df, labels_minus_test, Y_full_reg, force=True)

    train_inds, val_inds = pipeline.train_val_patient_split2(labels_minus_test)
    description, test_evaluation = train_linear_regressor(X_full, Y_full_reg, train_inds, val_inds)
    n = len(test_evaluation)
    N = 500
    scores = np.zeros((N, n))
    for i in range(N):
        train_inds, val_inds = pipeline.train_val_patient_split2(labels_minus_test)
       
        _, evaluation = train_linear_regressor(X_full, Y_full_reg, train_inds, val_inds)
        scores[i, :] = evaluation

    print("linear model")
    print_pretty_table(scores, description)


if __name__ == '__main__':
    main()


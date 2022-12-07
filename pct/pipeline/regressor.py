import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.stats import pearsonr as p
from sklearn.svm import SVR

import pct.pipeline.feature_ext as f
# import pct.pipeline.smote as smote


def predict_regressor(df, svr, m, s):
    X = np.expand_dims(f.extract_all_features(df), axis=0)
    X, _, _ = f.normalize_features(X, m, s)
    pred = svr.predict(X)
    return pred


def train_reg(X_full, Y_full):
    # trains and evaluates least squares model
    N = X_full.shape[0]
    n_train = int(N* 3 // 4)
    
    # make sure the train data is 50% positive and 50% negative
    train_inds = random.sample(range(N), n_train)
    val_inds = list(set(range(N)) - set(train_inds))

    X_train, m_train, s_train = f.normalize_features(X_full[train_inds,:])
    X_full_norm, _, _ = f.normalize_features(X_full, m_train, s_train)
    X_val, _, _ = f.normalize_features(X_full[val_inds,:], m_train, s_train)

    Y_train = Y_full[train_inds]
    Y_val = Y_full[val_inds]
    
    w = la.solve(X_train.T.dot(X_train), X_train.T.dot(Y_train))
    
    preds = X_full_norm.dot(w)
    train_preds = preds[train_inds]
    val_preds = preds[val_inds]

     # correlation
    train_corr, _ = p(Y_train,train_preds)
    val_corr,_ = p(Y_val, val_preds)
    tot_corr, _ = p(Y_full, preds)

    return w, m_train, s_train, tot_corr, train_corr, val_corr, train_inds, val_inds


def train_SVR(X_full, Y_full, train_inds, val_inds, epsilon=0.2, C=1):
    # trains
    train_inds = train_inds.astype(int)
    val_inds = val_inds.astype(int)
    X_train, m_train, s_train = f.normalize_features(X_full[train_inds,:])
    svr = SVR(C=C, epsilon=epsilon, kernel="rbf")
    
    X_full_norm, _, _ = f.normalize_features(X_full, m_train, s_train)
    X_val, _, _ = f.normalize_features(X_full[val_inds,:], m_train, s_train)


    Y_train = Y_full[train_inds]
    Y_val = Y_full[val_inds]
    
    # X_smote, Y_smote = smote.smote(X_train,Y_train)
    # X_smote_norm, m_train, s_train = f.normalize_features(X_smote)
    svr.fit(X_train, Y_train)

    preds = svr.predict(X_full_norm)
    
    train_preds = preds[train_inds]
    val_preds = preds[val_inds]

    nonzero_mask = Y_full > 0
    train_mask = np.zeros_like(Y_full, dtype=bool)
    val_mask = np.zeros_like(Y_full, dtype=bool)
    train_mask[train_inds] = True
    val_mask[val_inds] = True
    train_mask_nonzero = (train_mask) & (nonzero_mask)
    val_mask_nonzero = (val_mask) & (nonzero_mask)
    
    #MSE
    train_mse = la.norm(train_preds - Y_train)
    val_mse = la.norm(val_preds - Y_val)

    # Pearson correlation all samples
    train_corr_all, _ = p(Y_train, train_preds)
    val_corr_all,_ = p(Y_val, val_preds)
    tot_corr_all, _ = p(Y_full, preds)

    assert(len(np.unique(Y_val)) > 1, "All Y_val are the same")
    assert(len(np.unique(Y_train)) > 1, "All Y_train are the same")
    assert(len(np.unique(train_preds)) > 1, "All train preds are the same")
    assert(len(np.unique(val_preds)) > 1, "All val preds are the same")

    # Pearson correlation nonzero samples
    tot_corr_nonzero, _ = p(Y_full[nonzero_mask], preds[nonzero_mask])
    train_corr_nonzero, _ = p(Y_full[train_mask_nonzero], preds[train_mask_nonzero])
    val_corr_nonzero, _ = p(Y_full[val_mask_nonzero], preds[val_mask_nonzero])

    # mean absolute error
    train_mae = np.mean(abs(train_preds - Y_train))
    val_mae = np.mean(abs(val_preds - Y_val))
    # Worst case absolute error
    train_worst = np.max(abs(train_preds - Y_train))
    val_worst = np.max(abs(val_preds - Y_val))
    description = ["train mse", "val mse", "train correlation all", "val correlation all", "train correlaion nonzero", "val correlation nonzero", "train MAE", "val MAE", "train worst absolute error", "val worst absolute error"]
    return description, np.array([train_mse, val_mse, train_corr_all, val_corr_all, train_corr_nonzero, val_corr_nonzero, train_mae, val_mae, train_worst, val_worst])

def get_best_SVR(X_full, Y_full, plot=True):
    """Trains binary classifier 1000 times with different test/val splits. Since there is a normalizing step, where we normalize by mean and standard deviation, we need to return these values as well

    Args:
        X_full (np.array): Input features. Shape: (N_samples, N_features)
        Y_full (np.array): Shape: (N_samples,)
        plot (bool, optional): Defaults to True.

    Returns:
        best_model (svr model), best_m, best_s, train_inds, val_inds
    """
    best_model = -1
    best_val_corr = -1
    best_loss = 9999999
    best_train_inds = -1
    best_val_inds = -1

    nonzero_mask = Y_full > 0

    val_corrs_all = []
    train_corrs_all = []
    tot_corrs_all = []

    val_corrs_nonzero = []
    train_corrs_nonzero = []
    tot_corrs_nonzero = []

    for i in range(1000):
        svr, m_train, s_train, tot_corr_all, train_corr_all, val_corr_all, tot_corr_nonzero, train_corr_nonzero, val_corr_nonzero, train_inds, val_inds = train_SVR(X_full, Y_full)
        tot_corrs_all.append(tot_corr_all)
        train_corrs_all.append(train_corr_all)
        val_corrs_all.append(val_corr_all)

        tot_corrs_nonzero.append(tot_corr_nonzero)
        train_corrs_nonzero.append(train_corr_nonzero)
        val_corrs_nonzero.append(val_corr_nonzero)

        if val_corr_all > best_val_corr:
            best_val_corr = val_corr_all
            best_model = svr
            best_m = m_train
            best_s = s_train
            best_train_inds = train_inds
            best_val_inds = val_inds
    
    if plot:
        fig, ax = plt.subplots(2,3)
        fig.suptitle("Regressor predictions correlation with ground truth")
        # all
        ax[0,0].hist(tot_corrs_all)
        ax[0,0].set_title("tot corrs, all")
        ax[0,1].hist(train_corrs_all)
        ax[0,1].set_title("train corrs, all")
        ax[0,2].hist(val_corrs_all)
        ax[0,2].set_title("val corrs, all")
        
        # nonzero
        ax[1,0].hist(tot_corrs_nonzero)
        ax[1,0].set_title("tot corrs, nonzero")
        ax[1,1].hist(train_corrs_nonzero)
        ax[1,1].set_title("train corrs, nonzero")
        ax[1,2].hist(val_corrs_nonzero)
        ax[1,2].set_title("val corrs, nonzero")
        plt.show()

   
    return best_model, best_m, best_s, best_val_corr, best_train_inds, best_val_inds


def get_best_w(X_full, Y_full, plot=True):
    """Trains binary classifier 1000 times with different test/val splits. Since there is a normalizing step, where we normalize by mean and standard deviation, we need to return these values as well

    Args:
        X_full (np.array): Input features. Shape: (N_samples, N_features)
        Y_full (np.array): Shape: (N_samples,)
        plot (bool, optional): Defaults to True.

    Returns:
        w (linear regression weights), best_m, best_s, train_inds, val_inds
    """
    best_w = -1
    best_val_corr = -1
    best_loss = 9999999
    best_train_inds = -1
    best_val_inds = -1

    val_corrs = []
    train_corrs = []
    tot_corrs = []
    for i in range(1000):
        w, m_train, s_train, tot_corr, train_corr, val_corr, train_inds, val_inds = train_reg(X_full, Y_full)
        tot_corrs.append(tot_corr)
        train_corrs.append(train_corr)
        val_corrs.append(val_corr)

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_w = w
            best_m = m_train
            best_s = s_train
            best_train_inds = train_inds
            best_val_inds = val_inds
    
    if plot:
        fig, ax = plt.subplots(1,3)
        fig.suptitle("Hello")
        ax[0].hist(tot_corrs)
        ax[0].set_title("tot corrs")
        ax[1].hist(train_corrs)
        ax[1].set_title("train corrs")
        ax[2].hist(val_corrs)
        ax[2].set_title("val corrs")
        plt.show()

   
    return best_w, best_m, best_s, best_val_corr, best_train_inds, best_val_inds
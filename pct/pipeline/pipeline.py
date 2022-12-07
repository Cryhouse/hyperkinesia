import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as p
import pickle

from pct.preprocessing.get_orientation import get_orientation_naive, grav2angles
import pct.preprocessing.parsing as parsing
import pct.pipeline.binary_classifier as bc
import pct.pipeline.regressor as reg
import pct.pipeline.feature_ext as f

from pct.config import RAW_DATA_DIR, LABELS_PATH, TRAIN_PATIENTS, TEST_PATIENTS

def onpress(event):
    if event.key == "Q": exit()


def get_filename_df(labels, raw_data_path, t="all", force=False):
    buffer_names = {
        "all": "filename_df_all.pickle",
        "test": "filename_df_test.pickle",
        "train_val": "filename_df_train_val.pickle",
    }
    assert t in buffer_names.keys()
    buffer = buffer_names[t]
    if buffer not in os.listdir(os.getcwd()) or force:
        print(f"Preprocessing input data...")
        # read data
        files = labels["Filename"]
        files = files[files.notna()]
        files = files.unique()
        files = [file for file in files if type(file) != int]

        filename_df = {}

        for file in files:
            full_filename = os.path.join(raw_data_path, file)
            data = parsing.read_raw(full_filename)
            accs_raw = np.array([
                data["accX"],
                data["accY"],
                data["accZ"],
            ]).reshape((3,-1))
            grav = get_orientation_naive(accs_raw)
            lin_acc = accs_raw - grav
            data["linaccX"] = lin_acc[0,:]
            data["linaccY"] = lin_acc[1,:]
            data["linaccZ"] = lin_acc[2,:]
            angles = grav2angles(grav)
            data["theta"] = angles[0,:]
            data["phi"] = angles[1,:]
            data = data.iloc[2000:-2000] # truncate away first and last 20 seconds
            filename_df[file] = data

        with open(buffer, 'wb') as fp:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(filename_df, fp, pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Using cached input data...")
        with open(buffer, 'rb') as fp:
        # Pickle the 'data' dictionary using the highest protocol available.
            filename_df = pickle.load(fp)
    return filename_df

def get_filename_df_trunc(labels, raw_data_path, force=False):
    buffer = "filename_df_trunc.pickle"
    if buffer not in os.listdir(os.getcwd()) or force:
        # read data
        files = labels["Filename"]
        files = files[files.notna()]
        files = files.unique()
        files = [file for file in files if type(file) != int]

        filename_df = {}

        for file in files:
            labels_this_file = labels[labels["Filename"] == file]
            label = labels_this_file.iloc[0]
            full_filename = os.path.join(raw_data_path, file)
            data = parsing.read_raw(full_filename)
            start_sample = int((label["start_seconds"] - label["medo start"])*100 + 100)
            end_sample = int((label["end_seconds"] - label["medo start"])*100 - 100)
            if end_sample < start_sample or start_sample < 0:
                print(f"warning")
                continue
            data = data.iloc[start_sample:end_sample,:]
            accs_raw = np.array([
                data["accX"],
                data["accY"],
                data["accZ"],
            ]).reshape((3,-1))
            grav = get_orientation_naive(accs_raw)
            lin_acc = accs_raw - grav
            data["linaccX"] = lin_acc[0,:]
            data["linaccY"] = lin_acc[1,:]
            data["linaccZ"] = lin_acc[2,:]
            angles = grav2angles(grav)
            data["theta"] = angles[0,:]
            data["phi"] = angles[1,:]
            
            # data = data.iloc[start_sconds] # truncate away first and last 20 seconds
            filename_df[file] = data

        with open(buffer, 'wb') as fp:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(filename_df, fp, pickle.HIGHEST_PROTOCOL)
    else:
        with open(buffer, 'rb') as fp:
        # Pickle the 'data' dictionary using the highest protocol available.
            filename_df = pickle.load(fp)
    return filename_df



def get_X_full(filename_df, labels, Y_full, t="regressor", force=False):
    assert t in ["regressor", "bc"], "Argument 't' needs to be either 'regressor' or 'bc'"
    feat_ext_func = f.extract_regressor_features if t == "regressor" else f.extract_bc_features
    
    filename = 'X_full_regressor.pickle' if t == "regressor" else 'X_full_binary.pickle'
    if filename in os.listdir(os.getcwd()) and not force:
        print("Using cached features...")
        with open(filename, 'rb') as fp:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            X_full = pickle.load(fp)
            
    else:
        print("Extracting features...")
        test_feats = feat_ext_func(next(iter(filename_df.values())))
        print(f"test_feats: {type(test_feats)}")
        if type(test_feats) != list and type(test_feats) != np.ndarray:
            n_feats = 1
        else:
            n_feats = len(test_feats)
        print(f"n_feats: {n_feats}")
        X_full = np.zeros((len(Y_full), n_feats))
        for i, (index, label) in enumerate(labels.iterrows()):
            df = filename_df[label["Filename"]]
            X_full[i,:] = feat_ext_func(df)

        with open(filename, 'wb') as fp:
        # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(X_full, fp, pickle.HIGHEST_PROTOCOL)
    
    assert(sum(np.isnan(X_full[:,:])) == 0, "X_full contains nan")
    return X_full


def train_val_patient_split(labels):
    # random sampling
    # choose 3/4 of the patients as train and 1/4 as test
    all_pats = labels["patient"].unique()
    random.shuffle(all_pats)
    train_pats = set(all_pats[:int(len(all_pats)*3//4)])
    val_pats = set(all_pats) - train_pats
    
    train_mask = np.zeros((len(labels),))
    val_mask = np.zeros((len(labels),))

    for train_pat in train_pats:
        train_mask += labels["patient"] == train_pat

    for val_pat in val_pats:
        val_mask += labels["patient"] == val_pat


    train_inds = np.where(train_mask)[0]
    val_inds = np.where(val_mask)[0]

    return train_inds, val_inds


def train_val_patient_split2(labels):
    # stratified sampling
    all_pats = labels["patient"].unique()
    random.shuffle(all_pats)

    pats_avg_label = []
    for pat in all_pats:
        labels_this_pat = labels[labels["patient"] == pat]
        avg = np.sum((labels_this_pat["per"] + labels_this_pat["jonathan"]) / 2)
        pats_avg_label.append([pat, avg])

    pats_avg_label = np.asarray(sorted(pats_avg_label, key=lambda x: x[1], reverse=True))

    pats_high = pats_avg_label[:int(len(pats_avg_label)//2), 0]

    pats_high = set(pats_high)
    pats_low = set(all_pats) - pats_high

    pats_high = list(pats_high)
    pats_low = list(pats_low)

    random.shuffle(pats_high)
    random.shuffle(pats_low)

    n_train_high = int(len(pats_high) *3 // 4)
    n_train_low = int(len(pats_low) *3 // 4)
    train_pats = pats_high[:n_train_high] + pats_low[:n_train_low]
    val_pats = pats_high[n_train_high:] + pats_low[n_train_low:]
    
    train_mask = np.zeros((len(labels),), dtype=bool)
    val_mask = np.zeros((len(labels),), dtype=bool)

    for train_pat in train_pats:
        train_mask += labels["patient"] == train_pat

    for val_pat in val_pats:
        val_mask += labels["patient"] == val_pat


    train_inds = np.where(train_mask)[0]
    val_inds = np.where(val_mask)[0]
    assert(len(set(train_inds).intersection(set(val_inds))) == 0)

    return train_inds, val_inds


def misclassification_histogram(preds, Y, labels, title):
    num_correct = np.sum(preds == Y)
    num_incorrect = np.sum(preds != Y)
    print(f"num_correct: {num_correct}")
    print(f"num_incorrect: {num_incorrect}")
    
    false_negative_mask = (preds==0) & (Y > 0)
    false_negatives = labels[false_negative_mask]

    fig, ax = plt.subplots()

    u = sorted(np.unique(false_negatives))
    l = []
   
    for item in u:
        l.append(np.sum(false_negatives==item))
    print(f"u: {u}")
    print(f"l: {l}")
    ax.bar([str(i) for i in u], l,align='center', width=0.3)
    fig.suptitle(title)
    plt.show()

def get_Ys(labels):
    n = len(labels)
    Y_full_reg = np.zeros((n,))
    Y_full_class = np.zeros((n,))
    Y_full_per = np.zeros((n,))
    Y_full_jon = np.zeros((n,))
    for i, (index, row) in enumerate(labels.iterrows()):
        Y_full_reg[i] = (row["jonathan"] + row["per"])/2
        Y_full_class[i] = int((row["jonathan"] + row["per"]) > 0)
        Y_full_per[i] = row["per"]
        Y_full_jon[i] = row["jonathan"]
    return Y_full_class, Y_full_reg, Y_full_per, Y_full_jon



def main():
    

    labels_minus_test = parsing.parse_labels_minus_test()
    filename_df = get_filename_df(labels_minus_test, RAW_DATA_DIR, force=True)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = get_Ys(labels_minus_test)
    X_full = get_X_full(filename_df, labels_minus_test, Y_full_class, force=False)
    svm, best_m, best_s, train_inds, val_inds = bc.get_best_model(X_full, Y_full_class, plot=True)
    exit()
    X_full_norm, _, _ = f.normalize_features(X_full, best_m, best_s)

    binary_preds = svm.predict(X_full_norm)
    #misclassification_histogram(preds[train_inds], Y_full_class[train_inds], Y_full_reg[train_inds], "train")
    #misclassification_histogram(preds[val_inds], Y_full_class[val_inds], Y_full_reg[val_inds], "val")

    # Now to the regressor
    nonzero_mask = Y_full_reg > 0
    Y_full_nonzero = Y_full_reg[nonzero_mask]
    X_full_nonzero = X_full[nonzero_mask,:]
    best_svr, best_m, best_s, best_val_corr, best_train_inds, best_val_inds = reg.get_best_SVR(X_full_nonzero, Y_full_nonzero, plot=True)
    


if __name__ == "__main__":
    main()
from os.path import join as j

from numpy import genfromtxt
import pandas as pd
import numpy as np

import pct.preprocessing.get_orientation as go
from pct.preprocessing.get_orientation import normalize
from pct.config import RAW_DATA_DIR, LABELS_PATH, TRAIN_PATIENTS, TEST_PATIENTS


def parse_train_test_patients():
    with open(TEST_PATIENTS) as f:
        test_pats = []
        for item in f.read().split(", "):
            if len(item) > 0: test_pats.append(int(item))
    with open(TRAIN_PATIENTS) as f:
        train_pats = []
        for item in f.read().split(", "):
            if len(item) > 0: train_pats.append(int(item))
    return train_pats, test_pats


def parse_all_labels():
    labels = pd.read_csv(LABELS_PATH, sep=";")
    labels = labels[(labels["jonathan"].notna()) & (labels["per"].notna()) & (labels["Filename"].notna())]
    print(f"Number of labels available in total: {len(labels)}")
    print(f"Number of patients: {len(labels['patient'].unique())}")
    labels.reset_index(drop=True, inplace=True)

    print(f"Number of labels with samples longer than a minute available in total: {len(labels)}")
    print(f"Number of patients: {len(labels['patient'].unique())}")
    return labels

def parse_labels_minus_test():
    labels = pd.read_csv(LABELS_PATH, sep=";")
    labels = labels[(labels["jonathan"].notna()) & (labels["per"].notna()) & (labels["Filename"].notna())]
    print(f"Number of labels available in total: {len(labels)}")
    print(f"Number of patients: {len(labels['patient'].unique())}")
    train_pats, test_pats = parse_train_test_patients()

    train_pats_mask = np.zeros((len(labels),), dtype=bool)
    for pat in train_pats:
        train_pats_mask += labels["patient"] == pat

    labels_minus_test = labels[train_pats_mask]
    labels_minus_test.reset_index(drop=True, inplace=True)
    return labels_minus_test

def parse_test_labels():
    labels = pd.read_csv(LABELS_PATH, sep=";")
    labels = labels[(labels["jonathan"].notna()) & (labels["per"].notna()) & (labels["Filename"].notna())]
    print(f"Number of labels available in total: {len(labels)}")
    print(f"Number of patients: {len(labels['patient'].unique())}")
    train_pats, test_pats = parse_train_test_patients()

    test_pats_mask = np.zeros((len(labels),), dtype=bool)
    for pat in test_pats:
        test_pats_mask += labels["patient"] == pat

    labels_test = labels[test_pats_mask]
    labels_test.reset_index(drop=True, inplace=True)
    return labels_test

def read_raw(path, **kwargs):
    # parse data collected with tremor12 (iOS app)
    # skips rows starting with % and returns dataframe
    with open(path) as f:
        num_skip = 0
        for l in f.readlines():
            if l.startswith("%"):
                num_skip += 1
            else:
                break
    out = pd.read_csv(path, skiprows=num_skip, **kwargs, delimiter=";")
    out.rename(columns = {
        "acc_x": "accX", 
        "acc_y": "accY", 
        "acc_z": "accZ", 
        "gyro_x": "rotX", 
        "gyro_y": "rotY", 
        "gyro_z": "rotZ"}, inplace=True)
    return out

def read_apple(path):
    return pd.read_csv(path)


def main():
    # test the different functions
    pass

if __name__ == "__main__":
    main()
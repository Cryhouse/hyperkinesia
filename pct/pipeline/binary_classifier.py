import numpy as np
import random
import sklearn.svm as svm
import matplotlib.pyplot as plt

import pct.pipeline.feature_ext as f



def predict_bin(df, model, m, s):
    """Makes binary classification prediction

    Args:
        df (pandas.DataFrame): Pandas dataframe with raw data
        model (sklearn.svm): Trained SVM binary classification model
        m (np.array): Normalizing mean
        s (np.array): Normalizing std

    Returns:
        _type_: _description_
    """
    X = np.expand_dims(f.extract_all_features(df), axis=0)
    X, _, _ = f.normalize_features(X, m, s)
    pred = model.predict(X)
    return pred


def train_bin(X_full, Y_full, C=1):
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
    
    model = svm.SVC(C=C, kernel="rbf")

    loss = model.fit(X_train, Y_train)
    
    total_pred = model.predict(X_full_norm)
    num_correct_tot = np.sum(total_pred == Y_full)
    total_accuracy = num_correct_tot / len(Y_full)

    train_pred = model.predict(X_train)
    num_correct_train = np.sum(train_pred == Y_train)
    train_accuracy = num_correct_train / len(Y_train)
    
    val_pred = model.predict(X_val)
    num_correct_val = np.sum(val_pred == Y_val)
    val_accuracy = num_correct_val / len(Y_val)
    
    n_train_pos = np.sum(Y_train)
    n_train_neg = np.sum(Y_train == 0)
    n_val_pos = np.sum(Y_val)
    n_val_neg = np.sum(Y_val == 0)
    return model, m_train, s_train, total_accuracy, train_accuracy, val_accuracy, n_train_pos, n_train_neg, n_val_pos, n_val_neg, train_inds, val_inds


def train_bin2(X_full, Y_full, train_inds, val_inds, C=1):
    train_inds = train_inds.astype(int)
    val_inds = val_inds.astype(int)

    X_train, m_train, s_train = f.normalize_features(X_full[train_inds,:])

    X_full_norm, _, _ = f.normalize_features(X_full, m_train, s_train)
    X_val, _, _ = f.normalize_features(X_full[val_inds,:], m_train, s_train)


    
    Y_train = Y_full[train_inds]
    Y_val = Y_full[val_inds]
    # print(f"X_train contains nan: {np.sum((np.isnan(X_train.reshape((-1)))))}")
    # print(f"Y_train contains nan: {np.sum(np.isnan(Y_train))}")

    model = svm.SVC(C=C, kernel="rbf")

    loss = model.fit(X_train, Y_train)
    
    total_pred = model.predict(X_full_norm)
    num_correct_tot = np.sum(total_pred == Y_full)
    total_accuracy = num_correct_tot / len(Y_full)

    train_pred = model.predict(X_train)
    num_correct_train = np.sum(train_pred == Y_train)
    train_accuracy = num_correct_train / len(Y_train)
    train_TP = np.sum((train_pred == 1) & (Y_train == 1))
    train_precision = train_TP / sum(train_pred)
    train_recall = train_TP / sum(Y_train)
    
    val_pred = model.predict(X_val)
    num_correct_val = np.sum(val_pred == Y_val)
    val_accuracy = num_correct_val / len(Y_val)
    val_TP = np.sum((val_pred == 1) & (Y_val == 1))
    val_precision = val_TP / sum(val_pred)
    val_recall = val_TP / sum(Y_val)

    val_f1 = 2*val_recall*val_precision / (val_precision + val_recall)

    description = ["train accuracy", "val accuracy", "train precision", "val precision", "train recall", "val recall", "val f1"]
    return description, np.array([train_accuracy, val_accuracy, train_precision, val_precision, train_recall, val_recall, val_f1])



def get_best_model(X_full, Y_full, plot=True, N=1000):
    """Trains binary classifier 1000 times with different test/val splits. Since there is a normalizing step, where we normalize by mean and standard deviation, we need to return these values as well

    Args:
        X_full (np.array): Input features. Shape: (N_samples, N_features)
        Y_full (np.array): Shape: (N_samples,)
        plot (bool, optional): Defaults to True.
        N (int): Number of train/val splits. Defaults to 1000

    Returns:
        best_model (svm model), best_m, best_s, train_inds, val_inds
    """
    best_model = -1
    best_acc = -1
    best_acc_val = -1

    best_n_train_pos = 0
    best_n_train_neg = 0
    best_n_val_pos = 0
    best_n_val_neg = 0

    best_train_inds = -1
    best_val_inds = -1

    total_accs = []
    train_accs = []
    val_accs = []
    for i in range(N):
        model, m_train, s_train, total_accuracy, train_accuracy, val_accuracy, n_train_pos, n_train_neg, n_val_pos, n_val_neg, train_inds, val_inds = train_bin(X_full, Y_full)
        val_accs.append(val_accuracy)
        train_accs.append(train_accuracy)
        total_accs.append(total_accuracy)
        if val_accuracy > best_acc:
            best_acc = total_accuracy
            best_acc_val = val_accuracy
            best_model = model
            best_m = m_train
            best_s = s_train
            best_n_train_pos = n_train_pos
            best_n_train_neg = n_train_neg
            best_n_val_pos = n_val_pos
            best_n_val_neg = n_val_neg
            best_train_inds = train_inds
            best_val_inds = val_inds
            
    if plot:
        fig, ax = plt.subplots(1,3)
        fig.suptitle("Binary classifier accuracy for different test/ val splits")
        ax[0].hist(total_accs)
        ax[0].set_title("All data")
        ax[1].hist(train_accs)
        ax[1].set_title("Train data")
        ax[2].hist(val_accs)
        ax[2].set_title("Validation data")
        plt.show()

    print(f"mean train_acc: {np.mean(train_accs)}")
    print(f"mean val_acc: {np.mean(val_accs)}")
    print(f"best acc: {best_acc}")
    print(f"best acc val: {best_acc_val}")
    print(f"best_n_train_pos: {best_n_train_pos}")
    print(f"best_n_train_neg: {n_train_neg}")
    print(f"best_n_val_pos: {best_n_val_pos}")
    print(f"best_n_val_neg: {best_n_val_neg}")
    return best_model, best_m, best_s, best_train_inds, best_val_inds
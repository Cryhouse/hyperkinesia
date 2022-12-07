import matplotlib.pyplot as plt
import pct.preprocessing.parsing as parsing
import pandas as pd
import numpy as np


def onpress(event):
    if event.key == "Q": exit()

def last_non_zero(arr, i, j):
    loc = arr[:,i,j]
    m = loc != -1
    if sum(m) == 0: # all are minus 1
        return 0
    else:
        out =  sorted(list(np.where(m)[0]), reverse=True)[0]
        print(out)
        return out

def split_to_subsamples(sample, sub_length, stride):
    sample.reset_index(drop=True, inplace=True)
    length = 0
    if len(sample.shape) == 1:
        length = len(sample)
    else:
        length = sample.shape[1]
    if length < sub_length:
        return []
    n_sub = ((length - sub_length) // stride) + 1
    out = []
    start = 0
    if len(sample.shape) == 1:
        for _ in range(n_sub):
            out.append(sample.loc[start:start+sub_length])
            start += stride
    else:
        for _ in range(n_sub):
            out.append(sample.loc[:,start:start+sub_length])
            start += stride
    return out

def subsample_variance_histogram(df, ori_dict, subsample_length, stride):
    # for each sample, extract the variance of a a signal and make a histogram over the distribution
    act_labels = parsing.get_label_dict()
    # interesting_labels = [4, 5, 7, 8] # sitting, standing, stand_to_sit, sit_to_stand
    interesting_labels = [4, 5] # sitting, standing
    num_features = 1
    class_variances_np = -np.ones((100000, len(interesting_labels), num_features)) # num samples, num interesting labels, num features
    for i, label in enumerate(interesting_labels):
        df_trunc = df[df["activityID"] == label]
        for index, label_row in df_trunc.iterrows():
            filename_suffix = parsing.get_filename_suffix(label_row.to_list())

            
            full_sample = ori_dict[filename_suffix]
            # get the right sample
            sample_trunc = full_sample[0, label_row["start"]:label_row["end"]]
            
            subsamples = split_to_subsamples(sample_trunc, subsample_length, stride)
            
            fill = last_non_zero(class_variances_np, i, 0) + 1
            for j, subsample in enumerate(subsamples):
                class_variances_np[fill + j, i, 0] = np.var(subsample)
    
    fig, ax = plt.subplots(len(interesting_labels), num_features)
    fig.suptitle("variance histogram")
    fig.canvas.mpl_connect("key_press_event", onpress)
    for i, label in enumerate(interesting_labels):
        if num_features == 1:
            arr_loc = class_variances_np[:,i,0]
            arr_loc = arr_loc[arr_loc != -1]
            ax[i].hist(arr_loc, bins=50, range=(0,2))
            ax[i].set_title(f"{act_labels[label]}, n_samples = {len(arr_loc)}")
        else:
            for j in range(num_features):
                arr_loc = class_variances_np[:,i,j]
                ax[i,j].hist(arr_loc[arr_loc != -1])
                ax[i,j].set_title(act_labels[label])
    plt.show()


def subsample_mean_histogram(df, ori_dict, subsample_length, stride):
    # for each sample, extract the variance of a a signal and make a histogram over the distribution
    act_labels = parsing.get_label_dict()
    # interesting_labels = [4, 5, 7, 8] # sitting, standing, stand_to_sit, sit_to_stand
    interesting_labels = [4, 5] # sitting, standing
    num_features = 1
    class_variances_np = -np.ones((100000, len(interesting_labels), num_features)) # num samples, num interesting labels, num features
    for i, label in enumerate(interesting_labels): # i is class enum
        df_trunc = df[df["activityID"] == label]
        for index, label_row in df_trunc.iterrows():
            filename_suffix = parsing.get_filename_suffix(label_row.to_list())

            
            full_sample = ori_dict[filename_suffix]
            # get the right sample
            sample_trunc = full_sample[0, label_row["start"]:label_row["end"]]
            
            subsamples = split_to_subsamples(sample_trunc, subsample_length, stride)
            
            fill = last_non_zero(class_variances_np, i, 0) + 1
            for j, subsample in enumerate(subsamples):
                assert np.sum(np.isnan(np.mean(subsample))) == 0
                class_variances_np[fill + j, i, 0] = np.mean(subsample)
    
    fig, ax = plt.subplots(len(interesting_labels), num_features)
    fig.suptitle("mean histogram")
    fig.canvas.mpl_connect("key_press_event", onpress)
    for i, label in enumerate(interesting_labels):
        if num_features == 1:
            
            arr_loc = class_variances_np[:,i,0]
            arr_loc = arr_loc[arr_loc != -1]
            ax[i].hist(arr_loc, bins=50)
            ax[i].set_title(f"{act_labels[label]}, n_samples = {len(arr_loc)}")
        else:
            for j in range(num_features):
                arr_loc = class_variances_np[:,i,j]
                ax[i,j].hist(arr_loc[arr_loc != -1])
                ax[i,j].set_title(act_labels[label])
    plt.show()

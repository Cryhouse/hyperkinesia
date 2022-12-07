
import numpy as np

import pct.preprocessing.subsample_feature_analysis as sfh

# some util functions
def normalize_features(X, m=None, s=None):
    if m is None:
        m = np.expand_dims(np.mean(X, axis=0), axis=0)
        s = np.std(X, axis=0)
    out = X - m
    # s[s==0] = 0.0001
    out = out / s
    if m is None:
        assert abs(np.max(np.mean(out, axis=0)) - m < 1e-10)
        assert max(abs(s - np.std(out, axis=0))) < 1e-10
    return out, m, s

def prettify_pred(pred):
    pred[pred > 3] = 3
    pred[pred < 0] = 0
    return pred


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



def extract_regressor_features(df):
    return feature_map4(df["theta"])


def extract_bc_features(df):
    return np.concatenate([
        feature_map4(df["theta"]),
        feature_map3(df["linaccX"]),
        feature_map3(df["linaccY"]),
        feature_map3(df["linaccZ"]),
    ])


def feature_map1(sample):
    T = 0.1
    L = 200
    S = 100
    subsamples = sfh.split_to_subsamples(sample, L, S)
    variances = np.asarray([np.var(subsample) for subsample in subsamples])
    c = np.sum(variances < 0.1) / len(variances)
    s = np.std(variances)
    m = np.mean(variances)
    return np.array([
        c,
        s,
        m,
    ])


def feature_map2(sample):
    sample = sample[~sample.isna()]
    subsamples = sfh.split_to_subsamples(sample, 200, 100) # sample, subsample_length, stride
    
    variances = np.asarray([np.var(subsample) for subsample in subsamples])
    assert np.sum(np.isnan(variances)) == 0
    # 2 seconds
    c = np.sum(variances < 1) / len(variances)
    s = np.std(variances)
    m = np.mean(variances)
    
    # 4 seconds
    subsamples2 = sfh.split_to_subsamples(sample, 400, 200)
    variances2 = np.asarray([np.var(subsample) for subsample in subsamples2])
    c2 = np.sum(variances2 < 3) / len(variances2)
    s2 = np.std(variances2)
    m2 = np.mean(variances2)
    
    # 1 second
    subsamples3 = sfh.split_to_subsamples(sample, 100, 50)
    variances3 = np.asarray([np.var(subsample) for subsample in subsamples3])
    c3 = np.sum(variances3 < 0.1) / len(variances3)
    s3 = np.std(variances3)
    m3 = np.mean(variances3)
    
    out = np.array([
        c,
        s,
        m,

        c2,
        s2,
        m2,

        c3,
        s3,
        m3,
    ])
    assert np.sum(np.isnan(out)) == 0
    
    return out


def feature_map3(sample):
    nc = [10,20,30]
    out = np.zeros((3,))
    for i, n in enumerate(nc):
        out[i] = np.sum(np.abs(moving_average(sample, int(n))))
    return out


def feature_map4(sample):
    T1 = 1
    T2 = 0.3
    L = 200
    S = 100
    subsamples = sfh.split_to_subsamples(sample, L, S)
    variances = np.asarray([np.var(subsample) for subsample in subsamples])
    c = np.sum(variances < T1) / len(variances)
    c2 = np.sum(variances < T2) / len(variances)
    return np.array([
        c,
        c2,
    ])


# experimental

def longest_consecutive_above_threshold(sample, threshold):
    mask = sample > threshold
    last_true = False
    current = 0
    cons = []
    for item in mask:
        # three cases: 
        # 1. last was true and this sample is true
        # 2. last was false this sample true
        # 3. this is false
        if item:
            current += 1
            last_true = True

        else:
            if last_true:
                last_true = False
                cons.append(current)
                current = 0
    return max(cons) if len(cons) > 0 else 0

def longest_consecutive_below_threshold(sample, threshold):
    mask = sample < threshold
    last_true = False
    current = 0
    cons = []
    for item in mask:
        # three cases: 
        # 1. last was true and this sample is true
        # 2. last was false this sample true
        # 3. this is false
        if item:
            current += 1
            last_true = True

        else:
            if last_true:
                last_true = False
                cons.append(current)
            current = 0
    return max(cons) if len(cons) > 0 else 0
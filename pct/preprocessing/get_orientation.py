import numpy as np
import scipy.signal as s


def get_orientation_naive(acc):
    # non causal moving average! benefit of hindsight
    N = acc.shape[1]
    num = 20
    
    C = list(np.ones((1,num)).squeeze()/ num)
    A = [1] # No autoregression here
    
    ma = s.filtfilt(C,A, acc, axis=1)
    return ma


def grav2angles(grav, degrees=True):
    grav = normalize(grav)
    out = np.asarray(list(map(lambda x: [np.arccos(x[1]), np.arctan2(x[0],x[2])], grav.T))).T
    if degrees:
        return out * 180 / np.pi
    else:
        return out


def normalize(data):
    assert data.shape[0] == 3
    return np.array([
        list(map(lambda x: x/sum(x**2)**0.5, data.T))
    ]).T.squeeze()
import matplotlib.pyplot as plt
import numpy as np

import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline
from pct.config import RAW_DATA_DIR, LABELS_PATH, TRAIN_PATIENTS, TEST_PATIENTS
from pct.eval.pretty_table import print_pretty_table

def naive_model(Y_full_reg, train_inds, val_inds):
    n_train = len(train_inds)

    train_inds = train_inds.astype(int)
    val_inds = train_inds.astype(int)
    naive_prediction = np.mean(Y_full_reg[train_inds])

    train_preds = np.ones((n_train))*naive_prediction
    val_preds = np.ones((len(val_inds)))*naive_prediction


    loss_train, MAE_train, max_AE_train = eval_naive(train_preds, Y_full_reg[train_inds])
    loss_val, MAE_val, max_AE_val = eval_naive(val_preds, Y_full_reg[val_inds])
    return np.array((loss_train, loss_val, MAE_val, max_AE_val))

def eval_naive(preds, Y):
    loss = np.mean((preds - Y)**2)
    MAE = np.mean(abs(preds - Y))
    max_AE = np.max(abs(preds - Y))
    return loss, MAE, max_AE


def main():
    labels_minus_test = parsing.parse_labels_minus_test()
    
    filename_df = pipeline.get_filename_df(labels_minus_test, RAW_DATA_DIR, force=False)
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_minus_test)
    train_inds, val_inds = pipeline.train_val_patient_split(labels_minus_test)
    test_evaluation_naive = naive_model(Y_full_reg, train_inds, val_inds)
    n = len(test_evaluation_naive)
    N = 500
    scores = np.zeros((N, n))
    for i in range(N):
        train_inds, val_inds = pipeline.train_val_patient_split(labels_minus_test)
        evaluation = naive_model(Y_full_reg, train_inds, val_inds)
        scores[i, :] = evaluation
    

    description_naive = ["MSE train", "MSE validataion", "MAE validataion", "Max AE val"]
    print_pretty_table(scores, description_naive)

if __name__ == '__main__':
    main()


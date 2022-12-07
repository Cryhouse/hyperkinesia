import numpy as np
import sklearn.svm as svm

import pct.pipeline.feature_ext as f
import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline

# random and stratified test set

from pct.eval.pretty_table import print_pretty_table
from pct.eval.eval_binary import eval_binary

def main():
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels_all = parsing.parse_all_labels()
    filename_df = pipeline.get_filename_df_trunc(labels_all, raw_data_path, force=True)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_all)
    
    X_full = pipeline.get_X_full(filename_df, labels_all, Y_full_reg, force=True)

    C = 2
    model = svm.SVC(C=C, kernel="rbf")
    model.fit(X_full, Y_full_class)
    N = 100
    
    pred_t = model.predict(X_full)

    description, test_eval = eval_binary(pred_t, Y_full_class, Y_full_reg)
    scores = np.zeros((N, len(test_eval)))

    for i in range(N):
        train_inds, val_inds = pipeline.train_val_patient_split(labels_all)
        X_train, m, s = f.normalize_features(X_full[train_inds, :])
        Y_train = Y_full_class[train_inds]
        X_val, _, _ = f.normalize_features(X_full[val_inds, :], m, s)
        Y_val = Y_full_class[val_inds]
        Y_val_reg = Y_full_reg[val_inds]
        model = svm.SVC(C=C, kernel="rbf")
        model.fit(X_train, Y_train)

        pred = model.predict(X_val)
        _, e = eval_binary(pred, Y_val, Y_val_reg)
        scores[i,:] = e
    
    print_pretty_table(scores, description)
    

if __name__ == '__main__':
    main()


    
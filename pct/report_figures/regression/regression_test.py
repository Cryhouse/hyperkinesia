
import numpy as np
import scipy.linalg as la
from scipy.stats import pearsonr as p
import matplotlib.pyplot as plt
from sklearn.svm import SVR

import pct.pipeline.feature_ext as f
import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline


def main():
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels_minus_test = parsing.parse_labels_minus_test()
    filename_df = pipeline.get_filename_df(labels_minus_test, raw_data_path, force=True)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_minus_test)
    
    X_full = pipeline.get_X_full(filename_df, labels_minus_test, Y_full_reg, force=True)
    X_full_norm, m, s = f.normalize_features(X_full)
    # train svr
    epsilon = 0.05
    C = 2
    model = SVR(C=C, epsilon=epsilon, kernel="rbf")
    model.fit(X_full_norm, Y_full_reg)
    


    labels_test = parsing.parse_test_labels()
    filename_df_test = pipeline.get_filename_df(labels_test, raw_data_path, force=True)
    Y_test_class, Y_test_reg, Y_test_per, Y_test_jon = pipeline.get_Ys(labels_test)
    X_test = pipeline.get_X_full(filename_df_test, labels_test, Y_test_class, force=True)
    X_test_norm, _, _ = f.normalize_features(X_test, m, s)
    
    test_pred = model.predict(X_test_norm)

    # test data evaluation
    corr_test, _ = p(test_pred, Y_test_reg)
    nonzero_mask = Y_test_reg != 0
    corr_test_nonzero, _ = p(test_pred[nonzero_mask], Y_test_reg[nonzero_mask])
    mse = np.mean((test_pred - Y_test_reg)**2)
    mse_nonzero = np.mean((test_pred[nonzero_mask] - Y_test_reg[nonzero_mask])**2)
    mae = np.mean(abs(test_pred - Y_test_reg))
    mae_nonzero = np.mean(abs(test_pred[nonzero_mask] - Y_test_reg[nonzero_mask]))
    worst = np.max(abs(test_pred - Y_test_reg))
    print(f"max prediction: {np.max(test_pred)}")
    print(f"min prediction: {np.min(test_pred)}")
    print(f"corr: {corr_test}")
    # add nonzero correlation
    print(f"corr_test_nonzero: {corr_test_nonzero}")
    print(f"mae: {mae}")
    print(f"mae_nonzero: {mae_nonzero}")
    print(f"mse: {mse}")
    print(f"mse_nonzero: {mse_nonzero}")
    print(f"worst: {worst}")

    # correlation with per and jonathan
    per = labels_test["per"]
    jonathan = labels_test["jonathan"]
    corr_per, _ = p(test_pred, per)
    corr_jonathan, _ = p(test_pred, jonathan)
    print(f"correlation per: {corr_per}")
    print(f"correlation jonathan: {corr_jonathan}")

    
    plt.hist(test_pred - Y_test_reg)
    plt.grid(True, color="#93a1a1", alpha=0.3)
    plt.title("Distribution of errors")
    plt.show()

if __name__ == '__main__':
    main()


    
import numpy as np
from scipy.stats import pearsonr as p
import matplotlib.pyplot as plt
from sklearn.svm import SVR

import pct.pipeline.feature_ext as f
import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline
from pct.eval.pretty_table import print_pretty_table
from pct.eval.eval_regression import eval_reg

# level1 regression, test data

def eval_reg(pred, Y, labels):
    corr_test, _ = p(pred, Y)
    nonzero_mask = Y != 0
    corr_test_nonzero, _ = p(pred[nonzero_mask], Y[nonzero_mask])
    mae = np.mean(abs(pred - Y))
    mae_nonzero = np.mean(abs(pred[nonzero_mask] - Y[nonzero_mask]))
    mse = np.mean((pred - Y)**2)
    mse_nonzero = np.mean((pred[nonzero_mask] - Y[nonzero_mask])**2)
    worst = np.max(abs(pred - Y))
    # add nonzero correlation
    # correlation with per and jonathan
    per = labels["per"]
    jonathan = labels["jonathan"]
    corr_per, _ = p(pred, per)
    corr_jonathan, _ = p(pred, jonathan)

    out = np.array([corr_test, corr_test_nonzero, mse, mse_nonzero, mae, mae_nonzero, worst])
    description = [r"$\rho$", r"$\rho$ nonzero", r"mse", r"mse monzero",r"mae", r"mae nonzero", r"worst"]
    return description, out

def main():
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels_minus_test = parsing.parse_labels_minus_test()
    filename_df = pipeline.get_filename_df_trunc(labels_minus_test, raw_data_path, force=True)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_minus_test)
    
    X_train = pipeline.get_X_full(filename_df, labels_minus_test, Y_full_reg, force=True)
    X_train_norm, m, s = f.normalize_features(X_train)
    # train svr
    epsilon = 0.05
    C = 2
    model = SVR(C=C, epsilon=epsilon, kernel="rbf")
    model.fit(X_train_norm, Y_full_reg)
    


    labels_test = parsing.parse_test_labels()
    filename_df_test = pipeline.get_filename_df_trunc(labels_test, raw_data_path, force=True)
    Y_test_class, Y_test_reg, Y_test_per, Y_test_jon = pipeline.get_Ys(labels_test)
    X_test = pipeline.get_X_full(filename_df_test, labels_test, Y_test_class, force=True)
    X_test_norm, _, _ = f.normalize_features(X_test, m, s)
    
    test_pred = model.predict(X_test_norm)

    description, e = eval_reg(test_pred, Y_test_reg, labels_test)
    print_pretty_table(e, description)


    plt.hist(test_pred - Y_test_reg)
    plt.grid(True, color="#93a1a1", alpha=0.3)
    plt.title("Distribution of errors")
    plt.show()

if __name__ == '__main__':
    main()


    
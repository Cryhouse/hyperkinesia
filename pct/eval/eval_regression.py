import numpy as np
from scipy.stats import pearsonr as p


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

    out = np.array([corr_test, corr_test_nonzero, corr_per, corr_jonathan, mse, mse_nonzero, mae, mae_nonzero, worst])
    description = [r"$\rho$", r"$\rho$ nonzero", r"$\rho per$", r"$\rho jonathan$", r"mse", r"mse monzero",r"mae", r"mae nonzero", r"worst"]
    return description, out

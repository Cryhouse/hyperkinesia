import numpy as np
from scipy.stats import wilcoxon as w
import matplotlib.pyplot as plt
from sklearn.svm import SVR

import pct.pipeline.feature_ext as f
import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline
from pct.eval.eval_regression import eval_reg


def get_95_pred_interval(data):
    s = sorted(data)
    n = len(s)
    ind = int(2.5 * n // 100)
    low = s[ind]
    high = s[-ind]
    return low, high


def main():
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels_all = parsing.parse_all_labels()
    filename_df = pipeline.get_filename_df(labels_all, raw_data_path, force=False)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_all)
    
    X_full = pipeline.get_X_full(filename_df, labels_all, Y_full_reg, force=False)

    epsilon = 0.05
    C = 5
    model = SVR(C=C, epsilon=epsilon, kernel="rbf")
    model.fit(X_full, Y_full_reg)
    N = 500
    
    pred_t = model.predict(X_full)

    description, test_eval = eval_reg(Y_full_reg, pred_t, labels_all)
    scores = np.zeros((N, len(test_eval)))
    

     # levodopa and comb
    time_score_levodopa = []
    time_score_komb = []

    time_score_levodopa_doc = []
    time_score_komb_doc = []

    # train svr
    for i in range(N):
        train_inds, val_inds = pipeline.train_val_patient_split2(labels_all)
        # train_inds, val_inds = pipeline.get_stratified_Kfold_split(labels_all, Y_full_reg)
        X_train, m, s = f.normalize_features(X_full[train_inds, :])
        Y_train = Y_full_reg[train_inds]
        X_val, _, _ = f.normalize_features(X_full[val_inds, :], m, s)
        Y_val = Y_full_reg[val_inds]
        model = SVR(C=C, epsilon=epsilon, kernel="rbf")
        model.fit(X_train, Y_train)

        pred = model.predict(X_val)
        for j, ind in enumerate(val_inds):
            label = labels_all.iloc[ind,:]
            komb = label["komb"]
            if komb not in [0, 1]: continue
            pred_score = pred[j]
            minutes = int(label["minutes"])
            if label["komb"] == 0:
                time_score_levodopa.append([minutes, pred_score])
                time_score_levodopa_doc.append([minutes, (label["per"] + label["jonathan"])/2])
            else:
                time_score_komb.append([minutes, pred_score])
                time_score_komb_doc.append([minutes, (label["per"] + label["jonathan"])/2])
        _, e = eval_reg(pred, Y_val, labels_all.iloc[val_inds,:])
        scores[i,:] = e
    

    # cdrs predictions
    l_sorted = np.asarray(sorted(time_score_levodopa, key=lambda x: x[0], reverse=False))
    k_sorted = np.asarray(sorted(time_score_komb, key=lambda x: x[0], reverse=False))
    l_sorted_doc = np.asarray(sorted(time_score_levodopa_doc, key=lambda x: x[0], reverse=False))
    k_sorted_doc = np.asarray(sorted(time_score_komb_doc, key=lambda x: x[0], reverse=False))
    m = np.unique(l_sorted[:,0])

    l_means = []
    l_highs = []
    l_lows = []
    k_means = []
    k_highs = []
    k_lows = []

    l_means_doc = []
    l_doc_lows = []
    l_doc_highs = []
    k_means_doc = []
    k_doc_lows = []
    k_doc_highs = []

    for i, m_loc in enumerate(m):
        l_mask = l_sorted[:,0] == m_loc
        k_mask = k_sorted[:,0] == m_loc

        mean_l = np.mean(l_sorted[l_mask, 1])
        l_means.append(mean_l)
        l_low, l_high = get_95_pred_interval(l_sorted[l_mask, 1])
        l_lows.append(l_low)
        l_highs.append(l_high)

        mean_k = np.mean(k_sorted[k_mask,1])
        k_means.append(mean_k)
        k_low, k_high = get_95_pred_interval(k_sorted[k_mask, 1])
        k_lows.append(k_low)
        k_highs.append(k_high)

        l_mask_doc = l_sorted_doc[:,0] == m_loc
        k_mask_doc = k_sorted_doc[:,0] == m_loc

        mean_l_doc = np.mean(l_sorted_doc[l_mask_doc, 1])
        l_means_doc.append(mean_l_doc)
        l_doc_low, l_doc_high = get_95_pred_interval(l_sorted_doc[l_mask_doc, 1])
        l_doc_lows.append(l_doc_low)
        l_doc_highs.append(l_doc_high)

        mean_k_doc = np.mean(k_sorted_doc[k_mask_doc,1])
        k_means_doc.append(mean_k_doc)
        k_doc_low, k_doc_high = get_95_pred_interval(k_sorted_doc[k_mask_doc, 1])
        k_doc_lows.append(k_doc_low)
        k_doc_highs.append(k_doc_high)
        

    print(w(k_means_doc, l_means_doc))


    fig, ax = plt.subplots()
    ax.plot(m, l_means, label="Levodopa (algorithm)", color="blue")
    ax.plot(m, k_means, label="Komb (algorithm)", color="red")
    ax.plot(m, l_means_doc, label="Levodopa (doctor)", color="lightblue")
    ax.plot(m, k_means_doc, label="Komb (doctor)", color="orange")
    
    # prediction intervals
    # ax.plot(m, l_highs, color="blue", linestyle="dashed")
    # ax.plot(m, l_lows, color="blue", linestyle="dashed")

    # ax.plot(m, k_highs, color="red", linestyle="dashed")
    # ax.plot(m, k_lows, color="red", linestyle="dashed")

    # ax.plot(m, l_doc_highs, color="lightblue", linestyle="dashed")
    # ax.plot(m, l_doc_lows, color="lightblue", linestyle="dashed")

    # ax.plot(m, k_doc_highs, color="orange", linestyle="dashed")
    # ax.plot(m, k_doc_lows, color="orange", linestyle="dashed")


    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle("Torso CDRS")
    plt.show()

if __name__ == '__main__':
    main()


    
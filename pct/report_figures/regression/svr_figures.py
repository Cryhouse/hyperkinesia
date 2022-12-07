
import time

import numpy as np
from scipy.stats import pearsonr as p
import matplotlib.pyplot as plt

import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline
import pct.pipeline.regressor as reg
 

def grid_search(X_full, Y_full_reg, labels):
    num_es = 20 # number of Cs to try
    epsilons = np.linspace(0, 1, num_es)
    train_inds, val_inds = pipeline.train_val_patient_split2(labels)
    description, eval = reg.train_SVR(X_full, Y_full_reg, train_inds, val_inds)

    tic = time.time()
    N = 100
    all_scores = np.zeros((num_es, N, len(eval)))
    for e_num, epsilon in enumerate(epsilons):
        for i in range(N):
            train_inds, val_inds = pipeline.train_val_patient_split2(labels)
            
            description, eval = reg.train_SVR(X_full, Y_full_reg, train_inds, val_inds, epsilon=epsilon, C=2)
            _, eval = reg.train_SVR(X_full, Y_full_reg, train_inds, val_inds)
            all_scores[e_num, i, :] = eval
    toc = time.time()
    N_trains = num_es*N
    print(f"average training time: {(toc - tic)/N_trains}")
    return epsilons, description, all_scores


def plot_regressor(X_full, Y_full_reg, labels):

    epsilons, description, all_scores = grid_search(X_full, Y_full_reg, labels)

    fig, ax = plt.subplots()
    ax.scatter(epsilons, np.mean(all_scores[:, :, 3], axis=1), label=description[3])
    ax.scatter(epsilons, np.mean(all_scores[:, :, 5], axis=1), label=description[5])
    ax.scatter(epsilons, np.mean(all_scores[:, :, 7], axis=1), label=description[7])
    ax.scatter(epsilons, np.mean(all_scores[:, :, 9], axis=1), label=description[9])
    ax.scatter(epsilons, (np.mean(all_scores[:, :, 7], axis=1) + np.mean(all_scores[:, :, 9], axis=1))/2, label="mean of worst absolute error and MAE")
    ax.set_xlabel(r"$\epsilon$")
    # ax.set_ylim([0,1])
    ax.grid(True, color="#93a1a1", alpha=0.3)
    ax.set_title(r"SVR evaluation, feature map 2 applied to $\theta$ alone, feature map 3 applied to linear acceleration")
    ax.legend()
    # ax.set_title(r"SVC evaluation for different C, feature map 1 applied to $\varphi$")
    plt.show()

def main():
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels_minus_test = parsing.parse_labels_minus_test()
    filename_df = pipeline.get_filename_df(labels_minus_test, raw_data_path, force=False)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_minus_test)
    X_full = pipeline.get_X_full(filename_df, labels_minus_test, Y_full_reg, force=True)

    plot_regressor(X_full, Y_full_reg, labels_minus_test)
    

if __name__ == '__main__':
    main()


    
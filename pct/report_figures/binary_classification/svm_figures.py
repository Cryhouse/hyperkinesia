import time

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline
import pct.pipeline.binary_classifier as bc


def grid_search(X_full, Y_full_class):
    num_cs = 20 # number of Cs to try
    Cs = np.linspace(0.1, 5, num_cs)
    k = 5 # number of folds
    k_inds = pipeline.get_stratified_Kfold_split(k, Y_full_class)
    train_inds = k_inds[:, 0]
    val_inds = k_inds[:,1:].reshape((-1,))
    description, eval = bc.train_bin2(X_full, Y_full_class, train_inds, val_inds)

    tic = time.time()
    N = 10 # number of K-folds run
    all_scores = np.zeros((num_cs, N*k, len(eval)))
    for c_num, C in enumerate(Cs):
        for i in range(N):
            k_inds = pipeline.get_stratified_Kfold_split(k, Y_full_class)
            for j in range(k):
                train_inds = k_inds[:,j]
                if j < (k - 1):
                    val_inds = np.concatenate((k_inds[:,:j].reshape(-1), k_inds[:,j+1:].reshape(-1)), axis=0)
                else:
                    val_inds = k_inds[:,:j].reshape((-1,))
            
                _, eval = bc.train_bin2(X_full, Y_full_class, train_inds, val_inds, C=C)
                all_scores[c_num, i*k + j, :] = eval
    toc = time.time()
    N_trains = len(Cs)*N*k
    print(f"average training time: {(toc - tic)/N_trains}")
    return Cs, description, all_scores


def grid_search2(X_full, Y_full_class, labels):
    num_cs = 20 # number of Cs to try
    Cs = np.linspace(0.1, 5, num_cs)
    
    train_inds_test, val_inds_test = pipeline.train_val_patient_split2(labels)
    description, eval_test = bc.train_bin2(X_full, Y_full_class, train_inds_test, val_inds_test)

    tic = time.time()
    N = 500
    all_scores = np.zeros((num_cs, N, len(eval_test)))
    for c_num, C in enumerate(Cs):
        for i in range(N):
            train_inds, val_inds = pipeline.train_val_patient_split(labels)
            _, eval = bc.train_bin2(X_full, Y_full_class, train_inds, val_inds, C=C)
            all_scores[c_num, i, :] = eval
    toc = time.time()
    N_trains = len(Cs)*N
    print(f"average training time: {(toc - tic)/N_trains}")
    return Cs, description, all_scores


def print_pretty_table(scores, description):
    raw_table = np.zeros((4, len(description)))
    max_f1 = np.max(scores[:,-1])
    max_f1_ind = np.where(scores[:, -1] == max_f1)[0]
    for i in range(len(description)):
        # if i == len(description) - 1:
        #     raw_table[0, i] = format(scores[max_f1_ind,i], ".3f")
        #     raw_table[1, i] = format(scores[])
        raw_table[0,i] = format(np.mean(scores[:,i]), ".3f")
        raw_table[1,i] = format(np.std(scores[:,i]), ".3f")
        raw_table[2,i] = format(np.max(scores[:,i]), ".3f")
        raw_table[3,i] = format(np.min(scores[:,i]), ".3f")
    description.insert(0, "metric")
    pretty_table = PrettyTable(description)
    pretty_table.hrules = 2
    pretty_table.vrules = 2

    desc = ["mean", "std", "max", "min"]
    rs = [list(raw_table[i,:]) for i in range(raw_table.shape[0])]
    for i, r in enumerate(rs):
        r.insert(0, desc[i])
        pretty_table.add_row(r)
    # r1 = list(raw_table[0,:])
    # r1.insert(0, "Mean")
    # r2 = list(raw_table[1,:])
    # r2.insert(0, "std")

    # pretty_table.add_row(r1)
    # pretty_table.add_row(r2)
    print(pretty_table)


def plot_binary_predictor(X_full, Y_full_class, labels):

    Cs, description, all_scores = grid_search2(X_full, Y_full_class, labels)
    print_pretty_table(all_scores[7, :, :], description) # index 7 corresponds to C = 2

    # plot the results
    # all_scores =  np.zeros((num_cs, N*k, len(eval)))
    fig, ax = plt.subplots()
    # for i, a in enumerate(ax.reshape((-1))):
        # we want to plot the mean and and standard deviation of the different metrics as a function of C. first plot val accuracy, second spot on third axis.
    ax.scatter(Cs, np.mean(all_scores[:, :, 1], axis=1), label="validation accuracy")
    ax.scatter(Cs, np.mean(all_scores[:, :, 3], axis=1), label="validation precision")
    ax.scatter(Cs, np.mean(all_scores[:, :, 5], axis=1), label="validation recall")
    ax.scatter(Cs, np.mean(all_scores[:, :, 6], axis=1), label="validation F1 score")
    ax.set_xlabel("C")
    ax.set_ylim([0,1])
    ax.grid(True, color="#93a1a1", alpha=0.3)
    ax.legend()
    ax.set_title(r"SVC evaluation for different C, feature map 2 applied to $\theta$ alone, feature map 3 applied to linear acceleration")
    plt.show()

def main():
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels_minus_test = parsing.parse_labels_minus_test()
    filename_df = pipeline.get_filename_df(labels_minus_test, raw_data_path, force=False)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_minus_test)
    X_full = pipeline.get_X_full(filename_df, labels_minus_test, Y_full_class, force=True)

    plot_binary_predictor(X_full, Y_full_class, labels_minus_test)
    
    #
    # 
    # k = 5
    # k_inds = pipeline.get_Kfold_split(k, Y_full_reg)
    # train_inds = k_inds[:, 0]
    # val_inds = k_inds[:,1:].reshape((-1,))

    # print(val_inds)
    # description, eval = bc.train_bin2(X_full, Y_full_class, train_inds, val_inds)
    
    # N = 100
    # scores = np.zeros((N*k, len(eval)))
    # for i in range(N):
    #     k_inds = pipeline.get_Kfold_split(k, Y_full_reg)
    #     for j in range(k):
    #         train_inds = k_inds[:,j]

    #         if j < (k - 1):
    #             val_inds = np.concatenate((k_inds[:,:j].reshape(-1), k_inds[:,j+1:].reshape(-1)), axis=0)
    #         else:
    #             val_inds = k_inds[:,:j].reshape((-1,))
        
    #         _, eval = bc.train_bin2(X_full, Y_full_class, train_inds, val_inds)
    #         scores[i*k + j, :] = eval
    
    # # plot the scores
    # fig, ax = plt.subplots(3,2)
    # for i, a in enumerate(ax.reshape((-1,))):
    #     a.grid(True, color="#93a1a1", alpha=0.3)
    #     a.hist(scores[:,i])
    #     a.set_title(description[i])
    # plt.show()

    # svm, best_m, best_s, train_inds, val_inds = bc.get_best_model(X_full, Y_full_class, plot=True)

    # X_full_norm, _, _ = f.normalize_features(X_full, best_m, best_s)

    # binary_preds = svm.predict(X_full_norm)
    #misclassification_histogram(preds[train_inds], Y_full_class[train_inds], Y_full_reg[train_inds], "train")
    #misclassification_histogram(preds[val_inds], Y_full_class[val_inds], Y_full_reg[val_inds], "val")

    # Now to the regressor
    # nonzero_mask = Y_full_reg > 0
    # Y_full_nonzero = Y_full_reg[nonzero_mask]
    # X_full_nonzero = X_full[nonzero_mask,:]
    # best_svr, best_m, best_s, best_val_corr, best_train_inds, best_val_inds = reg.get_best_SVR(X_full_nonzero, Y_full_nonzero, plot=True)
    


if __name__ == '__main__':
    main()


    
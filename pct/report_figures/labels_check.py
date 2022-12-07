
import pandas as pd
import numpy as np
import pct.preprocessing.parsing as parsing

from scipy.stats import pearsonr as p

def get_corrs(labels):
    per = labels["per"]
    jonathan = labels["jonathan"]
    
    corr, _ = p(per, jonathan)


    # nonzero corr
    nonzero_mask = (per != 0) & (jonathan != 0)
    corr_nonzero, _ = p(per[nonzero_mask], jonathan[nonzero_mask])
    return corr, corr_nonzero

def main():
    labels = parsing.parse_all_labels()
    corr_all, corr_all_nonzero = get_corrs(labels)

    labels_train = parsing.parse_labels_minus_test()
    corr_train, corr_train_nonzero = get_corrs(labels_train)

    labels_test = parsing.parse_test_labels()
    corr_test, corr_test_nonzero = get_corrs(labels_test)

    def f(n):
        return format(n, ".3f")
    print(f"corr all: {f(corr_all)}, corr_all_nonzero: {f(corr_all_nonzero)}")
    print(f"corr_train: {f(corr_train)}, corr_train_nonzero: {f(corr_train_nonzero)}")
    print(f"corr_test: {f(corr_test)}, corr_test_nonzero: {f(corr_test_nonzero)}")

    def mean_difference(labels, title):
        mean_abs_diff = np.mean(abs(labels["per"] - labels["jonathan"]))
        print(f"mean_abs_diff {title}: {mean_abs_diff}")
        nonzero_mask = (labels["per"] + labels["jonathan"]) != 0
        labels_nonzero = labels[nonzero_mask]
        mean_abs_diff_nonzero = np.mean(abs(labels_nonzero["per"] - labels_nonzero["jonathan"]))
        print(f"mean_abs_diff_nonzero {title}: {mean_abs_diff_nonzero}")
    
    mean_difference(labels, "all")
    mean_difference(labels_train, "train")
    mean_difference(labels_test, "test")

if __name__ == '__main__':
    main()


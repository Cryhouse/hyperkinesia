import random

import numpy as np
import scipy.linalg as la
from scipy.stats import pearsonr as p
import matplotlib.pyplot as plt

import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline




def main():
    labels = parsing.parse_all_labels()
    fig, ax = plt.subplots(1,2)
    fig.suptitle("Label distribution")
    per = [sum(labels["per"] == l) for l in labels["per"].unique()]
    jonathan = [sum(labels["jonathan"] == l) for l in labels["jonathan"].unique()]
    average = (labels["per"] + labels["jonathan"])/2
    average= [sum(average == l) for l in average.unique()]
    for a in ax:
        a.grid(True, color="#93a1a1", alpha=0.3)
    ax[0].bar([0, 1, 2, 3], per)
    ax[0].set_title("Per")
    ax[1].bar([0,1,2,3], jonathan)
    ax[1].set_title("Jonathan")
    plt.show()


    plt.bar([0,0.5,1,1.5,2,2.5,3], average, width=0.4, align="center")
    plt.grid(True, color="#93a1a1", alpha=0.3)
    plt.title("Average label distribution")
    plt.xticks([0,0.5,1,1.5,2,2.5,3])
    plt.show()


    # labels_minus_test = parsing.parse_labels_minus_test()
    # labels_test = parsing.parse_test_labels()

    # labels_minus_test["mean"] = (labels_minus_test["per"] + labels_minus_test["jonathan"]) / 2
    # labels_test["mean"] = (labels_test["per"] + labels_test["jonathan"]) / 2
    # fig, ax = plt.subplots(1,2)
    # def bar(m):
    #     u = sorted(np.unique(m))
    #     l = []
    #     for item in u:
    #         l.append(np.sum(m == item))
    #     return u, l
    # x, y = bar(labels_minus_test["mean"])
    # ax[0].bar(x, y, width=0.25)
    # ax[0].grid(True, color="#93a1a1", alpha=0.3)
    # ax[0].set_title("Train data label distribution")
    # x, y = bar(labels_test["mean"])
    # ax[1].bar(x, y, width=0.25)
    # ax[1].grid(True, color="#93a1a1", alpha=0.3)
    # ax[1].set_title("Test data label distribution")
    # plt.show()

if __name__ == '__main__':
    main()


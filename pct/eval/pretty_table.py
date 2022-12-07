from prettytable import PrettyTable
import numpy as np


def print_pretty_table(scores, description):
    if len(scores.shape) > 1:
        pretty_multirow(scores, description)
    else:
        pretty_singlerow(scores, description)


def pretty_singlerow(scores, description):
    pretty_table = PrettyTable(description)
    pretty_table.hrules = 2
    pretty_table.vrules = 2
    pretty_table.add_row(scores)
    print(pretty_table)


def pretty_multirow(scores, description):
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
    print(pretty_table)
import matplotlib.pyplot as plt
import sklearn.metrics as m
from prettytable import PrettyTable

import pct.preprocessing.parsing as parsing


def main():
    labels = parsing.parse_all_labels()
    res = m.confusion_matrix(labels["per"], labels["jonathan"])
    x = PrettyTable(res.dtype.names)
    for row in res:
        x.add_row(row)
    print(f"confusion matrix:\n{x.get_string(border=False)}")
    plt.imshow(res)
    ticks = [0, 1, 2, 3]
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlabel("Jonathan torso CDRS")
    plt.ylabel("Per torso CDRS")
    plt.show()


if __name__ == '__main__':
    main()


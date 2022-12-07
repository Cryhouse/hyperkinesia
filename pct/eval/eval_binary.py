import numpy as np
import matplotlib.pyplot as plt

# Code for evaluating a binary predictor
def eval_binary(pred, Y_class, Y_reg, plot=False):
    true_pos = np.sum((pred == 1) & (Y_class==1))
    false_pos = np.sum((pred == 1) & (Y_class == 0))
    true_ned = np.sum((pred == 0) & (Y_class == 0))
    false_neg = np.sum((pred == 0) & (Y_class == 1))

    accuracy = np.sum(pred == Y_class) / len(Y_class)
    # precision tp / (tp + fp)
    # recall tp / (tp + fn)
    precision = true_pos / np.sum(pred)
    recall = true_pos / np.sum(Y_class)
    f1_score = 2* precision *recall / (precision + recall)

    description = ["accuracy", "precision", "recall", "f1_score"]
    out = [accuracy, precision, recall, f1_score]
    # how does the distribution look for the misclassifications?
    false_negatives = Y_reg[(pred == 0) & (Y_class == 1)]
    
    false_positives = Y_reg[(pred == 1) & (Y_class == 0)]

    u_n = sorted(np.unique(false_negatives))
    l_n = []
    if plot:
        fig, ax = plt.subplots()
        for item in u_n:
            l_n.append(np.sum(false_negatives==item))

        
        ax.bar([str(i) for i in u_n], l_n,align='center', width=0.3)
        ax.set_title("Label distribution for false negatives")
        ax.grid(True, color="#93a1a1", alpha=0.3)
        plt.show()
    return description, out
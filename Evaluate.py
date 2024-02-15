from Classifiers import *


def get_accuracy(preds, labels):
    accuracy = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            accuracy += 1
    return accuracy/len(preds)

def build_confusion_matrix(preds, labels):
    confusion_matrix = np.zeros(15,15)
    for pred, label in zip(preds, labels):
        confusion_matrix[label, pred] += 1
    return confusion_matrix

def print_confusion_matrix(matrix):
    for i in range(15):
        for j in range(15):
            if j != 14:
                print("{x} &".format(x=matrix[i,j]), end = "")
            else:
                print("{x} & {total} \\ \hline".format(x=matrix[i,j], total = sum(matrix[i, :])))

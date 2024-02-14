from Classifiers import *


def get_accuracy(preds, labels):
    accuracy = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            accuracy += 1
    return accuracy/len(preds)
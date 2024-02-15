from sklearn import svm
import numpy as np

class OneVRestSVM():
    def __init__(self):
        self.classifiers = []
        for i in range(15):
            self.classifiers.append(svm.SVC(probability=True))
    
    def fit(self, data, labels):
        for i in range(15):
            labels_masked = np.array([1 if label==i else 0 for label in labels])
            self.classifiers[i].fit(data, labels_masked)

    def predict(self, data):
        raw_preds = np.zeros((len(data), 15))
        for j in range(15):
            probs = None
            probs = self.classifiers[j].predict_proba(data)
            for i in range(len(probs)):
                raw_preds[i, j] = probs[i, 1]
        return np.argmax(raw_preds, axis= 1)

class NearestNeighbourClassifier():
    def __init__(self, distance):
        self.dataset_train = None
        self.labels_train = None
        self.distance = distance

    def fit(self, data, labels):
        self.dataset_train = data
        self.labels_train = labels

    def predict(self, data):
        labels = np.full(len(data), -1)

        for i in range(len(data)):
            min_dist = 10000000
            for j in range(len(self.dataset_train)):
                dist = self.distance(self.dataset_train[j], data[i])
                if dist < min_dist:
                    min_dist = dist
                    labels[i] = self.labels_train[j]
        return labels
 
        
        

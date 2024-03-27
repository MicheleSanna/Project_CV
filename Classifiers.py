from sklearn import svm
import numpy as np
from tqdm import tqdm
from Evaluate import get_accuracy

class OneVRestSVM():
    def __init__(self, kernel = 'linear', gamma = 2.8):
        self.gamma = gamma
        self.kernel = kernel
        self.classifiers = []
        if kernel != 'linear':
            kernel = lambda x, y: kernel(x, y, gamma)
        for i in range(15):
            self.classifiers.append(svm.SVC(kernel=kernel, probability=True))
    
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
    
    def score(self, data, labels):
        preds = self.predict(data)
        return get_accuracy(preds, labels)
    
    def get_params(self, deep=True):
        return {'gamma': self.gamma}
    
    def set_params(self, gamma):
        self.classifiers = []
        if self.kernel != 'linear':
            self.kernel = lambda x, y: self.kernel(x, y, gamma)
        for i in range(15):
            self.classifiers.append(svm.SVC(kernel=self.kernel, probability=True))


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
    

def gaussian_chi_kernel(x, y):
    ct = len(x[0]) 
    a = 1
    dist_matrix = np.zeros((len(x), len(y)))
    for i in tqdm(range(len(x))):
        for j in range(len(y)):
            dist = 0
            for k in range(ct):
                dist += ((x[i][k] - y[j][k]) * (x[i][k] - y[j][k])) / (x[i][k] + y[j][k] + 0.00001) #Added a small amount for numerical stability
            kernel = (dist*0.5)*a
            dist_matrix[i, j] = np.exp(-kernel)
    return dist_matrix

def gaussian_chi_kernel_fast(x, y, gamma=2.8):
    dist_matrix = np.zeros((len(x), len(y)))
    for i in tqdm(range(len(x))):
        for j in range(len(y)):
            b = x[i] - y[j]
            np.divide(np.multiply(b, b), x[i] + y[j] + 0.00001)
            dist = np.sum(np.divide(np.multiply(b, b), x[i] + y[j] + 0.00001)) #Added a small amount for numerical stability
            dist_matrix[i, j] = np.exp(-(dist*0.5)*gamma)
    return dist_matrix
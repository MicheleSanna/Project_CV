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
            print("Trained {i} over 15".format(i=i))

    def predict(self, data):
        preds = []
        for i in range(15):
            preds.append(self.classifiers[i].predict_proba(data))
        print(preds)

 
        
        

from sklearn.model_selection import KFold
from Evaluate import *

def validate_model(model, train_dataset, train_labels, kfolds):
    splitter = KFold(n_splits = kfolds, shuffle=True)
    acc = 0
    for j, (train_index, test_index) in enumerate(splitter.split(train_dataset)):
        train_split = [train_dataset[i] for i in train_index]
        train_split_labels = [train_labels[i] for i in train_index]
        test_split = [train_dataset[i] for i in test_index]
        test_split_labels = [train_labels[i] for i in test_index]

        model.fit(train_split, train_split_labels)
        preds = model.predict(test_split)

        print(f"Accuracy at fold {j}: {get_accuracy(preds, test_split_labels)}")
        acc += get_accuracy(preds, test_split_labels)
    print("CV accuracy: ", acc/kfolds)

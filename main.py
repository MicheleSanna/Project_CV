from classifiers import * 
from Dataset import *
from Histograms import *
from EMD import * 
from test import *
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

LOAD = True
KNN = True
TEST = False
CV = False
n_clusters = 50
plt.rcParams['font.size'] = 7

print("START")
print("N CLUSTERS: ", n_clusters)

if LOAD:
    centroids = np.load("centroids_50.npy")
else:
    dataset = Dataset("dataset\\", "train")
    sift_descriptors = create_descriptor_dataset(dataset)
    kmeans = KMeans(n_clusters = n_clusters, n_init='auto')
    kmeans.fit(sift_descriptors)
    centroids = kmeans.cluster_centers_
    np.save("centroids_50", centroids)


if LOAD:
    bow_dataset_train = np.load("bow_dataset_train_50.npy")
    labels_train = np.load("labels_train_50.npy")
else:
    bow_dataset_train, labels_train = create_bag_of_words_dataset(dataset, centroids, l2_distance)
    np.save("bow_dataset_train_50", bow_dataset_train)
    np.save("labels_train_50", labels_train)

print("Bag of words train dataset: DONE")

if LOAD:
    bow_dataset_test = np.load("bow_dataset_test_50.npy")
    labels_test = np.load("labels_test_50.npy")
else:
    dataset_test = Dataset("dataset\\", "test")
    bow_dataset_test, labels_test = create_bag_of_words_dataset(dataset_test, centroids, l2_distance)
    np.save("bow_dataset_test_50", bow_dataset_test)
    np.save("labels_test_50", labels_test)

print("Bag of words test dataset: DONE")

if KNN:
    knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs = 4)
    onenn_classifier = NearestNeighbourClassifier(l2_distance)

    onenn_classifier.fit(bow_dataset_train, labels_train)
    preds = onenn_classifier.predict(bow_dataset_test)
    print("ACCURACY with 1nn: ", get_accuracy(onenn_classifier.predict(bow_dataset_test), labels_test))
    disp = ConfusionMatrixDisplay.from_predictions(labels_test, preds, normalize='true', display_labels=['Bedroom', 'Coast', 'Forest','Highway','Industrial','InsideCity','Kitchen','LivingRoom','Mountain','Office','OpenCountry','Store','Street','Suburb','TallBuilding'])
    disp.plot()
    plt.show()

one_vs_rest_SVM = OneVsRestClassifier(svm.SVC(kernel=gaussian_chi_kernel_fast, probability=True), n_jobs=4) #For fast computation


if TEST:
    one_vs_rest_SVM.fit(bow_dataset_train, labels_train)
    preds = one_vs_rest_SVM.predict(bow_dataset_test)
    print(f"ACCURACY with one vs rest SVM and gamma {2.8}:", get_accuracy(preds, labels_test))
    print(labels_test)
    disp = ConfusionMatrixDisplay.from_predictions(labels_test, preds, normalize='true', display_labels=['Bedroom', 'Coast', 'Forest','Highway','Industrial','InsideCity','Kitchen','LivingRoom','Mountain','Office','OpenCountry','Store','Street','Suburb','TallBuilding'])
    disp.plot()
    plt.show()

if CV:
    validate_model(one_vs_rest_SVM, bow_dataset_train, labels_train, 10)
    

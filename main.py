from Evaluate import *
from Dataset import *
from Centroids import *

def min_dist(centroids, descriptors):
    min_dist = 100000
    for centroid in centroids:
        for descriptor in descriptors:
            dist = np.linalg.norm(centroid - descriptor)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def print_list(l):
    for element in l:
        print(element)

def pseudomedian(descriptor):
    min_diff = 100
    center = 0
    for i in range(len(descriptor)):
        diff = abs(sum(descriptor[0:i-1]) - sum(descriptor[i:len(descriptor)-1]))
        if diff < min_diff:
            mind_diff = diff
            center = i
    return center








print("Passo 1")
dataset = Dataset("dataset\\", "train")

sift_descriptors = create_descriptor_dataset(dataset)
#best_cluster_number(sift_descriptors) #Not useful
n_clusters = 22
kmeans = KMeans(n_clusters = n_clusters, n_init='auto')
kmeans.fit(sift_descriptors)
centroids = kmeans.cluster_centers_


print("Passo 2")

bow_dataset_train, labels_train = create_bag_of_words_dataset(dataset, centroids)
print(bow_dataset_train)
print("Bag of words train dataset: DONE")
print("Length: ", len(bow_dataset_train))

dataset_test = Dataset("dataset\\", "test")

bow_dataset_test, labels_test = create_bag_of_words_dataset(dataset_test, centroids)

print("Bag of words test dataset: DONE")

onenn_classifier = NearestNeighbourClassifier()
onenn_classifier.fit(bow_dataset_train, labels_train)

print("ACCURACY with 1nn", get_accuracy(onenn_classifier.predict(bow_dataset_test), labels_test))

svm_classifier = svm.SVC(decision_function_shape='ovr')
svm_classifier.fit(bow_dataset_train, labels_train)

print("ACCURACY with SVM", get_accuracy(svm_classifier.predict(bow_dataset_test), labels_test))


one_vs_rest_SVM = OneVRestSVM()
one_vs_rest_SVM.fit(bow_dataset_train, labels_train)
print("ACCURACY with one vs rest SVM", get_accuracy(one_vs_rest_SVM.predict(bow_dataset_test), labels_test))




#for datapoint in dataset_test:
#    img, real_class = datapoint
#    predicted_class = svm_classifier.predict(img_to_histogram(img, centroids))
#    print(real_class, predicted_class)
#    if real_class == predicted_class:
#        accuracy += 1
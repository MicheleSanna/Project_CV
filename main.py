import random
from Dataset import *
from Classifiers import * 
from sklearn.cluster import KMeans


VERBOSE = False 

def silhouette_coefficient(datapoint, centroids):
    dist = np.zeros(len(centroids))

    for i in range(len(centroids)):
        dist[i] = np.linalg.norm(datapoint-centroids[i])

    dist = np.sort(dist)
    silhouette = (dist[1] - dist[0])/max(dist[0], dist[1])

    return silhouette


def silhouette_avg(datapoints, centroids):
    f = 1/len(datapoints)
    avg = 0
    for datapoint in datapoints:
        avg += silhouette_coefficient(datapoint, centroids) * f

    return avg

def class_pop(dataset):
    class_pop = np.zeros(15)

    for i in range(len(dataset)):
        if (i%100==0):
            print(i)
        img, class_id = dataset[i]
        class_pop[class_id] += 1
    return class_pop

def create_descriptor_dataset(dataset):
    keypoints = []
    descriptors = []

    for i in range(len(dataset)):
        sift = cv2.SIFT_create(50)
        if(i%250 == 0):
            print("Step ", i)
        img, class_id = dataset[i]
        kp, dscrptrs = sift.detectAndCompute(img, None)
        for keypoint, descriptor in zip(kp, dscrptrs):
            if(i%500 == 0 and i != 0 and VERBOSE):
                print("Keypoint: ", keypoint)
                print("Descriptor len: ", len(descriptor))
            keypoints.append(keypoint)
            descriptors.append(descriptor)

    return np.array(descriptors)

def best_cluster_number(sift_descriptors):
    for i in range(2, 48):
        kmeans = KMeans(n_clusters = i, n_init='auto')
        kmeans.fit(sift_descriptors)
        centroids = kmeans.cluster_centers_
        print("N_clusters: {i} |  Silhouette: {avg}".format(i=i, avg=silhouette_avg(sift_descriptors, centroids)))

def min_dist(centroids, descriptors):
    min_dist = 100000
    for centroid in centroids:
        for descriptor in escriptors:
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

def l2_normalization(vector):
    norm = np.linalg.norm(vector)
    for i in range(len(vector)):
        vector[i] = vector[i]/norm
    return vector

def l2_distance(a, b):
    return np.linalg.norm(a - b)

def centroid_vote1(centroids, descriptors, distance_fun):
    vote = 1000 #will give out of bound error if not updated
    vote_histogram = np.zeros(len(centroids))
    for descriptor in descriptors:
        min_dist = 10000000
        for i in range(len(centroids)):
            dist = distance_fun(centroids[i], descriptor)
            if dist < min_dist:
                min_dist = dist
                vote = i
        vote_histogram[vote] +=1
    return l2_normalization(vote_histogram)

def img_to_histogram(img, centroids):
    sift = cv2.SIFT_create(1000)
    kp, dscrptrs = sift.detectAndCompute(img, None)
    return centroid_vote1(centroids, dscrptrs, l2_distance)

def create_bag_of_words_dataset(dataset, centroids):
    bag_of_words_dataset = []
    labels = []
    for i in range(len(dataset)):
        img, class_id = dataset[i]
        bag_of_words_dataset.append(img_to_histogram(img, centroids))
        labels.append(class_id)
    return np.array(bag_of_words_dataset), np.array(labels) 

def nearest_neighbour_classifier(img_hist, dataset_train, labels_train):
    img_class = -1
    min_dist = 10000000

    for i in range(len(dataset_train)):
        dist = np.linalg.norm(dataset_train[i] - img_hist)
        if dist < min_dist:
            min_dist = dist
            img_class = labels_train[i]
    return img_class

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
print("Length: ", len(bow_dataset_train))

dataset_test = Dataset("dataset\\", "test")

bow_dataset_test, labels_test = create_bag_of_words_dataset(dataset_test, centroids)

accuracy = 0
for datapoint in dataset_test:
    img, real_class = datapoint
    predicted_class = nearest_neighbour_classifier(img_to_histogram(img, centroids), bow_dataset_train, labels_train)
    print(real_class, predicted_class)
    if real_class == predicted_class:
        accuracy += 1
print("ACCURACY with 1nn", accuracy/len(dataset_test))

svm_classifier = svm.SVC(decision_function_shape='ovr')
svm_classifier.fit(bow_dataset_train, labels_train)

preds = svm_classifier.predict(bow_dataset_test)

accuracy = 0

one_vs_rest_SVM = OneVRestSVM()
one_vs_rest_SVM.fit(bow_dataset_train, labels_train)
one_vs_rest_SVM.predict(bow_dataset_test)

for i in range(len(preds)):
    if preds[i] == labels_test[i]:
        accuracy += 1
print("ACCURACY with SVM", accuracy/len(preds))

#for datapoint in dataset_test:
#    img, real_class = datapoint
#    predicted_class = svm_classifier.predict(img_to_histogram(img, centroids))
#    print(real_class, predicted_class)
#    if real_class == predicted_class:
#        accuracy += 1




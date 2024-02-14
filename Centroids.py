from sklearn.cluster import KMeans
import numpy as np



def l2_normalization(vector):
    norm = np.linalg.norm(vector)
    for i in range(len(vector)):
        vector[i] = vector[i]/norm
    return vector

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

def best_cluster_number(sift_descriptors):
    for i in range(2, 48):
        kmeans = KMeans(n_clusters = i, n_init='auto')
        kmeans.fit(sift_descriptors)
        centroids = kmeans.cluster_centers_
        print("N_clusters: {i} |  Silhouette: {avg}".format(i=i, avg=silhouette_avg(sift_descriptors, centroids)))

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
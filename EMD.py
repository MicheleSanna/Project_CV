import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import cv2

class EMDCalculator():
    def __init__(self, centroids, len):
        self.distance_matrix = distance_matrix(centroids).astype("float32") #* 0.03 #scaled it in order to not have super small numbers out of the kernel
        self.flatten_distance_matrix = self.distance_matrix.flatten(order='C')
        self.A_ub = prepare_Aub(len)
        self.A_eq = np.ones((1, len*len))
        self.bounds = prepare_bounds(len)

    def kernel2(self, x, y, gamma=0.2):
        dist_matrix = np.zeros((len(x), len(y)))

        for i in tqdm(range(len(x))): 
            for j in range(len(y)):
                dist = cv2.EMD(np.array([x[i]]), np.array([y[j]]), cv2.DIST_L2, cost=self.distance_matrix)
                dist_matrix[i, j] = np.exp(-dist[2][0]*gamma)
        return dist_matrix
    
    def kernel(self, x, y, gamma=0.2):
        dist_matrix = np.zeros((len(x), len(y)))
        x =np.array(x)
        y =np.array(y)
        for i in tqdm(range(len(x))): 
            for j in range(len(y)):
                flows = linprog(self.flatten_distance_matrix, A_ub=self.A_ub, b_ub=np.concatenate([x[i],y[j]]), A_eq=self.A_eq, b_eq=np.min([np.sum(x[i]), np.sum(y[j])]), bounds=self.bounds).x
                dist = np.sum(self.flatten_distance_matrix * flows) / np.sum(flows)
                dist_matrix[i, j] = np.exp(-dist*gamma)
        return dist_matrix



def distance_matrix(x):
    dist = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            dist[i, j] = np.linalg.norm(x[i] - x[j])
    return dist

def prepare_Aub(len):
    A_ub = np.zeros((len*2, len*len))
    for i in range(len):
        for j in range(i*len, i*len+len):
            A_ub[i, j] = 1

    for i in range(len, 2*len):
        for j in range(i-len, len*len, len):
            A_ub[i, j] = 1
    return A_ub

def prepare_bounds(len):
    bounds = []
    for i in range(len*len):
        bounds.append((0, None))
    return bounds

def wasserstein_distance_kernel(x, y, centroids, gamma=0.2):
    dist_matrix = np.zeros((len(x), len(y)))
    for i in tqdm(range(len(x))):
        for j in range(len(y)):
            dist = wasserstein_distance(x[i], y[j])
            dist_matrix[i, j] = np.exp(-(dist*0.5)*gamma)
    return dist_matrix
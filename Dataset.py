import glob
import cv2
from torch.utils.data import Dataset
from Centroids import *

WINDOWS = True
VERBOSE = False 

class Dataset(Dataset):
    def __init__(self, data_dir, mode):
        if WINDOWS:
            separator = '\\'
        else:
            separator = '/'
        file_list = []
        self.data = []
        self.imgs_path = data_dir + mode + separator

        self.class_map = {"Bedroom" : 0, "Coast": 1, "Forest": 2, "Highway": 3, "Industrial": 4,
                          "InsideCity": 5, "Kitchen": 6, "LivingRoom": 7, "Mountain": 8, "Office": 9,
                          "OpenCountry": 10, "Store": 11, "Street": 12, "Suburb": 13, "TallBuilding": 14}

        if (mode == 'train' or mode == 'test'):
            folder_list = glob.glob(self.imgs_path + "*")
            for folder in folder_list:
                file_list.append(glob.glob(folder + separator + "*"))
            file_list = [item for sublist in file_list for item in sublist]


        for img_path in file_list:
            class_name = img_path.split(separator)[-2]
            self.data.append([img_path, class_name])
        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        #print(class_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        class_id = self.class_map[class_name]
        return img, class_id

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

def img_to_histogram(img, centroids):
    sift = cv2.SIFT_create(1000)
    kp, dscrptrs = sift.detectAndCompute(img, None)
    return centroid_vote1(centroids, dscrptrs, l2_distance)

def l2_distance(a, b):
    return np.linalg.norm(a - b)

def create_bag_of_words_dataset(dataset, centroids):
    bag_of_words_dataset = []
    labels = []
    for i in range(len(dataset)):
        img, class_id = dataset[i]
        bag_of_words_dataset.append(img_to_histogram(img, centroids))
        labels.append(class_id)
    return np.array(bag_of_words_dataset), np.array(labels) 

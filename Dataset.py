import glob
import cv2
from torch.utils.data import Dataset
from Histograms import *

WINDOWS = True
DEBUG = False

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
        if (DEBUG):
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
        if (i%100==0 and DEBUG):
            print(i)
        img, class_id = dataset[i]
        class_pop[class_id] += 1
    return class_pop

def create_descriptor_dataset(dataset):
    descriptors = []

    for i in range(len(dataset)):
        sift = cv2.SIFT_create(50)
        if(i%250 == 0 and DEBUG):
            print("Step ", i)
        img, _ = dataset[i]
        _ , dscrptrs = sift.detectAndCompute(img, None)
        for descriptor in dscrptrs:
            descriptors.append(descriptor)

    return np.array(descriptors)

def create_bag_of_words_dataset(dataset, centroids, distance):
    bag_of_words_dataset = np.zeros((len(dataset), len(centroids)))
    labels = np.zeros((len(dataset)))
    for i in range(len(dataset)):
        img, class_id = dataset[i]
        bag_of_words_dataset[i] = img_to_histogram(img, centroids, distance)
        labels[i] = class_id
    return np.array(bag_of_words_dataset), np.array(labels) 

def img_to_histogram(img, centroids, distance):
    sift = cv2.SIFT_create(1000)
    _ , dscrptrs = sift.detectAndCompute(img, None)
    return centroid_soft_vote2(centroids, dscrptrs, distance)
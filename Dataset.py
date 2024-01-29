import glob
import cv2
from torch.utils.data import Dataset

WINDOWS = True

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

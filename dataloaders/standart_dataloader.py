import os
import torch
from skimage import io, transform
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import transforms, utils
from tqdm import tqdm
from PIL import Image
import time
class EmotionImagesDataset(Dataset):


    def __init__(self, path_to_txt, root_dir):

        self.data_transformations = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.path_to_txt = path_to_txt
        self.root_dir = root_dir
        self.imgs_paths, self.imgs_labels = self._read_txt()


    def __len__(self):
        return len(self.imgs_labels)


    def __getitem__(self, idx):
        path_to_image = os.path.join(self.root_dir, self.imgs_paths[idx])

        X = cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)
        X = cv2.resize(X, (256, 256), interpolation=cv2.INTER_NEAREST)
        X = self.data_transformations(X)
        
        y = self.imgs_labels[idx]
        
        return X, y


    def _read_txt(self):
        img_labels = []
        img_paths = []

        with open(self.path_to_txt, 'r') as a_f:

            for line in a_f:
                current_img_path, current_img_label = line.strip().split(' ')
                img_labels.append(int(current_img_label))
                img_paths.append(current_img_path)

        return img_paths, img_labels

if __name__ == '__main__':
    data_loader = EmotionImagesDataset('/home/petryshak/ImageEmotionClassification/dataset/b-t4sa_train.txt', 
                                       '/home/petryshak/ImageEmotionClassification/dataset/')
    for i in tqdm(range(len(data_loader))):
        X, y = data_loader.__getitem__(i)

        break
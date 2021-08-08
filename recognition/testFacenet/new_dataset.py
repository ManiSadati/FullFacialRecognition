import os
import sys
sys.path.insert(1, '../../data/lfw')
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math
import itertools
from torch.autograd import Variable


class LFW_DataSet(Dataset):
    def __init__(self, data_direction, transform):
        self.data_path = data_direction
        self.transform = transform
        self.unique_labels = []
        self.image_dict = {}
        self.id_image = {}
        self.labels = []
        self.images = []
        self.load_images()
        

    def load_images(self):
        self.unique_labels = os.listdir(self.data_path)
        for i in range(len(self.unique_labels)):
            self.image_dict[self.unique_labels[i]] = []
        for i in range(len(self.unique_labels)):            
            path = os.path.join(self.data_path, self.unique_labels[i])        
            imgs = os.listdir(path)
            self.image_dict[self.unique_labels[i]] = imgs
            for imgid in imgs:
                self.id_image[imgid] = []
            for imgid in imgs:
                self.labels.append(self.unique_labels[i])
                img_path = os.path.join(path, imgid)
                self.images.append(img_path)
                # pic = Image.open(img_path)
                # pict = self.transform(pic)
                # self.id_image[imgid] = pict
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        path = self.images[idx]
        pic = Image.open(path)
        pict = self.transform(pic)
        return pict, self.labels[idx]

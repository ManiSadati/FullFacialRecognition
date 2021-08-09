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

# class BalancedBatchSampler(BatchSampler):

#     def __init__(self, labels, n_classes, n_samples):
#         self.labels = labels
#         self.labels_set = list(set(self.labels.numpy()))
#         self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
#                                  for label in self.labels_set}
#         for l in self.labels_set:
#             np.random.shuffle(self.label_to_indices[l])
#         self.used_label_indices_count = {label: 0 for label in self.labels_set}
#         self.count = 0
#         self.n_classes = n_classes
#         self.n_samples = n_samples
#         self.n_dataset = len(self.labels)
#         self.batch_size = self.n_samples * self.n_classes

#     def __iter__(self):
#         self.count = 0
#         while self.count + self.batch_size < self.n_dataset:
#             classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
#             indices = []
#             for class_ in classes:
#                 indices.extend(self.label_to_indices[class_][
#                                self.used_label_indices_count[class_]:self.used_label_indices_count[
#                                                                          class_] + self.n_samples])
#                 self.used_label_indices_count[class_] += self.n_samples
#                 if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
#                     np.random.shuffle(self.label_to_indices[class_])
#                     self.used_label_indices_count[class_] = 0
#             yield indices
#             self.count += self.n_classes * self.n_samples

#     def __len__(self):
#         return self.n_dataset // self.batch_size
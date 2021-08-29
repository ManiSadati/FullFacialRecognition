from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile

from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
import matplotlib.pyplot as plt
import pickle

from testSphereFace import *
from testSphereFace.inference_recognition import *

def add2gallery(gallery_embeddings, gallery_names, new_embeddings, new_names):
    total_embeddings = np.concatenate(gallery_embeddings, new_embeddings)
    sz = len(gallary_names)
    total_names = {}
    for n in gallery_names:
        total_names[n] = gallary_names[n]
    for n in new_names:
        total_names[n] = new_names[n] + sz
    return total_embeddings, total_names


if __name__ == "__main__":

    recognition_model, gallery_embeddings, gallery_names = load_recognition()
    total_embeddings, total_names = add2gallery(gallery_embeddings, gallary_names, new_embeddings, new_names)
    

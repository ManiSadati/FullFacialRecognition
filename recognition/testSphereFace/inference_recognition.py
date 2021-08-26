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

def alignment(src_img, name):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],[48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    src_pts = ref_pts
    if name in landmark:
        src_pts = landmark[name]

    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)
    for i, x in enumerate(src_pts):
        w = int(x[0])
        h = int(x[1])
        src_img[h-1:h+1, w-1:w+1, :] = 100
        if i == 2:
            src_img[h-1:h+1, w-1:w+1, 0] = 255

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

    
def load_image(name):
    img1 = cv2.imdecode(np.frombuffer(zfile.read(name),np.uint8),1)
    img1 = alignment(img1, name)
    img2 = img1.transpose(2, 0, 1).reshape((1,3,112,96))
    img2 = (img2-127.5)/128.0
    return img2, img1


def lfw_test_a_pair(name1, name2):
    
    f1 = torch.tensor(embeddings[index[name1]])
    f2 = torch.tensor(embeddings[index[name2]])
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    return cosdistance.item() 
    
def save_landmarks_and_embeddings():
    names = [x[0] for x in landmark.items()]
    imglist = []
    last_seen = 0
    embeddings = np.zeros((len(names), 512))
    for i in range(len(names)):
        img1 = load_image(names[i])
        imglist.append(img1)
        if (i - last_seen + 1 == 100) or (i == len(names) - 1):
            print(i)
            images = np.vstack(imglist)
            images = Variable(torch.from_numpy(images).float(),volatile=True).cuda()
            output = net(images)
            embed = output.data
            for j in range(i - last_seen + 1):
                s = embed[j]
                s = s.cpu().numpy()
                embeddings[j + last_seen,:] = s
            imglist = []
            last_seen = i + 1
    #print(embedding.items())
    with open('../../data/lfw_landmarks.pickle', 'wb') as handle:
        pickle.dump(landmark, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open('../../data/lfw_numpy_embeddings.pickle', 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL) 


def load_recognition():
    parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
    parser.add_argument('--net','-n', default='sphere20a', type=str)
    parser.add_argument('--lfw', default='../data/lfw.zip', type=str)
    parser.add_argument('--model','-m', default='../data/sphere20a_20171020.pth', type=str)
    args = parser.parse_args()

    predicts=[]
    net = getattr(net_sphere,args.net)()
    net.load_state_dict(torch.load(args.model))
    net.cuda()
    net.eval()
    net.feature = True

    zfile = zipfile.ZipFile(args.lfw)

    landmark = {}
    embeddings = {}
    index = {}
    reverse_index = {}
    cnt = 0
    with open('../data/lfw_landmark.txt') as f:
        landmark_lines = f.readlines()
    for line in landmark_lines:
        l = line.replace('\n','').split('\t')
        landmark[l[0]] = [int(k) for k in l[1:]]
        index[l[0]] = cnt
        reverse_index[cnt] = l[0]
        cnt += 1

    with open ('../data/lfw_numpy_embeddings.pickle',"rb") as pick:
        embeddings = torch.tensor(pickle.load(pick)).cuda()
    return net, embeddings, reverse_index

def face2embedding(faces, net):
    face_embeddings = torch.zeros(len(faces), 512)
    images = np.vstack(faces)
    print(type(images), images.shape)
    images = Variable(torch.from_numpy(images).float(),volatile=True).cuda()
    output = net(images)
    embed = output.data
    for i in range(len(faces)):
        face_embeddings[i] = embed[i]
    return face_embeddings

def verify_embeddings(face_embeddings, gallery_embeddings, gallary_names):
    f2norm = gallery_embeddings.norm(dim=1)
    verified_names = []
    for i in range(face_embeddings.shape[0]):
        x = torch.tensor(face_embeddings[i]).cuda()
        f1norm = x.norm()
        a1 = x.inner(gallery_embeddings.float())
        a2 = f1norm*f2norm+1e-5
        a3 = a1 / a2
        cosdistance = a3
        threshold = 0.5050
        maxim = torch.max(cosdistance).item()
        values, indices = torch.topk(cosdistance,7)
        print(values, indices)
        if(maxim > threshold):
            verified_names.append(gallary_names[indices[0].item()])
        else:
            verified_names.append("UNKNOWN")
    return verified_names

def add_to_gallary(faces, names):
    "d"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
    parser.add_argument('--net','-n', default='sphere20a', type=str)
    parser.add_argument('--lfw', default='../../data/lfw.zip', type=str)
    parser.add_argument('--model','-m', default='../../data/sphere20a_20171020.pth', type=str)
    args = parser.parse_args()

    predicts=[]
    net = getattr(net_sphere,args.net)()
    net.load_state_dict(torch.load(args.model))
    net.cuda()
    net.eval()
    net.feature = True

    zfile = zipfile.ZipFile(args.lfw)

    landmark = {}
    embeddings = {}
    index = {}
    cnt = 0
    with open('../../data/lfw_landmark.txt') as f:
        landmark_lines = f.readlines()
    for line in landmark_lines:
        l = line.replace('\n','').split('\t')
        landmark[l[0]] = [int(k) for k in l[1:]]
        index[l[0]] = cnt
        cnt += 1

    
    with open ('../../data/lfw_landmarks.pickle',"rb") as pick:
        landmark = pickle.load(pick)
    with open ('../../data/lfw_numpy_embeddings.pickle',"rb") as pick:
        embeddings = pickle.load(pick)


    print("myfunc ", lfw_test_a_pair("Ann_Veneman/Ann_Veneman_0003.jpg", "Gary_Williams/Gary_Williams_0002.jpg") )
    print("myfunc ", lfw_test_a_pair("Gary_Williams/Gary_Williams_0001.jpg", "Gary_Williams/Gary_Williams_0002.jpg"))
    print("myfunc ", lfw_test_a_pair("Mani_Sadati/Mani_Sadati_0001.jpg", "Mani_Sadati/Mani_Sadati_0003.jpg"))
    print("myfunc ", lfw_test_a_pair("Mani_Sadati/Mani_Sadati_0001.jpg", "Ann_Veneman/Ann_Veneman_0003.jpg"))
    print("-------------------------------------------------------------------------------------------------")
    img1, img2 = load_image("Mani_Sadati/Mani_Sadati_0001.jpg")
    print(img1.shape, img2.shape)
    
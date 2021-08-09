import argparse
import time
from pathlib import Path

import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression_face, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from tqdm import tqdm

#from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt


#from mss import mss
bounding_box = {'top': 0, 'left': 0, 'width': 1750, 'height': 1080}
#sct = mss()


def dynamic_resize(shape, stride=64):
    max_size = max(shape[0], shape[1])
    if max_size % stride != 0:
        max_size = (int(max_size / stride) + 1) * stride 
    return max_size

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(int(class_num)) + ': ' + str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def detect(model, img0):
    stride = int(model.stride.max())  # model stride
    imgsz = opt.img_size
    if imgsz <= 0:                    # original size    
        imgsz = dynamic_resize(img0.shape)
    imgsz = check_img_size(imgsz, s=64)  # check img_size
    img = letterbox(img0, imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]
    # Apply NMS
    pred = non_max_suppression_face(pred, opt.conf_thres, opt.iou_thres)[0]
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
    gn_lks = torch.tensor(img0.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
    boxes = []
    h, w, c = img0.shape
    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        pred[:, 5:15] = scale_coords_landmarks(img.shape[2:], pred[:, 5:15], img0.shape).round()
        for j in range(pred.size()[0]):
            xywh = (xyxy2xywh(pred[j, :4].view(1, 4)) / gn).view(-1)
            xywh = xywh.data.cpu().numpy()
            conf = pred[j, 4].cpu().numpy()
            landmarks = (pred[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
            class_num = pred[j, 15].cpu().numpy()
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
            boxes.append([x1, y1, x2-x1, y2-y1, conf])
    return boxes


def distance(e1, e2):
    print('in distance and e1 and e2 are',e1.shape,e2.shape)
    return (e1-e2).norm().item()


def crop(image,x1,x2,y1,y2):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print('image shapes: ',image.shape)
    print('crop shapes :',x1,x2,y1,y2)
    #input()
    image = image[y1:y2,x1:x2,:]
    image = cv2.resize(image, (160, 160))
    #plt.imshow(fixed_image_standardization(image))
    #plt.savefig('image.png')
    #input()
    image = np.transpose(image,(2,0,1))
    image = np.float32(image)
    image = torch.from_numpy(image).unsqueeze(0).cuda()
    image = fixed_image_standardization(image)
    return image


embeddings = []
names = []


def pre_process(model,res):
    aligned = []
    global embeddings
    global names
    #from facenet_pytorch import MTCNN
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    def collate_fn(x):
        return x[0]
    dataset = datasets.ImageFolder('data/test_images')
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    print(dataset[0])
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=1)

    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        plt.imshow(x_aligned.detach().cpu().numpy().transpose(1,2,0))
        plt.savefig(str(y)+'.png')
        #input()
        if x_aligned is not None:
            #print(x_aligned)

            #input()
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])
    aligned = torch.stack(aligned).to(device)
    embeddings = res(aligned).detach().cpu()
    print('names ',names)
    print('embeddings ',len(embeddings))

def yolo_pre_process(model,res):
    aligned = []
    global embeddings
    global names
    def collate_fn(x):
        return x[0]
    dataset = datasets.ImageFolder('data/test_images')
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    print(dataset[0])
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=1)

    for x, y in loader:
        image0 = np.array(x.getdata()).reshape(x.size[0], x.size[1], 3)
        #image0 = image0[,:]
        print(type(image0))
        print(image0.shape)
        if image0.shape[0]*image0.shape[1] > 480*640:
            continue
        boxes = detect(model, image0)
        box = boxes[0]
        x_aligned = crop(image0,box[0], box[0]+box[2] ,box[1], box[1]+box[3])
        plt.imshow(x_aligned.detach().cpu().numpy().transpose(1,2,0))
        plt.savefig(str(y)+'.png')
        #input()
        if x_aligned is not None:
            #print(x_aligned)

            #input()
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])
    aligned = torch.stack(aligned).to(device)
    embeddings = res(aligned).detach().cpu()
    print('names ',names)
    print('embeddings ',len(embeddings))

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
    y = (x - mean) / std_adj
    return y


if __name__ == '__main__':

    #res = InceptionResnetV1(pretrained='casia-webface').cuda().eval()
    #res = resnet = InceptionResnetV1(pretrained='vggface2').cuda().eval()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.02, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='../WiderFace/val/images/', type=str, help='dataset path')
    opt = parser.parse_args()
    #print(opt)

    # Load model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    pre_process(model,res)
    #yolo_pre_process(model,res)
    #for i in range(4):
    #    for j in range(4):
    #        print(i,j,distance(embeddings[i],embeddings[j]))
    #input()
    with torch.no_grad():
        # testing dataset
        testset_folder = opt.dataset_folder
        testset_list = opt.dataset_folder[:-7] + "wider_val.txt"
        
        video_capture = cv2.VideoCapture(4)
        cap = cv.VideoCapture('http://192.168.1.33:4747/mjpegfeed')


        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            #frame = np.array(sct.grab(bounding_box))[:,:,:3]
            print(frame.shape)
            #input()
            img0 = frame
            boxes = detect(model, img0)
            for box in boxes:
                if box[4]>0.5:
                    left = box[0]
                    bottom = box[1]+box[3]
                    right = box[0]+box[2]
                    print("boxes ",box[0],box[1],box[2],box[3])
                    image = crop(frame,box[0], box[0]+box[2] ,box[1], box[1]+box[3])
                    b_embeddings = res(image)
                    #b_embeddings = resnet(xb)
                    b_embeddings = b_embeddings.to('cpu')
                    #print('b_embedings',b_embeddings)

                    #global embeddings
                    #global names
                    name = 'unknown'
                    min_dist = 100
                    for i in range(4):
                        print(names[i])
                        dis = distance(b_embeddings[0],embeddings[i])
                        if(dis<min_dist):
                            min_dist = dis
                            print(names[i],dis)
                            name  = names[i]
                    #input()
                    cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
   





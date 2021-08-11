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
import cv2 as cv
import copy

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

def show_results(image, xywh, conf, landmarks, class_num):
    h,w,c = image.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = xywh[0]
    y1 = xywh[1]
    x2 = xywh[2] + x1
    y2 = xywh[3] + y1
    #img = cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=2, lineType=cv2.LINE_AA)
    img = copy.deepcopy(image)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0 , 0), 2,  lineType=cv2.LINE_AA)
    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        img = cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    label = str(int(class_num)) + ': ' + str(conf)[:5]
    img = cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    return img

def detect(model, img0, opt):
    device = select_device(opt.device)
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
            #boxes.append([x1, y1, x2-x1, y2-y1, conf])
            boxes.append([x1, y1, x2-x1, y2-y1, conf, j, landmarks, class_num])
    return boxes

def make_rects(img, boxes):
    faces = []
    newboxes = []
    for box in boxes:
        if(box[4]<0.5):
            continue
        newboxes.append(box)
        x1 = box[0]
        y1 = box[1]
        x2 = box[2] + x1
        y2 = box[3] + y1
        ind = box[5]        
        img = show_results(img, [box[0], box[1], box[2], box[3]], box[4], box[6], box[7])

    return img, newboxes


if __name__ == '__main__':
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
    #path = yolov5n-0.5/archive/data.pkl
    # Load model
    
    device = select_device(opt.device)
    
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    
    with torch.no_grad():
        # testing dataset
        testset_folder = opt.dataset_folder
        testset_list = opt.dataset_folder[:-7] + "wider_val.txt"

        # with open(testset_list, 'r') as fr:
        #     test_dataset = fr.read().split()
        #     num_images = len(test_dataset)
        path1 = 'http://192.168.1.33:4747/mjpegfeed'
        path2 = 'http://10.28.0.50:4747/mjpegfeed'
        path3 = 'mani.mp4'
        path4 = 0
        cap = cv.VideoCapture(path1)

        start = time.time()
        lastframe = 0
        allignedframe = 0
        cnt = 10000
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            now = time.time()
            cnt -= 1
            if(cnt == 0):
                break
            ret, frame = cap.read()
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if(now - start > 0.1) or (type(lastframe) == int):
                p1 = time.time()
                boxes = detect(model, frame, opt)
                #print(boxes)
                result = make_rects(frame, boxes)
                lastframe = result
                p2 = time.time()
                start = now
                print(p2 - p1)

            cv.imshow('frame', lastframe)
            cv.imshow('alligned', frame)

            if cv.waitKey(1) == ord('q') :
                break

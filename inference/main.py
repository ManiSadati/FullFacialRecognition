import os
import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../detection')
sys.path.insert(1, '../allignment')
import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np


from tqdm import tqdm
import cv2 as cv


from detection.test_widerface import *
from allignment.allign import allign

def main_process(frame):
    boxes = detect(model, frame, opt)
    result, boxes = make_rects(frame, boxes)
    faces = allign(frame, boxes)
    return result, faces


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../data/yolov5n-0.5.pt', help='model.pt path(s)')
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
        
        path1 = 'http://192.168.1.33:4747/mjpegfeed'
        path2 = 'http://10.28.0.50:4747/mjpegfeed'
        path3 = 'mani.mp4'
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
                result, faces = main_process(frame)
                lastframe = result
                p2 = time.time()
                start = now
                print(p2 - p1)
                cv.imshow('frame', lastframe)
            

            if cv.waitKey(1) == ord('q') :
                break
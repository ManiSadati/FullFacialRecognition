import cv2 as cv
import numpy as np
import copy 
import sys
sys.path.insert(1, '../recognition/testSphereFace')
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
import matplotlib.pyplot as plt
import cv2

def rotateImage(image, angle ):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.2)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

def allign_face2(face, xywh, landmarks):
    x1 = xywh[0]
    y1 = xywh[1]
    x2 = xywh[2] + x1
    y2 = xywh[3] + y1
    lefteye_x, lefteye_y = landmarks[0], landmarks[1]
    righteye_x, righteye_y = landmarks[2], landmarks[3]
    nose_x, nose_y = landmarks[4], landmarks[5]
    dx = righteye_x - lefteye_x
    dy = righteye_y - lefteye_y
    angle = np.degrees(np.arctan2(dy, dx))
    img = rotateImage(face, angle)
    cv.imshow('test' + str(xywh[5]), img)
    return img



def allign_face(face, h, w,xywh, landmarks):
    
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],[48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    src_pts = []
    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    x1 = xywh[0]
    y1 = xywh[1]
    x2 = xywh[2] + x1
    y2 = xywh[3] + y1

    for i in range(5):
        point_x = int(landmarks[2 * i] * w - x1)
        #print(point_x , " <--> ", landmarks[2*i])
        point_y = int(landmarks[2 * i + 1] * h - y1)
        src_pts.append([point_x, point_y])   
        #face = cv2.circle(face, (point_x, point_y), 2, clors[i], -1)
    

    #crop_size = (96, 112)
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(face, tfm, crop_size)
    cv.imshow('test' + str(xywh[5]), face_img)
    return face_img


def allign(img, boxes):
    faces = []
    h, w, c = img.shape
    for box in boxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2] + x1
        y2 = box[3] + y1     
        face = img[y1:y2, x1:x2, :]
        alligned_face = allign_face(face, h, w,box, box[6])
        alligned_face = alligned_face.transpose(2, 0, 1).reshape((1,3,112,96))
        alligned_face = (alligned_face-127.5)/128.0
        faces.append(alligned_face)
    return faces














# # import the necessary packages
# from .helpers import FACIAL_LANDMARKS_IDXS
# from .helpers import shape_to_np
# import numpy as np
# import cv2
# class FaceAligner:
#     def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
#         desiredFaceWidth=256, desiredFaceHeight=None):
#         # store the facial landmark predictor, desired output left
#         # eye position, and desired output face width + height
#         self.predictor = predictor
#         self.desiredLeftEye = desiredLeftEye
#         self.desiredFaceWidth = desiredFaceWidth
#         self.desiredFaceHeight = desiredFaceHeight
#         # if the desired face height is None, set it to be the
#         # desired face width (normal behavior)
#         if self.desiredFaceHeight is None:
#             self.desiredFaceHeight = self.desiredFaceWidth  

            
#     def align(self, image, gray, rect):
#         # convert the landmark (x, y)-coordinates to a NumPy array
#         shape = self.predictor(gray, rect)
#         shape = shape_to_np(shape)
#         # extract the left and right eye (x, y)-coordinates
#         (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
#         (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
#         leftEyePts = shape[lStart:lEnd]
#         rightEyePts = shape[rStart:rEnd]


#         # compute the center of mass for each eye
#         leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
#         rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
#         # compute the angle between the eye centroids
#         dY = rightEyeCenter[1] - leftEyeCenter[1]
#         dX = rightEyeCenter[0] - leftEyeCenter[0]
#         angle = np.degrees(np.arctan2(dY, dX)) - 180

#         # compute the desired right eye x-coordinate based on the
#         # desired x-coordinate of the left eye
#         desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
#         # determine the scale of the new resulting image by taking
#         # the ratio of the distance between eyes in the *current*
#         # image to the ratio of distance between eyes in the
#         # *desired* image
#         dist = np.sqrt((dX ** 2) + (dY ** 2))
#         desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
#         desiredDist *= self.desiredFaceWidth
#         scale = desiredDist / dist

#         # compute center (x, y)-coordinates (i.e., the median point)
#         # between the two eyes in the input image
#         eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
#             (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
#         # grab the rotation matrix for rotating and scaling the face
#         M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
#         # update the translation component of the matrix
#         tX = self.desiredFaceWidth * 0.5
#         tY = self.desiredFaceHeight * self.desiredLeftEye[1]
#         M[0, 2] += (tX - eyesCenter[0])
#         M[1, 2] += (tY - eyesCenter[1])

#         # apply the affine transformation
#         (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
#         output = cv2.warpAffine(image, M, (w, h),
#             flags=cv2.INTER_CUBIC)
#         # return the aligned face
#         return output

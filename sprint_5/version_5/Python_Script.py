import os
import cv2
import dlib
import numpy as np
import imutils
import ntpath
from imutils import face_utils
from elasticsearch import Elasticsearch
from collections import OrderedDict
from PIL import Image
# from pypylon import pylon
from numpy import *
import cv2
import time
import numpy as np
from random import randint
from datetime import datetime
from mtcnn.mtcnn import MTCNN
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

face_landmark_path = './model/shape_predictor_68_face_landmarks.dat'
detector = MTCNN()

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])



def get_head_pose(size, gray, shape):

    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    initial_camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    initial_distCoeffs = np.zeros((8, 1), np.float32)
    # _, camera_matrix, dist_coeffs,_,_ = cv2.calibrateCamera([object_pts], [image_pts], gray.shape[::-1], initial_camera_matrix, initial_distCoeffs,flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT)
    _, rotation_vector, translation_vector = cv2.solvePnP(object_pts, image_pts, initial_camera_matrix, initial_distCoeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_matrix)
    return euler_angle


def process_img(img_path):
    head, img_name = ntpath.split(img_path)
    img_path = os.path.join(BASE_DIR,'media\images', img_name)
    head, img_name = ntpath.split(img_path)
    itr=0
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    predictor = dlib.shape_predictor(face_landmark_path)
    num_faces = detector.detect_faces(img)
    if len(num_faces) > 0:
        attentive_faces_num = 0
        for face in num_faces:
            x,y,w,h = face['box']
            shape = predictor(img, dlib.rectangle(x,y,w+x,y+h))
            # shape = predictor(img, face)
            shape = face_utils.shape_to_np(shape)
            euler_angle = get_head_pose(img.shape, gray, shape)
            for (x, y) in shape:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            i, j = shape[0]
            face_point = (i + 10, j - 120)
            if (euler_angle[0, 0] > 27) or (euler_angle[0, 0] < -27):
                cv2.putText(img, "InAttentive", face_point, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0),
                            thickness=2)
            elif (euler_angle[1, 0] > 40) or (euler_angle[1, 0] < -40):
                cv2.putText(img, "InAttentive", face_point, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0),
                            thickness=2)
            else:
                cv2.putText(img, "Attentive", face_point, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0),
                            thickness=2)

                attentive_faces_num += 1
        cv2.putText(img,
                    "Engagement_Percentage" + "{:6.2f}".format((attentive_faces_num / len(num_faces)) * 100),
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
        engagement_percentage = {"time": datetime.now(),
                                 "engagement": (attentive_faces_num / len(num_faces)) * 100,
                                 "faces": len(num_faces), "attentive": attentive_faces_num,
                                 "inattentive": len(num_faces) - attentive_faces_num}
        if not os.path.exists('images_processed'):
            os.makedirs('images_processed')
        cv2.imwrite('images_processed/'+img_name,img)
        return engagement_percentage

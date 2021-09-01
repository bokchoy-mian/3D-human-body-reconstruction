#!/usr/bin/python
#-- coding:utf8 --
import sys
import cv2
import os
import matplotlib.pyplot as plt
import json
import numpy as  np
# Import Openpose (Ubuntu)
import pyopenpose as op

def show_img(datum):
    plt.figure (figsize=(12, 10))
    plt.imshow (datum.cvOutputData[:, :, ::-1])
    plt.title ("OpenPose 1.5.0 - Tutorial Python API")
    plt.axis ("off")
    plt.show ()

def openpose(img_path,save_path):
    params = dict ()
    params["model_folder"] = "/home/234-235/openpose/models"
    params["model_pose"]="BODY_25"
    params["face"] = False
    params["hand"] = True
    params["num_gpu"] = 1
    params["num_gpu_start"] = 0
    params["write_json"] = os.path.join(save_path)

    # Starting OpenPose
    opWrapper = op.WrapperPython ()
    opWrapper.configure (params)
    opWrapper.start ()

    # Process Image
    datum = op.Datum ()
    imageToProcess = cv2.imread (img_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop ([datum])


    show_img(datum)

def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)


        keypoints.append(body_keypoints)

    return keypoints
def main(path):

    for root, dirs, files in os.walk (path, topdown=True):
        for dir in dirs:
            file_path = os.path.join (root, dir)
            openpose ( os.path.join (file_path, 'front_rgb.png'), file_path)
if __name__ == '__main__':
    openpose('../data/test_IronMan/front_rgb.png', '../data/test_IronMan/')
    # main('../data/test_web_512')
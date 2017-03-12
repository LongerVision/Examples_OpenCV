import sys
sys.path.append('/usr/local/python/3.5')

# Standard imports
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
#import yaml
import unittest


#############################Read Stereo Calibrated Parameters#####################################
# Load Calibrated Parameters
stereoCalibrationFile = "/media/peijia/cinetec/Databases/cinetech/Calibration/stereo_pointgrey_feb24_2.xml"
stereoCalibrationParams = cv2.FileStorage(stereoCalibrationFile, cv2.FILE_STORAGE_READ)
# image_width = stereoCalibrationParams.getNode("image_width")
# image_height = stereoCalibrationParams.getNode("image_height")
image_width = 1280
image_height = 1024
image_size = (image_width, image_height)
camera_matrix_l = stereoCalibrationParams.getNode("camera_matrix_l").mat()
distortion_coefficients_l = stereoCalibrationParams.getNode("distortion_coefficients_l").mat()
camera_matrix_r = stereoCalibrationParams.getNode("camera_matrix_r").mat()
distortion_coefficients_r = stereoCalibrationParams.getNode("distortion_coefficients_r").mat()
r_matrix = stereoCalibrationParams.getNode("r_matrix").mat()    # rotation between stereo
# t_matrix = stereoCalibrationParams.getNode("t_matrix").mat()    # translation between stereo
r1_matrix = stereoCalibrationParams.getNode("r1_matrix").mat()
r2_matrix = stereoCalibrationParams.getNode("r2_matrix").mat()
new_rect_cam_matrix_l = stereoCalibrationParams.getNode("new_rect_cam_matrix_l").mat()
new_rect_cam_matrix_r = stereoCalibrationParams.getNode("new_rect_cam_matrix_r").mat()
mapx_l, mapy_l = cv2.fisheye.initUndistortRectifyMap(camera_matrix_l, distortion_coefficients_l, r1_matrix, new_rect_cam_matrix_l, image_size, cv2.CV_16SC2)
mapx_r, mapy_r = cv2.fisheye.initUndistortRectifyMap(camera_matrix_r, distortion_coefficients_r, r2_matrix, new_rect_cam_matrix_r, image_size, cv2.CV_16SC2)
###################################################################################################


#####################################List All Image Pairs##########################################
# List image names
leftImgsDir = "/media/peijia/cinetec/Databases/cinetech/Calibration/images/left"
rightImgsDir = "/media/peijia/cinetec/Databases/cinetech/Calibration/images/right"

leftImgFileNames = [os.path.join(leftImgsDir, fn) for fn in next(os.walk(leftImgsDir))[2]]
rightImgFileNames = [os.path.join(rightImgsDir, fn) for fn in next(os.walk(rightImgsDir))[2]]
if (len(leftImgFileNames) != len(rightImgFileNames) ):
    print("The number of left images must be equal to the number of right images.")
    exit()
###################################################################################################


########################################Blob Detector##############################################
# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 8;
blobParams.maxThreshold = 255;

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 64
blobParams.maxArea = 2500

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)
###################################################################################################


############################For Loop, for All Captured Image Pairs#################################
nbOfImgs = len(leftImgFileNames)
for i in range(0, nbOfImgs-1):
# i = 0
    # Load images
    imLeft = cv2.imread(leftImgFileNames[i], cv2.IMREAD_GRAYSCALE)
    imRight = cv2.imread(rightImgFileNames[i], cv2.IMREAD_GRAYSCALE)

    # Calibration
    imLeftRemapped = cv2.remap(imLeft, mapx_l, mapy_l, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    imRightRemapped = cv2.remap(imRight, mapx_r, mapy_r, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # Detect blobs.
    keypointsLeft = blobDetector.detect(imLeftRemapped)
    keypointsRight = blobDetector.detect(imRightRemapped)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints_left = cv2.drawKeypoints(imLeftRemapped, keypointsLeft, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.circle(img,(447,63), 63, (0,0,255), -1)
    im_with_keypoints_right = cv2.drawKeypoints(imRightRemapped, keypointsRight, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Save/show keypoints
    # cv2.imwrite("KeypointsLeft.jpg", im_with_keypoints_left)
    # cv2.imwrite("KeypointsRight.jpg", im_with_keypoints_right)
    cv2.imshow("KeypointsLeft", im_with_keypoints_left)
    cv2.imshow("KeypointsRight", im_with_keypoints_right)


    cv2.waitKey(2)
###################################################################################################



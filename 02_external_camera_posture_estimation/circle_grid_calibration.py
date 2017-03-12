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
stereoCalibrationFile = "/media/peijia/cinetec/Databases/cinetech/calibration/stereo_pointgrey_feb24_2.xml"
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
leftImgsDir = "/media/peijia/cinetec/Databases/cinetech/calibration/02_circlegrids2/left"
rightImgsDir = "/media/peijia/cinetec/Databases/cinetech/calibration/02_circlegrids2/right"


leftImgFileNames = [os.path.join(leftImgsDir, fn) for fn in next(os.walk(leftImgsDir))[2]]
rightImgFileNames = [os.path.join(rightImgsDir, fn) for fn in next(os.walk(rightImgsDir))[2]]
if (len(leftImgFileNames) != len(rightImgFileNames)):
    print("The number of left images must be equal to the number of right images.")
    exit()
###################################################################################################


########################################Blob Detector##############################################
# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
#blobParams.minArea = 64
#blobParams.maxArea = 2500
blobParams.minArea = 25
blobParams.maxArea = 1024

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

# Original blob coordinates
objectPoints = np.zeros((44, 3))
objectPoints[0]  = (0  , 0  , 0)
objectPoints[1]  = (0  , 72 , 0)
objectPoints[2]  = (0  , 144, 0)
objectPoints[3]  = (0  , 216, 0)
objectPoints[4]  = (36 , 36 , 0)
objectPoints[5]  = (36 , 108, 0)
objectPoints[6]  = (36 , 180, 0)
objectPoints[7]  = (36 , 252, 0)
objectPoints[8]  = (72 , 0  , 0)
objectPoints[9]  = (72 , 72 , 0)
objectPoints[10] = (72 , 144, 0)
objectPoints[11] = (72 , 216, 0)
objectPoints[12] = (108, 36,  0)
objectPoints[13] = (108, 108, 0)
objectPoints[14] = (108, 180, 0)
objectPoints[15] = (108, 252, 0)
objectPoints[16] = (144, 0  , 0)
objectPoints[17] = (144, 72 , 0)
objectPoints[18] = (144, 144, 0)
objectPoints[19] = (144, 216, 0)
objectPoints[20] = (180, 36 , 0)
objectPoints[21] = (180, 108, 0)
objectPoints[22] = (180, 180, 0)
objectPoints[23] = (180, 252, 0)
objectPoints[24] = (216, 0  , 0)
objectPoints[25] = (216, 72 , 0)
objectPoints[26] = (216, 144, 0)
objectPoints[27] = (216, 216, 0)
objectPoints[28] = (252, 36 , 0)
objectPoints[29] = (252, 108, 0)
objectPoints[30] = (252, 180, 0)
objectPoints[31] = (252, 252, 0)
objectPoints[32] = (288, 0  , 0)
objectPoints[33] = (288, 72 , 0)
objectPoints[34] = (288, 144, 0)
objectPoints[35] = (288, 216, 0)
objectPoints[36] = (324, 36 , 0)
objectPoints[37] = (324, 108, 0)
objectPoints[38] = (324, 180, 0)
objectPoints[39] = (324, 252, 0)
objectPoints[40] = (360, 0  , 0)
objectPoints[41] = (360, 72 , 0)
objectPoints[42] = (360, 144, 0)
objectPoints[43] = (360, 216, 0)
axis = np.float32([[360,0,0], [0,240,0], [0,0,-120]]).reshape(-1,3)
###################################################################################################


############################For Loop, for All Captured Image Pairs#################################
# https://www.learnopencv.com/blob-detection-using-opencv-python-c/
# http://stackoverflow.com/questions/39272510/camera-calibration-with-circular-pattern
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
nbOfImgs = len(leftImgFileNames)
for i in range(0, nbOfImgs-1):
# for i,frame in enumerate(camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)):
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
    im_with_keypoints_left_gray = cv2.cvtColor(im_with_keypoints_left, cv2.COLOR_BGR2GRAY)
    im_with_keypoints_right = cv2.drawKeypoints(imRightRemapped, keypointsRight, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_right_gray = cv2.cvtColor(im_with_keypoints_right, cv2.COLOR_BGR2GRAY)
    
    
    # Find the chess board corners
    retLeft, cornersLeft = cv2.findCirclesGrid(im_with_keypoints_left, (4,11), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)
    retRight, cornersRight = cv2.findCirclesGrid(im_with_keypoints_right, (4,11), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)

    # For left image: if found, add object points, image points (after refining them)
    if retLeft == True:
        corners2Left = cv2.cornerSubPix(im_with_keypoints_left_gray, cornersLeft, (11,11), (-1,-1), criteria)

        # Draw and display the corners
        im_with_keypoints_left_gray = cv2.drawChessboardCorners(imLeftRemapped, (4,11), corners2Left, retLeft)

        # 3D posture
        if len(corners2Left) == len(objectPoints):
            retvalLeft, rvecLeft, tvecLeft = cv2.solvePnP(objectPoints, corners2Left, camera_matrix_l, distortion_coefficients_l)
        
        if retvalLeft:
            # project 3D points to image plane
            projectedPointsLeft, jacLeft = cv2.projectPoints(objectPoints, rvecLeft, tvecLeft, camera_matrix_l, distortion_coefficients_l)
            # project axis to image plane
            projectedAxisLeft, jacLeftAsix = cv2.projectPoints(axis, rvecLeft, tvecLeft, camera_matrix_l, distortion_coefficients_l)
            for p in projectedPointsLeft:
                p = np.int32(p).reshape(-1,2)
                cv2.circle(im_with_keypoints_left, (p[0][0], p[0][1]), 3, (0,0,255))
            corner = tuple(corners2Left[0].ravel())
            im_with_keypoints_left = cv2.line(im_with_keypoints_left, corner, tuple(projectedAxisLeft[0].ravel()), (255,0,0), 2)
            im_with_keypoints_left = cv2.line(im_with_keypoints_left, corner, tuple(projectedAxisLeft[1].ravel()), (0,255,0), 2)
            im_with_keypoints_left = cv2.line(im_with_keypoints_left, corner, tuple(projectedAxisLeft[2].ravel()), (0,0,255), 2)
            #im_with_circle_center_left_BGR = cv2.drawChessboardCorners(im_with_circle_center_left, (4,11), projectedPointsLeft, retvalLeft)
    else:
        # im_with_circle_center_left = cv2.cvtColor(im_with_circle_center_left, cv2.COLOR_GRAY2BGR)
        im_with_keypoints_left = imLeftRemapped


    # For right image: if found, add object points, image points (after refining them)
    if retRight == True:
        corners2Right = cv2.cornerSubPix(im_with_keypoints_right_gray, cornersRight, (11,11), (-1,-1), criteria)

        # Draw and display the corners
        im_with_keypoints_right_gray = cv2.drawChessboardCorners(imRightRemapped, (4,11), corners2Right, retRight)

        # 3D posture
        if len(corners2Right) == len(objectPoints):
            retvalRight, rvecRight, tvecRight = cv2.solvePnP(objectPoints, corners2Right, camera_matrix_r, distortion_coefficients_r)

        if retvalRight:
            # project 3D points to image plane
            projectedPointsRight, jacRight = cv2.projectPoints(objectPoints, rvecRight, tvecRight, camera_matrix_r, distortion_coefficients_r)
            # project axis to image plane
            projectedAxisRight, jacRightAsix = cv2.projectPoints(axis, rvecRight, tvecRight, camera_matrix_r, distortion_coefficients_r)
            for p in projectedPointsRight:
                p = np.int32(p).reshape(-1,2)
                cv2.circle(im_with_keypoints_right, (p[0][0], p[0][1]), 3, (0,0,255))
            corner = tuple(corners2Right[0].ravel())
            im_with_keypoints_right = cv2.line(im_with_keypoints_right, corner, tuple(projectedAxisRight[0].ravel()), (255,0,0), 2)
            im_with_keypoints_right = cv2.line(im_with_keypoints_right, corner, tuple(projectedAxisRight[1].ravel()), (0,255,0), 2)
            im_with_keypoints_right = cv2.line(im_with_keypoints_right, corner, tuple(projectedAxisRight[2].ravel()), (0,0,255), 2)
            #im_with_circle_center_right_BGR = cv2.drawChessboardCorners(im_with_circle_center_right, (4,11), projectedPointsRight, retvalRight)
    else:
        # im_with_circle_center_right_BGR = cv2.cvtColor(im_with_circle_center_right, cv2.COLOR_GRAY2BGR)
        im_with_keypoints_right = imRightRemapped
    

    # Save/show circule grid
    #cv2.imwrite("CircularGridLeft.jpg", im_with_keypoints_left)
    #cv2.imwrite("CircularGridRight.jpg", im_with_keypoints_right)
    cv2.imshow("KeypointsLeft", im_with_keypoints_left)
    cv2.imshow("KeypointsRight", im_with_keypoints_right)

    cv2.waitKey(2)

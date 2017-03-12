import sys
sys.path.append('/usr/local/python/3.5')  # whichever folder that contains **cv2.so** when you built OpenCV


import os
import cv2
from cv2 import aruco
import numpy as np


calibrationFile = "calibrationFileName.xml"
calibrationParams = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ)
camera_matrix = calibrationParams.getNode("cameraMatrix").mat()
dist_coeffs = calibrationParams.getNode("distCoeffs").mat()


r = calibrationParams.getNode("R").mat()
new_camera_matrix = calibrationParams.getNode("newCameraMatrix").mat()


image_size = (1920, 1080)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, r, new_camera_matrix, image_size, cv2.CV_16SC2)


aruco_dict = aruco.Dictionary_get( aruco.DICT_6X6_1000 )
arucoParams = aruco.DetectorParameters_create()
markerLength = 20 # Here, I'm using centimetre as a unit.


imgDir = "imgSequence"  # Specify the image directory
imgFileNames = [os.path.join(imgDir, fn) for fn in next(os.walk(imgDir))[2]]
nbOfImgs = len(imgFileNames)


for i in range(0, nbOfImgs):
    img = cv2.imread(imgFileNames[i], cv2.IMREAD_GRAYSCALE)
    imgRemapped = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT) # for fisheye remapping
    imRemapped_color = cv2.cvtColor(imgRemapped, cv2.COLOR_GRAY2BGR)  # for later display in color
    corners, ids, rejectedImgPoints = aruco.detectMarkers(imgRemapped, aruco_dict, parameters=arucoParams) # Detect aruco
    if ids != None: # if aruco marker detected
        rvec, tvec = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_coeffs) # For a single marker
        imgWithAruco = aruco.drawDetectedMarkers(imRemapped_color, corners, ids, (0,255,0))
        imgWithAruco = aruco.drawAxis(imgWithAruco, camera_matrix, dist_coeffs, rvec, tvec, 100)  # axis length 100 can be changed according to your requirement
    else: # if aruco marker is NOT detected
        imgWithAruco = imRemapped_color # assign imRemapped_color to imgWithAruco directly
        cv2.imshow("aruco", imgWithAruco) # display
    if cv2.waitKey(2) & 0xFF == ord('q'): # if 'q' is pressed, quit.
        break


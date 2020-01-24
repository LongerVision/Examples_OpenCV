################################################################################
#                                                                              #
#                                                                              #
#           IMPORTANT: READ BEFORE DOWNLOADING, COPYING AND USING.             #
#                                                                              #
#                                                                              #
#      Copyright [2017] [ShenZhen Longer Vision Technology], Licensed under    #
#      ******** GNU General Public License, version 3.0 (GPL-3.0) ********     #
#      You are allowed to use this file, modify it, redistribute it, etc.      #
#      You are NOT allowed to use this file WITHOUT keeping the License.       #
#                                                                              #
#      Longer Vision Technology is a startup located in Chinese Silicon Valley #
#      NanShan, ShenZhen, China, (http://www.longervision.cn), which provides  #
#      the total solution to the area of Machine Vision & Computer Vision.     #
#      The founder Mr. Pei JIA has been advocating Open Source Software (OSS)  #
#      for over 12 years ever since he started his PhD's research in England.  #
#                                                                              #
#      Longer Vision Blog is Longer Vision Technology's blog hosted on github  #
#      (http://longervision.github.io). Besides the published articles, a lot  #
#      more source code can be found at the organization's source code pool:   #
#      (https://github.com/LongerVision/OpenCV_Examples).                      #
#                                                                              #
#      For those who are interested in our blogs and source code, please do    #
#      NOT hesitate to comment on our blogs. Whenever you find any issue,      #
#      please do NOT hesitate to fire an issue on github. We'll try to reply   #
#      promptly.                                                               #
#                                                                              #
#                                                                              #
# Version:          0.0.1                                                      #
# Author:           JIA Pei                                                    #
# Contact:          jiapei@longervision.com                                    #
# URL:              http://www.longervision.cn                                 #
# Create Date:      2017-03-10                                                 #
# Modified Date:    2020-01-18                                                 #
################################################################################

import sys

# Standard imports
import os
import cv2
from cv2 import aruco
import numpy as np



# Load Calibrated Parameters
fs = cv2.FileStorage("calibration.yml", cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode("camera_matrix").mat()
dist_coeffs = fs.getNode("dist_coeff").mat()
fs.release()
image_size = (640, 480)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), camera_matrix, image_size, cv2.CV_16SC2)



aruco_dict = aruco.Dictionary_get( aruco.DICT_4X4_1000 )
markerLength = 100   # Here, our measurement unit is millimeters.
arucoParams = aruco.DetectorParameters_create()



imgDir = "imgSequence/4x4_1000-0"  # Specify the image directory
imgFileNames = [os.path.join(imgDir, fn) for fn in next(os.walk(imgDir))[2]]
nbOfImgs = len(imgFileNames)

count = 0
for i in range(0, nbOfImgs):
    img = cv2.imread(imgFileNames[i], cv2.IMREAD_COLOR)
    filename = "original" + str(i).zfill(3) +".jpg"
    cv2.imwrite(filename, img)
    imgRemapped = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT) # for fisheye remapping
    imgRemapped_gray = cv2.cvtColor(imgRemapped, cv2.COLOR_BGR2GRAY)    # aruco.detectMarkers() requires gray image
    filename = "remappedgray" + str(i).zfill(3) +".jpg"
    cv2.imwrite(filename, imgRemapped_gray)
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(imgRemapped_gray, aruco_dict, parameters=arucoParams) # Detect aruco
    if ids != None: # if aruco marker detected
        rvec, tvec, trash = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_coeffs) # posture estimation from a single marker
        imgWithAruco = aruco.drawDetectedMarkers(imgRemapped, corners, ids, (0,255,0))
        imgWithAruco = aruco.drawAxis(imgWithAruco, camera_matrix, dist_coeffs, rvec, tvec, 100)    # axis length 100 can be changed according to your requirement
        filename = "calibrated" + str(count).zfill(3) +".jpg"
        cv2.imwrite(filename, imgWithAruco)
        count += 1
    else:   # if aruco marker is NOT detected
        imgWithAruco = imgRemapped  # assign imRemapped_color to imgWithAruco directly

    cv2.imshow("aruco", imgWithAruco)   # display

    if cv2.waitKey(10) & 0xFF == ord('q'):   # if 'q' is pressed, quit.
        break


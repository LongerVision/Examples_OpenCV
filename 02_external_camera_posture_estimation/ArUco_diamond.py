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
# Create Date:      2017-03-20                                                 #
################################################################################

import sys
sys.path.append('/usr/local/python/3.5')

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



aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_250 )
squareLength = 40   # Here, our measurement unit is centimetre.
markerLength = 25   # Here, our measurement unit is centimetre.
arucoParams = aruco.DetectorParameters_create()



videoFile = "aruco_diamond.mp4"
cap = cv2.VideoCapture(videoFile)

while(True):
    ret, frame = cap.read() # Capture frame-by-frame
    if ret == True:
        frame_remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)     # for fisheye remapping
        frame_remapped_gray = cv2.cvtColor(frame_remapped, cv2.COLOR_BGR2GRAY)  # aruco.detectMarkers() requires gray image

        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_remapped_gray, aruco_dict, parameters=arucoParams)  # First, detect markers

        if ids != None: # if there is at least one marker detected
            diamondCorners, diamondIds = aruco.detectCharucoDiamond(frame_remapped_gray, corners, ids, squareLength/markerLength)   # Second, detect diamond markers
            if len(diamondCorners) >= 1:    # if there is at least one diamond detected
                im_with_diamond = aruco.drawDetectedDiamonds(frame_remapped, diamondCorners, diamondIds, (0,255,0))
                rvec, tvec = aruco.estimatePoseSingleMarkers(diamondCorners, squareLength, camera_matrix, dist_coeffs)  # posture estimation from a diamond
                im_with_diamond = aruco.drawAxis(im_with_diamond, camera_matrix, dist_coeffs, rvec, tvec, 100)    # axis length 100 can be changed according to your requirement
        else:
            im_with_diamond = frame_remapped

        cv2.imshow("diamond", im_with_diamond)   # display

        if cv2.waitKey(2) & 0xFF == ord('q'):   # press 'q' to quit
            break
    else:
        break


cap.release()   # When everything done, release the capture
cv2.destroyAllWindows()


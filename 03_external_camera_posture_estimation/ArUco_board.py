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
# Create Date:      2017-03-12                                                 #
# Modified Date:    2020-01-18                                                 #
################################################################################

import sys

# Standard imports
import os
import cv2
from cv2 import aruco
import numpy as np



# Load Calibrated Parameters
fs = cv2.FileStorage('calibration.yml', cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode('camera_matrix').mat()
dist_coeffs = fs.getNode("dist_coeff").mat()
fs.release()
image_size = (640, 480)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), camera_matrix, image_size, cv2.CV_16SC2)



aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )
markerLength = 40   # Here, our measurement unit is milimeter.
markerSeparation = 8   # Here, our measurement unit is milimeter.
board = aruco.GridBoard_create(5, 7, markerLength, markerSeparation, aruco_dict)
arucoParams = aruco.DetectorParameters_create()


videoFile = "aruco_board_66.mp4"
cap = cv2.VideoCapture(videoFile)

idx = 0
count = 0
while(True):
    ret, frame = cap.read() # Capture frame-by-frame

    if ret == True:
        frame_remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)     # for fisheye remapping
        filename = "remapped" + str(idx).zfill(3) +".jpg"
        cv2.imwrite(filename, frame_remapped)
        idx += 1
        frame_remapped_gray = cv2.cvtColor(frame_remapped, cv2.COLOR_BGR2GRAY)  # aruco.detectMarkers() requires gray image

        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_remapped_gray, aruco_dict, parameters=arucoParams)  # First, detect markers
        aruco.refineDetectedMarkers(frame_remapped_gray, board, corners, ids, rejectedImgPoints)

        if ids != None: # if there is at least one marker detected
            im_with_aruco_board = aruco.drawDetectedMarkers(frame_remapped, corners, ids, (0,255,0))
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs)  # posture estimation from a aruco board
            if retval != 0:
                im_with_aruco_board = aruco.drawAxis(im_with_aruco_board, camera_matrix, dist_coeffs, rvec, tvec, 100)  # axis length 100 can be changed according to your requirement
                # Enable the following 2 lines if you want to save the calibration images.
                filename = "calibrated" + str(count).zfill(3) +".jpg"
                cv2.imwrite(filename, im_with_aruco_board)

                count += 1
        else:
            im_with_aruco_board = frame_remapped

        cv2.imshow("arucoboard", im_with_aruco_board)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()   # When everything done, release the capture
# cv2.destroyAllWindows()

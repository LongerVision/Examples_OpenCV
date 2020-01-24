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


cap = cv2.VideoCapture(2)
#initialize the jevois cam. See below - don't change these as it capture videos without any process
cap.set(3,640) #width
cap.set(4,480) #height
cap.set(5,15) #fps
s,img = cap.read()


count = 0
while(True):
    ret, frame = cap.read() # Capture frame-by-frame

    if ret == True:
        filename = "img" + str(count).zfill(3) +".jpg"
        cv2.imwrite(filename, frame)
        count += 1
        
    cv2.imshow("Image Capturing", frame)
        
    if cv2.waitKey(2) & 0xFF == ord('q'):
            break


cap.release()   # When everything done, release the capture

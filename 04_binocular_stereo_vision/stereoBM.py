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
# Create Date:      2020-01-25                                                 #
# Cited and Modified from: https://github.com/opencv/opencv/issues/9783        #
################################################################################

from __future__ import print_function
import numpy as np
import cv2

print('load and downscale images')
imgL = cv2.pyrDown( cv2.imread('./imgPairs/aloeL.jpg') )
imgR = cv2.pyrDown( cv2.imread('./imgPairs/aloeR.jpg') )

min_disp = 16
num_disp = 112 - min_disp
window_size = 17

stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = window_size)
stereo.setMinDisparity(min_disp)
stereo.setNumDisparities(num_disp)
stereo.setBlockSize(window_size)
stereo.setDisp12MaxDiff(0)
stereo.setUniquenessRatio(10)
stereo.setSpeckleRange(32)
stereo.setSpeckleWindowSize(100)

print('compute disparity')
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
disp = stereo.compute(grayL, grayR)     # .astype(np.float32) / 16.0
disp_map = np.int16(disp)

print('display')
cv2.imshow('left', imgL)
cv2.imshow('right', imgR)
disp_map_normalized = cv2.normalize(src=disp_map, dst=disp_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
disp_map_normalized = np.uint8(disp_map_normalized)
cv2.imshow('disparity', disp_map_normalized)
cv2.imwrite('disparity_BM.png', disp_map_normalized)

cv2.waitKey()
cv2.destroyAllWindows()

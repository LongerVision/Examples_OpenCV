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

# Standard imports
import numpy as np
import cv2
import yaml


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

########################################Blob Detector##############################################
# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 64     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 2500   # maxArea may be adjusted to suit for your experiment

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


###################################################################################################
# Original blob coordinates, supposing all blobs are of z-coordinates 0
# And, the distance between every two neighbour blob circle centers is 72 centimetres
# In fact, any number can be used to replace 72.
# Namely, the real size of the circle is pointless while calculating camera calibration parameters.
objp = np.zeros((44, 3), np.float32)
objp[0]  = (0  , 0  , 0)
objp[1]  = (0  , 72 , 0)
objp[2]  = (0  , 144, 0)
objp[3]  = (0  , 216, 0)
objp[4]  = (36 , 36 , 0)
objp[5]  = (36 , 108, 0)
objp[6]  = (36 , 180, 0)
objp[7]  = (36 , 252, 0)
objp[8]  = (72 , 0  , 0)
objp[9]  = (72 , 72 , 0)
objp[10] = (72 , 144, 0)
objp[11] = (72 , 216, 0)
objp[12] = (108, 36,  0)
objp[13] = (108, 108, 0)
objp[14] = (108, 180, 0)
objp[15] = (108, 252, 0)
objp[16] = (144, 0  , 0)
objp[17] = (144, 72 , 0)
objp[18] = (144, 144, 0)
objp[19] = (144, 216, 0)
objp[20] = (180, 36 , 0)
objp[21] = (180, 108, 0)
objp[22] = (180, 180, 0)
objp[23] = (180, 252, 0)
objp[24] = (216, 0  , 0)
objp[25] = (216, 72 , 0)
objp[26] = (216, 144, 0)
objp[27] = (216, 216, 0)
objp[28] = (252, 36 , 0)
objp[29] = (252, 108, 0)
objp[30] = (252, 180, 0)
objp[31] = (252, 252, 0)
objp[32] = (288, 0  , 0)
objp[33] = (288, 72 , 0)
objp[34] = (288, 144, 0)
objp[35] = (288, 216, 0)
objp[36] = (324, 36 , 0)
objp[37] = (324, 108, 0)
objp[38] = (324, 180, 0)
objp[39] = (324, 252, 0)
objp[40] = (360, 0  , 0)
objp[41] = (360, 72 , 0)
objp[42] = (360, 144, 0)
objp[43] = (360, 216, 0)
###################################################################################################


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


cap = cv2.VideoCapture(0)
found = 0
while(found < 10):  # Here, 10 can be changed to whatever number you like to choose
    ret, img = cap.read() # Capture frame-by-frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    keypoints = blobDetector.detect(gray) # Detect blobs.

    # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() . 
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findCirclesGrid(im_with_keypoints, (4,11), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

    if ret == True:
        objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.

        corners2 = cv2.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)    # Refines the corner locations.
        imgpoints.append(corners2)

        # Draw and display the corners.
        im_with_keypoints = cv2.drawChessboardCorners(img, (4,11), corners2, ret)
        found += 1


    cv2.imshow("img", im_with_keypoints) # display
    cv2.waitKey(2)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# It's very important to transform the matrix to list.
data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)


# You can use the following 4 lines of code to load the data in file "calibration.yaml"
# with open('calibration.yaml') as f:
#     loadeddict = yaml.load(f)
# mtxloaded = loadeddict.get('camera_matrix')
# distloaded = loadeddict.get('dist_coeff')


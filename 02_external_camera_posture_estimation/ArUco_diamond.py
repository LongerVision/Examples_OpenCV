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
arucoParams = aruco.DetectorParameters_create()
squareLength = 40   # Here, our measurement unit is centimetre.
markerLength = 25   # Here, our measurement unit is centimetre.



videoFile = "aruco_diamond.mp4"
cap = cv2.VideoCapture(videoFile)

while(True):
    ret, frame = cap.read() # Capture frame-by-frame
    if ret == True:
        frame_remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)     # for fisheye remapping
        frame_remapped_gray = cv2.cvtColor(frame_remapped, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_remapped_gray, aruco_dict, parameters=arucoParams)  # First, detect markers

        if ids != None: # if there is at least one marker detected
            diamondCorners, diamondIds = aruco.detectCharucoDiamond(frame_remapped_gray, corners, ids, squareLength/markerLength)   # Second, detect diamond markers
            if len(diamondCorners) >= 1:    # if there is at least one diamond detected
                im_with_diamond = aruco.drawDetectedDiamonds(frame_remapped, diamondCorners, diamondIds, (0,255,0))
                rvec, tvec = aruco.estimatePoseSingleMarkers(diamondCorners, squareLength, camera_matrix, dist_coeffs)  # posture estimation from a diamond
                im_with_diamond = aruco.drawAxis(im_with_diamond, camera_matrix, dist_coeffs, rvec, tvec, 100)    # axis length 100 can be changed according to your requirement
        else:
            im_with_diamond = frame_remapped

        cv2.imshow("diamondLeft", im_with_diamond)   # display

        if cv2.waitKey(2) & 0xFF == ord('q'):   # press 'q' to quit
            break
    else:
        break


cap.release()   # When everything done, release the capture
cv2.destroyAllWindows()


import sys
sys.path.append('/usr/local/python/3.5')

# Standard imports
import os
import cv2
from cv2 import aruco
import numpy as np



# Load Calibrated Parameters
calibrationFile = "calibrationFileName.xml"
calibrationParams = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ)
camera_matrix = calibrationParams.getNode("cameraMatrix").mat()
dist_coeffs = calibrationParams.getNode("distCoeffs").mat()

r = calibrationParams.getNode("R").mat()
new_camera_matrix = calibrationParams.getNode("newCameraMatrix").mat()

image_size = (1920, 1080)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, r, new_camera_matrix, image_size, cv2.CV_16SC2)



aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )
markerLength = 40   # Here, our measurement unit is centimetre.
markerSeparation = 8   # Here, our measurement unit is centimetre.
board = aruco.GridBoard_create(5, 7, markerLength, markerSeparation, aruco_dict)
arucoParams = aruco.DetectorParameters_create()



videoFile = "aruco_board_57.mp4"
cap = cv2.VideoCapture(videoFile)

while(True):
    ret, frame = cap.read() # Capture frame-by-frame
    if ret == True:
        frame_remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)     # for fisheye remapping
        frame_remapped_gray = cv2.cvtColor(frame_remapped, cv2.COLOR_BGR2GRAY)  # aruco.detectMarkers() requires gray image

        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_remapped_gray, aruco_dict, parameters=arucoParams)  # First, detect markers
        aruco.refineDetectedMarkers(frame_remapped_gray, board, corners, ids, rejectedImgPoints)

        if ids != None: # if there is at least one marker detected
            im_with_aruco_board = aruco.drawDetectedMarkers(frame_remapped, corners, ids, (0,255,0))
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs)  # posture estimation from a aruco board
            if retval != 0:
                im_with_aruco_board = aruco.drawAxis(im_with_aruco_board, camera_matrix, dist_coeffs, rvec, tvec, 100)  # axis length 100 can be changed according to your requirement
        else:
            im_with_aruco_board = frame_remapped

        cv2.imshow("arucoboard", im_with_aruco_board)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()   # When everything done, release the capture
cv2.destroyAllWindows()

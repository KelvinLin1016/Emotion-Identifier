# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:53:39 2019

@author: Data245Lin
"""
import numpy as np
import cv2
import glob

#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.01)

#prepare object points
objp = np.zeros((6*7,3),np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

#Arrays to store object points and image points from all the images.
objpoints = []#3d point in real world space
imgpoints = []#2d points in image plane.

images = glob.glob('*.jpg')
cap=cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray,(7,6),None)
    print(ret)

    #If found, add object points, image points 
    if ret == True:
        objpoints.append(objp)
        
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
        
        #Draw and display the corners
        cv2.drawChessboardCorners(frame,(7,6),corners,ret)
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break     

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez("webcam_calibration_ouput", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
#calib=np.load("webcam_calibration_output");ret,mtx,dist,rvecs,tvecs=calib.files
cap.release()        
cv2.destroyAllWindows()

import os.path as osp
import numpy as np
import cv2
import glob
import pickle

CAL_DIR = 'camera_cal'

# Prepare object points
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object and image points
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

# Make a lis of calibration images
images = glob.glob(osp.join(CAL_DIR, 'calibration*.jpg'))

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    
    # If found, add object points, image points
    if ret == True:
        print('working on ', fname)
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        write_name = 'corners_found_{}.jpg'.format(str(idx))
        cv2.imwrite(osp.join(CAL_DIR, write_name), img)

# Load image for reference
img = cv2.imread(osp.join(CAL_DIR, "calibration1.jpg"))
img_size = (img.shape[1], img.shape[0])

# Do image calibration given object points and image points
ret, mtx, dist, rvects, tvects = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save the camera calibration results for later use
dist_pickle = {"mtx": mtx, "dist": dist}
pickle.dump(dist_pickle, open(osp.join(CAL_DIR, "calibration_pickle.p"), "wb"))

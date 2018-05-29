import os.path as osp
import numpy as np
import cv2
import pickle
import glob
from tracker import tracker

# Read in the saved objpoints and imgpoint
TEST_IMG_DIR = 'test_images'
CAL_DIR = 'camera_cal'
dist_pickle = pickle.load(open(osp.join(CAL_DIR, 'calibration_pickle.p'), 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

def abs_sobel_thresh(img, orient, thresh = (0, 255)):
    # Convert to grayscale

    # Take the derivative or gradient

    # Scale to 8-bit (uint) then convert to type = np.uint8

    # Create mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    

def mag_thresh(img, sobel_kernel = 3, thresh = (0, 255)):
    # Convert to grayscale

    # Take the gradient in x and y seperately

    # Calculate the magnitude

    # Scale to 8-bit (uint) then convert to type = np.uint8

    # Create a binary mask where mag thresholds are met

    # Return this mask as binary_output image
    

def dir_thresh(img, sobel_kernel = 3, thresh = (0, 255)):
    # Convert to grayscale
    
    # Take gradient in x and y seperately
    
    # Take the absolute value of the x and y gradients

    # Return mask as binary output image
    pass

# Thresholds the S-channel of HLS
def hls_select(img, thresh = (0, 255)):
    # Convert to HLS color space
    # Apply a threshold to the S channel
    # Return binary image of threshold results
    pass

def main():
    # pipeline to detect lane
    # Make a lis of calibration images
    images = glob.glob(osp.join(TEST_IMG_DIR, 'calibration*.jpg'))

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
    
        

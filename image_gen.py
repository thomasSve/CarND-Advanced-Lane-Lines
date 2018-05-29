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

####### Sobel gradient absolute (X, Y) threshold
def abs_sobel_thresh(img, orient, thresh = (0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the derivative or gradient
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y')

    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
    # Create mask of 1's where the scaled gradient magnitude thresholds are met
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1

    return binary_output
    
###### Sobel gradient magnitude threshold
def mag_thresh(img, sobel_kernel = 3, thresh = (0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take the gradient in x and y seperately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    # Calculate the magnitude
    mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled = np.uint8(255 * mag_sobel / np.max(mag_sobel))

    # Create a binary mask where magnitude thresholds are met
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1

    return binary_output

###### Sobel gradient direction threshold
def dir_thresh(img, sobel_kernel = 3, thresh = (0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take gradient in x and y seperately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # Use np.arctan2 (abs_sobelx, abs_sobely) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sobelx, abs_sobely)
    
    # Return mask as binary output image
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    return binary_output

###### HLS

def hls_select(img, thresh = (0, 255), channel = 2):
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # Apply a threshold to the S channel
    binary_output = np.zeros_like(hls[:, :, channel])
    binary_output[(hls[:, :, channel] > thresh[0]) & (hls[:, :, channel] <= thresh[1])] = 1

    return binary_output

def color_threshold(img, sthresh = (0, 255), vthresh = (0, 255)):
    s_binary = hls_select(img, thresh = sthresh)
    v_binary = hls_select(img, thresh = vthresh)

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    output = np.zeros_like(hls[:, :, 2])
    output[(s_binary == 1) & (v_binary == 1)] = 1

    return output

def pipeline(img):   
    # Undistort
    img_undistort = undistort(img)
    
    # Perspective Transform
    img_unwarp, M, Minv = unwarp(img_undistort, src, dst)

    # Sobel Absolute
    #img_sobel_abs = abs_sobel_thresh(img_unwarp)

    # Sobel Magnitude
    #img_sobel_mag = mag_thresh(img_unwarp)
    
    # Sobel Gradient Direction
    #img_sobel_dir = dir_thresh(img_unwarp)
    
    # HLS S-channel Threshold
    #img_hls = hls_sthresh(img_unwarp)
    
    return combined, Minv

def main():
    # pipeline to detect lane
    # Make a list of test images
    images = glob.glob(osp.join(TEST_IMG_DIR, 'test*.jpg'))

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        
    
        

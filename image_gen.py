import os.path as osp
import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
# from tracker import tracker

# Read in the saved objpoints and imgpoint
TEST_IMG_DIR = 'test_images'
OUTPUT_IMG_DIR = 'output_images'
CAL_DIR = 'camera_cal'
dist_pickle = pickle.load(open(osp.join(CAL_DIR, 'calibration_pickle.p'), 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

####### Undistort image using saved values from the camera calibration
def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

###### Unwarp image to transform the perspective
# src and dst are the defined destination points for transform
def unwarp(img):
    h, w = img.shape[:2]
    
    src = np.float32([(575,455),
                  (705,455), 
                  (1130,720), 
                  (190,720)])
    dst = np.float32([(200,0),
                  (w-200,00),
                  (w-200,h),
                  (200,h)])
    
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

def binary_mask(scaled, thresh = (0, 255)):
    # Create mask of 1's where the scaled gradient magnitude thresholds are met
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return binary_output

####### Sobel gradient absolute (X, Y) threshold
def abs_sobel_thresh(img, orient, kernel = 3, thresh = (25, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take the derivative or gradient
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y', ksize = kernel)

    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled = np.uint8(255.0 * abs_sobel / np.max(abs_sobel))
    
    return binary_mask(scaled, thresh)
    
###### Sobel gradient magnitude threshold
def mag_thresh(img, sobel_kernel = 3, thresh = (25, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take the gradient in x and y seperately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    # Calculate the magnitude
    mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled = np.uint8(255 * mag_sobel / np.max(mag_sobel))

    return binary_mask(scaled, thresh)

###### Sobel gradient direction threshold
def dir_thresh(img, sobel_kernel = 7, thresh = (0, np.pi/2)):
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

    return binary_mask(grad_dir, thresh)

###### HLS
def hls_select(img, thresh = (200, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 2]
    s_binary = np.zeros_like(hls)
    s_binary[(hls >= thresh[0]) & (hls < thresh[1])] = 1
    return s_binary
    
def combine_gradient(img):
    # Sobel Absolute
    sobelX = abs_sobel_thresh(img, orient = 'x', thresh = (10, 160))
    sobelY = abs_sobel_thresh(img, orient = 'y', thresh = (10, 160))

    # HLS S-Channel binaries
    color_binary = hls_select(img)

    # Combine the sobel binaries and HLS color binaries
    combined_binary = np.zeros_like(sobelX) 
    combined_binary[((sobelX == 1) & (sobelY == 1)) | (color_binary == 1)] = 1
    
    return combined_binary
    
def pipeline(img):   
    # Undistort
    img_undist = undistort(img)

    # Perspective Transform
    img_unwarp = unwarp(img_undist)

    combined_binary = combine_gradient(img_unwarp)

    #window_polyfit(combined_binary, visualize_polyfit = True)

    return np.uint8(combined_binary * 255)

def window_polyfit(binary_warped, visualize_polyfit = False):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                      (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                      (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
        
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if visualize_polyfit:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()


def main():
    # pipeline to detect lane
    # Make a list of test images
    images = glob.glob(osp.join(TEST_IMG_DIR, 'test*.jpg'))
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_distorted = pipeline(img)
        cv2.imwrite(osp.join(OUTPUT_IMG_DIR, 'combined_test_{}.jpg'.format(idx)), img_distorted)

if __name__ == '__main__':
    main()
    
        

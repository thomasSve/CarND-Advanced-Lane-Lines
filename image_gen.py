import os.path as osp
import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
from lane import LaneLinesTracker

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
                  (w-200,0),
                  (w-200,h),
                  (200,h)])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, Minv, M

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
def hls_select(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 2]

def color_thresh(img, sthresh = (100, 255), vthresh = (50, 255)):
    # Process the hls color
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = binary_mask(s_channel, thresh = sthresh)

    # Process the hsv color
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = binary_mask(v_channel, thresh = vthresh)

    # Combine the two thresholds
    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_channel == 1)] = 1

    return output
    
def combine_gradient(img):
    # Sobel Absolute
    sobelX = abs_sobel_thresh(img, orient = 'x', thresh = (12, 255))
    sobelY = abs_sobel_thresh(img, orient = 'y', thresh = (25, 255))

    # Color threshold
    color_binary = color_thresh(img, sthresh = (100, 255), vthresh = (50, 255))

    # Combine the sobel binaries and HLS color binaries
    combined_binary = np.zeros_like(sobelX) 
    combined_binary[((sobelX == 1) & (sobelY == 1) | (color_binary == 1))] = 1
    
    return combined_binary

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1)* height):int(img_ref.shape[0] - level * height), max(0, int(center - width))]
    return output

def process_img(img):   
    # Undistort
    img_undist = undistort(img)

    combined_binary = combine_gradient(img_undist)

    img_unwarp, Minv, M = unwarp(combined_binary)
    img_unwarp = np.uint8(img_unwarp * 255)
    
    curve_centers = LaneLinesTracker(window_height = 80, window_width = 25, margin = 25, my_ym = 10 / 720, my_xm = 4 / 384)

    window_centroids = curve_centers.find_window_centroids(img_unwarp)

    return draw_center_windows(img_unwarp, img, window_centroids, curve_centers, Minv)
    
    
def draw_center_windows(warped, orig_img, window_centroids, curve_centers, Minv):
    # image size
    h, w = orig_img.shape[:2]
    # set width and height
    window_width = curve_centers.window_width
    window_height = curve_centers.window_height
    
    # points to draw left and right windows
    r_points = np.zeros_like(warped)
    l_points = np.zeros_like(warped)

    # points to find left and right lanes
    rightx = []
    leftx = []

    # Go through each level and draw windows
    for level in range(0, len(window_centroids)):
        # window mask to draw window areas
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        
        # Add center values found
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        
        # Add graphic points from window mask to total pixels found
        l_points[(l_points == 255) | ((l_mask == 255))] = 255
        r_points[(r_points == 255) | ((r_mask == 255))] = 255
        
    # Fit lane boundaries to the (left, right) center position found
    yvals = range(0, warped.shape[0])

    res_yvals = np.arange(warped.shape[0] - (window_height / 2), 0, -window_height)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = np.array(left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2], np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = np.array(right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2], np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width / 2, left_fitx[::-1] + window_width / 2), axis = 0), np.concatenate((yvals, yvals[::-1]), axis = 0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis = 0), np.concatenate((yvals, yvals[::-1]), axis = 0))), np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx + window_width / 2, right_fitx[::-1] - window_width / 2), axis = 0), np.concatenate((yvals, yvals[::-1]), axis = 0))), np.int32)

    road = np.zeros_like(orig_img)
    road_bkg = np.zeros_like(orig_img)
    cv2.fillPoly(road, [left_lane], color = [255, 0, 0])
    cv2.fillPoly(road, [right_lane], color = [0, 0, 255])
    cv2.fillPoly(road, [inner_lane], color = [0, 255, 0])
    cv2.fillPoly(road_bkg, [left_lane], color = [255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color = [255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, (w, h), flags = cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, (w, h), flags = cv2.INTER_LINEAR)

    base = cv2.addWeighted(orig_img, 1.0, road_warped_bkg, -1.0, 0.0)
    img_with_lines = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)

    ym_per_pix = curve_centers.ym_per_pix
    xm_per_pix = curve_centers.xm_per_pix

    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix, np.array(leftx, np.float32) * xm_per_pix, 2)
    curvered = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2 * curve_fit_cr[0])
    
    # calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - warped.shape[1]/2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    cv2.putText(img_with_lines, 'Radius of curvature = {} (m)'.format(str(round(curvered, 3))), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_with_lines, 'Vehicle is {} m {} of center'.format(str(abs(round(center_diff, 3))), side_pos), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img_with_lines
        
def main():
    # pipeline to detect lane
    # Make a list of test images
    images = glob.glob(osp.join(TEST_IMG_DIR, 'test*.jpg'))
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_distorted = process_img(img)
        cv2.imwrite(osp.join(OUTPUT_IMG_DIR, 'combined_test_{}.jpg'.format(idx)), img_distorted)

if __name__ == '__main__':
    main()
    
        

import numpy as np

class LaneLinesTracker(object):
    # Identifies the lane lines in a set of images or frames in video

    def __init__(self, window_height = 80, window_width = 25, margin = 25, nwindows = 9, ym_per_pix = 1, xm_per_pix = 1, my_ym = 1, my_xm = 1, smooth_factor = 15):
        self.recent_centers = []
        self.margin = margin
        
        self.window_height = window_height
        self.window_width = window_width

        self.nwindows = nwindows

        self.ym_per_pix = my_ym
        self.xm_per_pix = my_xm

        self.smooth_factor = smooth_factor

    # function for finding and storing lane segment positions
    def find_window_centroids(self, warped):
        window_height = self.window_height
        window_width = self.window_width
        margin = self.margin

        window_centroids = [] # Store the (left, right) window centroids position pr level
        window = np.ones(window_width) # Create our window template that we will use for convolution

        ### First find starting position for left and right lane

        # Sum quarter bottom of image to get size, could use different ratio
        l_sum = np.sum(warped[int(3 * warped.shape[0]/4):, :int(warped.shape[1]/2)], axis = 0)
        r_sum = np.sum(warped[int(3 * warped.shape[0]/4):, int(warped.shape[1]/2):], axis = 0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1]/2)

        # Add what we found for first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, int(warped.shape[0] / window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),:], axis = 0)
            conv_signal = np.convolve(window, image_layer)
            
            # Find best left centroids by using past ones as reference
            # Use window_width / 2 as offset because the convolutional signal is on right side of image, not center
            offset = window_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            # Find best right centroids by using past as reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            
            # Add the results
            window_centroids.append((l_center, r_center))

        self.recent_centers.append(window_centroids)

        return np.average(self.recent_centers[-self.smooth_factor:], axis = 0)
        
        

#############################################################################################################
# ENGI 9804 - G_42
# Dimensional Analysis Using Blob Detection
# Preprocessing Operations Module
#############################################################################################################

import cv2

def preprocess(gray, blur, enhance):
    
    enhanced_image = None
    
    
    # Noise Filtering based on the Filter
    if blur == "gaussian":
        # Apply Gaussian blur to reduce noise
        enhanced_image = cv2.GaussianBlur(gray, (11, 11), 0)
    elif blur == "median":
        # Apply Median blur to reduce noise
        enhanced_image = cv2.medianBlur(gray, 11)
    
    if enhance:
        # Contrast Enhancement
        enhanced_image = cv2.equalizeHist(blurred)  # Apply histogram equalization for contrast enhancement

    return enhanced_image


# Performed thresholding on the blurred image to create a binary image using cv2.threshold(). 
# This step helps to segment the foreground objects from the background.

# Apply Global Thresholding
def binary_threshold(image, threshold_value):

    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    return binary_image

# Apply Adaptive Thresholding
def adaptive_threshold(image, block_size, constant):

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, block_size, constant)

    return binary_image
    
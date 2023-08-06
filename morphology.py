#############################################################################################################
# ENGI 9804 - G_42
# Dimensional Analysis Using Blob Detection
# Morphological Operations Module
#############################################################################################################

import cv2
import numpy as np

# https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

"""
Morphological Closing: 
Morphological closing is an operation that combines dilation followed by erosion.
It is used to fill gaps in the detected blobs and smooth the blobs. In this code, a square-shaped kernel of size (5, 5) is used for the closing operation. 
The cv2.morphologyEx() function is applied with cv2.MORPH_CLOSE as the operation type.

* Morphological Erosion: 
Morphological erosion is an operation that reduces the size of bright regions (white areas in this case). 
It helps distinguish blobs that are close to each other. In this code, a smaller square-shaped kernel of size (3, 3) is used for the erosion operation. 
The cv2.erode() function is applied with iterations=1 to perform one iteration of erosion.

* Morphological Dilation: 
Morphological dilation is an operation that increases the size of bright regions. 
It is used here to fill small gaps inside each blobs, which might have occurred due to thresholding or erosion. 
In this code, a larger square-shaped kernel of size (7, 7) is used for the dilation operation. The cv2.dilate() function is applied with iterations=1 
to perform one iteration of dilation.

By applying morphological closing, erosion, and dilation in this order, we improve the accuracy of blob detection and separation of adjacent blobs. 
Closing helps to fill small gaps and smooth the blobs, erosion helps distinguish closely connected blobs, and dilation fills small gaps inside each blueberry, 
leading to more accurate detection of individual blobs in the final output.
"""

def morphology(thresh, cl_ker, er_ker, di_ker):
    # Perform morphological closing to fill gaps and smooth the blobs
    kernel = np.ones((cl_ker, cl_ker), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Perform morphological erosion to distinguish blobs
    kernel_erosion = np.ones((er_ker, er_ker), np.uint8)
    erosion = cv2.erode(closing, kernel_erosion, iterations=1)

    # Perform morphological dilation to fill small gaps inside each blob
    kernel_dilation = np.ones((di_ker, di_ker), np.uint8)
    dilation = cv2.dilate(erosion, kernel_dilation, iterations=1)

    # Set the Processed Image for Connected Component Analysis
    processed_img = closing
    
    return processed_img
#############################################################################################################
# ENGI 9804 - G_42
# Dimensional Analysis Using Blob Detection
# Blob Detection and Measurement Module
#############################################################################################################

import cv2
import numpy as np


def detectAndMeasure(org_img, proc_img, blobMin, filtMax, filtMin):
    
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(proc_img, connectivity=8)

    # Print Total Number of Blobs Detected
    print(f"Total Blobs Count: {num_labels}")

    # Define the minimum and maximum area thresholds to filter the blobs (adjust these values as needed)
    min_area_threshold = blobMin

    # Define the Filtering Blob Size for Measurement Analysis
    over_sized = filtMax
    under_sized = filtMin

    # Loop through each detected blob
    for label in range(1, num_labels):  # Start from 1 to exclude the background label 0
        area = stats[label, cv2.CC_STAT_AREA]

        # Check if the area is within the specified range
        if min_area_threshold < area:

            # Get the bounding box coordinates for the blob
            x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], \
                        stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]

            # Draw the bounding box around the blob based on Filtered Criteria

            # Filtering Condition
            if  area < under_sized:
                cv2.rectangle(org_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            elif area >= over_sized:
                cv2.rectangle(org_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                cv2.rectangle(org_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Print the area of the blob
            print(f"Blob {label}: Area = {area} pixels")
            
            
    return org_img
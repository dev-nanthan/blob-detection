#############################################################################################################
# ENGI 9804 - G_42
# Dimensional Analysis Using Blob Detection
# Main Module
#############################################################################################################

# Import necessary libraries
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import preprocessing as pre
from morphology import morphology
from blobDetection import detectAndMeasure

def main(img_name, glb_thresh, blob_min, filt_max, filt_min):
    
    # Set the Product Name
    prod = "Potato Chips"
    prod = "Blueberries"

    # Load the image
    img_path = os.path.join("./images/", img_name)
    
    image = cv2.imread(img_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Display both Original and Grayscale images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), num=1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image of "+prod)
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale Image")
    axes[1].axis("off")

    plt.show()

    # Preprocessing
    #==========================================================================================
    blur = "median" # "median"/"gaussian"
    preprocessed = pre.preprocess(gray, blur, False)

    plt.figure(num=2)
    plt.imshow(preprocessed, cmap="gray")
    plt.title("De-noised, Enhanced Image")
    plt.axis("off")
    plt.show()
    
    
    # Thresholding
    #==========================================================================================
    # Apply thresholding to segment the Foreground Objects from Background

    # Adjust the Global threshold_value
    threshold_value = glb_thresh
    thresh_global = pre.binary_threshold(preprocessed, threshold_value)

    # Adjust the block_size and constant
    block_size = 21
    constant = 5
    thresh_adaptive = pre.adaptive_threshold(preprocessed, block_size, constant)

    # Assign the Best Thresholded Image as the Final Binay Image
    thresh = thresh_global

    # Display both binary images side by side using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5),  num=3)

    # Global Thresholding Result
    axes[0].imshow(thresh_global, cmap="gray")
    axes[0].set_title("Global Thresholding")
    axes[0].axis("off")

    # Adaptive Thresholding Result
    axes[1].imshow(thresh_adaptive, cmap="gray")
    axes[1].set_title("Adaptive Thresholding")
    axes[1].axis("off")

    plt.show()

    # Morphological Processing
    #==========================================================================================
    morphed_img = morphology(thresh, 11, 3, 5)
    
    # Display Morphological processed image and Thresholded images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5),  num=4)

    # Thresholded Image 
    axes[0].imshow(thresh, cmap="gray")
    axes[0].set_title("Thresholded Image")
    axes[0].axis("off")
    
    # Morphological Processsing - Closing Result
    axes[1].imshow(morphed_img, cmap="gray")
    axes[1].set_title("Morphological Processed Image")
    axes[1].axis("off")
    
    plt.show()
    
    # Blob Detection and Measurement
    #==========================================================================================
    analyzed_image = detectAndMeasure(image, morphed_img, blob_min, filt_max, filt_min)
    
    # Display the final output with detected and Filtered Blobs
    analyzed_image_rgb = cv2.cvtColor(analyzed_image, cv2.COLOR_BGR2RGB)
    plt.figure(num=5)
    plt.imshow(analyzed_image_rgb)
    plt.title("Detected and Filtered "+prod)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
     # Create an argument parser
    parser = argparse.ArgumentParser(description="G_42 Dimensional Analysis Using Blob Detection")

    # Add arguments
    parser.add_argument("--imgname", default="berries1.png", help="input image name, image should be inside subdir '/images' ")
    parser.add_argument("--thresh", type=int, default=130, help="Global Threshold for Binarization.")
    parser.add_argument("--blobmin", type=int, default=1000, help="Minimum Size for blob detection.")
    parser.add_argument("--filtmax", type=int, default=8000, help="Filter object larger than filtmax Size.")
    parser.add_argument("--filtmin", type=int, default=6000, help="Filter object lesser than filtmin Size.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with provided arguments
    main(args.imgname, args.thresh, args.blobmin,  args.filtmax, args.filtmin)
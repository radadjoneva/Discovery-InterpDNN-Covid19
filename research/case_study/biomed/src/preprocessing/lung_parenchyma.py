# Code Reference: https://ngdc.cncb.ac.cn/ictcf/HUST-19.php
# Adapted: added docstrings, comments, refactored
# added function extarct_lung_parenchyma

import os
from collections import Counter

import cv2
import numpy as np
from skimage import measure


def extract_img_lung_parenchyma(image_path, img_dir, patient=False):
    """Process a single image to extract lung parenchyma and save to the specified output path.

    Args:
        image_path (str): Path to the input image.
        img_dir (str): Root directory to images.
        patient (bool): Flag to indicate if the img_dir is organised in patient directories.
    """
    # Get lung parenchyma image path
    if patient:
        lung_parenchyma_dir = os.path.join(os.path.dirname(img_dir), "CT_lung_parenchyma")
        patient_file = os.path.join(*image_path.split("/")[-3:])
    else:
        lung_parenchyma_dir = os.path.join(img_dir, "CT_lung_parenchyma")
        patient_file = os.path.join(*image_path.split("/")[-2:])
    output_path = os.path.join(lung_parenchyma_dir, patient_file)

    # Check if the processed image already exists
    if os.path.exists(output_path):
        return output_path

    # Extract lung parenchyma
    img_split = split_lung_parenchyma(image_path, 15599, -96)

    # Create the output directory if it does not exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the processed image
    cv2.imencode(".jpg", img_split)[1].tofile(output_path)
    # print(f"Processed and saved image to {output_path}")
    return output_path


# def extract_patient_lung_parenchyma(target_dir, output_dir):
#     """Process images in the target directory, apply lung parenchyma extraction, save results in output directory.

#     Args:
#         target_dir (str): Path to the directory containing target images.
#         output_dir (str): Path to the directory to save processed images.
#     """
#     target_list = [os.path.join(target_dir, file) for file in os.listdir(target_dir)]
#     for target in target_list:
#         # Extract lung parenchyma
#         img_split = split_lung_parenchyma(target, size=15599, thr=-96)
#         dst = target.replace(target_dir, output_dir)
#         dst_dir = os.path.split(dst)[0]
#         if not os.path.exists(dst_dir):
#             os.makedirs(dst_dir)
#         # Save the processed image
#         cv2.imencode(".jpg", img_split)[1].tofile(dst)
#     print(f"Target list done with {len(target_list)} items")


def split_lung_parenchyma(target, size, thr):
    """Perform lung parenchyma extraction using adaptive thresholding and morphological operations.

    Args:
        target (str): Path to the target image.
        size (int): Size parameter for adaptive thresholding.
        thr (int): Threshold parameter for adaptive thresholding.

    Returns:
        np.ndarray: Processed image with lung parenchyma extracted.
    """
    # Read the image in grayscale
    img = cv2.imdecode(np.fromfile(target, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    # Apply adaptive thresholding (convert grayscale to binary image)
    # adjusts thresholds locally, segment despite varying intensities and noise
    # -> clear separation of lung tissue from background and other structures
    img_thr = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, size, thr
    ).astype(np.uint8)
    img_thr = 255 - img_thr  # Invert the mask

    # Label connected components
    img_test = measure.label(img_thr, connectivity=1)
    props = measure.regionprops(img_test)  # Get properties of connected components

    # Find the largest connected component and create a mask
    areas = [prop.area for prop in props]
    ind_max_area = np.argmax(areas) + 1
    del_array = np.zeros(img_test.max() + 1)
    del_array[ind_max_area] = 1
    del_mask = del_array[img_test]
    img_new = img_thr * del_mask

    # Fill holes in the mask
    mask_fill = fill_water(img_new)
    img_new[mask_fill == 1] = 255
    img_new = 255 - img_new  # Invert the mask again

    # Perform connected component analysis to refine the mask
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img_new.astype(np.uint8))
    labels = np.array(labels, dtype=float)

    # Identify the most common labels (background and foreground)
    maxnum = Counter(labels.flatten()).most_common(3)
    maxnum = sorted([x[0] for x in maxnum])

    # Create a background mask
    background = np.zeros_like(labels)
    if len(maxnum) == 1:
        pass
    elif len(maxnum) == 2:
        background[labels == maxnum[1]] = 1
    else:
        background[labels == maxnum[1]] = 1
        background[labels == maxnum[2]] = 1

    # Apply the background mask to the image
    img_new[background == 0] = 0
    # Perform morphological operations to clean up the mask
    img_new = cv2.dilate(img_new, np.ones((5, 5), np.uint8), iterations=3)
    img_new = cv2.erode(img_new, np.ones((5, 5), np.uint8), iterations=2)
    img_new = cv2.morphologyEx(
        img_new,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)),
        iterations=2,
    )
    img_new = cv2.medianBlur(img_new.astype(np.uint8), 21)

    # Multiply the original image by the mask to extract the lung parenchyma
    img_out = img * img_new.astype(bool)

    return img_out


def fill_water(img):
    """Fill holes in a binary mask.

    Args:
        img (np.ndarray): Input binary image.

    Returns:
        np.ndarray: Mask after flood fill operations.
    """
    height, width = img.shape
    img_exp = np.zeros((height + 20, width + 20))
    img_exp[10:-10, 10:-10] = img

    masks = [np.zeros([height + 22, width + 22], np.uint8) for _ in range(4)]
    points = [(0, 0), (width + 19, height + 19), (width + 19, 0), (0, height + 19)]

    for mask, point in zip(masks, points):
        cv2.floodFill(np.float32(img_exp), mask, point, 1)

    mask = np.logical_or.reduce(masks)
    output = mask[1:-1, 1:-1][10:-10, 10:-10]

    return output

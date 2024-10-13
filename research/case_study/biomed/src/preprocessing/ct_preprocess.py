# ruff: noqa: E402
# ruff: noqa: I001

# ruff: noqa: F841
# REMOVE ABOVE!
import os
import sys
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image


# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.preprocessing.lung_parenchyma import extract_img_lung_parenchyma


def get_transforms(config, is_train=False):
    transform_list = []

    # Resize
    if config["resize"]:
        transform_list.append(transforms.Resize(config["resize"], interpolation=Image.BILINEAR))

    # Data augmentation for training
    if is_train and config["data_augmentation"]:
        p = config["crop_prob"]  # probability of applying the random crop transformation
        p2 = config["transform_prob"]  # probability of applying other transformations

        if "random_crop" in config["transform"]:
            transform_list.append(
                transforms.RandomApply(
                    [
                        transforms.RandomResizedCrop(
                            config["resize"], scale=(0.7, 1.0), interpolation=Image.BILINEAR
                        )
                    ],
                    p=p,
                )
            )
        if "rotate" in config["transform"]:
            transform_list.append(
                transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=p2)
            )
        if "brightness" in config["transform"]:
            transform_list.append(
                transforms.RandomApply([transforms.ColorJitter(brightness=0.1)], p=p2)
            )
        if "contrast" in config["transform"]:
            transform_list.append(
                transforms.RandomApply([transforms.ColorJitter(contrast=0.1)], p=p2)
            )
        if "horizontal_flip" in config["transform"]:
            transform_list.append(transforms.RandomHorizontalFlip(p=p2))
        if "gaussian_blur" in config["transform"]:
            transform_list.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=p2)
            )

    # Convert to Tensor
    transform_list.append(transforms.ToTensor())  # divides pixels by 255 -> [0, 1]

    # Normalize
    if config["normalise_pixels"] == "standardise":
        pixel_stats = pd.read_csv(
            f"research/case_study/biomed/datasets/iCTCF/pixel_stats_{config['resize'][0]}.csv"
        )
        mean, std = pixel_stats["mean"].values[0], pixel_stats["std"].values[0]
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def load_and_preprocess_ct_image(img_path, config, root_dir, is_train=True, log_wandb=False):
    """Read a CT image from a directory, crop margins, and resize.

    Args:
        img_path (str): Path to the directory containing the CT image.
        config (dict): Configuration dictionary.
        root_dir (str): Root directory to images.
        is_train (bool): Flag to indicate if the image is for training.
        log_wandb (bool): Flag to indicate if the image should be logged to WandB.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Extract lung parenchyma if specified
    if config["extract_lung_parenchyma"]:
        original_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img_path = extract_img_lung_parenchyma(img_path, root_dir, patient=False)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        # Load image
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        original_img = img

    # Crop margin if specified
    if config["crop_margin"]:
        img = crop_margin(img, threshold=50, min_size=100)

    # Convert to PIL Image
    img = Image.fromarray(img)

    # Get transformations
    transform = get_transforms(config, is_train)

    # Apply transformations, convert to tensor, and add channel dimension
    processed_img = transform(img)

    # WandB logging (TEST)
    # wandb.log({
    #     "original_image": wandb.Image(original_img),
    #     "processed_image": wandb.Image(ToPILImage(processed_img, mode="L")),
    #     "img_path": img_path,
    #     })

    return processed_img


# Code Reference: https://ngdc.cncb.ac.cn/ictcf/HUST-19.php
# Adapted: added docstrings, comments
# fixed cropping error ??


def crop_margin(img, threshold=30, min_size=100):
    """Crop the image to remove margins with values below a threshold.

    Args:
        img (np.ndarray): Input image as a NumPy array.

    Returns:
        np.ndarray: Cropped image.
    """
    img = np.asarray(img)
    (row, col) = img.shape
    row_top = 0
    row_down = 0
    col_top = 0
    col_down = 0

    axis1 = img.sum(axis=1)
    axis0 = img.sum(axis=0)

    # Find the top row with values above the threshold
    for r in range(0, row):
        if axis1[r] > threshold:
            row_top = r
            break

    # Find the bottom row with values above the threshold
    for r in range(row - 1, 0, -1):
        if axis1[r] > threshold:
            row_down = r
            break

    # Find the left column with values above the threshold
    for c in range(0, col):
        if axis0[c] > threshold:
            col_top = c
            break

    # Find the right column with values above the threshold
    for c in range(col - 1, 0, -1):
        if axis0[c] > threshold:
            col_down = c
            break

    # difference between count of rows and columns (margins with values above the threshold)
    a = row_down + 1 - row_top - (col_down + 1 - col_top)

    if a > 0:
        # case when the number of rows is greater than the number of columns
        w = row_down + 1 - row_top  # width of row range
        col_down = int((col_top + col_down + 1) / 2 + w / 2)  # center of the column range
        col_top = col_down - w  # update col_top to maintain the row width

        # Ensure col_top and col_down are within the image dimensions
        if col_top < 0:
            col_top = 0
            # col_down = col_top + w
            col_down = col_top + w - 1  # added
        elif col_down >= col:
            col_down = col - 1
            # col_top = col_down - w
            col_top = col_down - w + 1  # added
    else:
        # case when the number of columns is greater than or equal to the number of rows
        w = col_down + 1 - col_top
        row_down = int((row_top + row_down + 1) / 2 + w / 2)
        row_top = row_down - w

        # Ensure row_top and row_down are within the image dimensions
        if row_top < 0:
            row_top = 0
            # row_down = row_top + w
            row_down = row_top + w - 1  # added
        elif row_down >= row:
            row_down = row - 1
            # row_top = row_down - w
            row_top = row_down - w + 1  # added

    # Handle case where row_top equals row_down by setting a default crop region of 100x100 pixels
    # if row_top == row_down:
    if row_down - row_top < min_size:  # added
        row_top = 0
        row_down = 99
        col_top = 0
        col_down = 99

    # Crop image to specified region
    new_img = img[row_top : row_down + 1, col_top : col_down + 1]
    return new_img

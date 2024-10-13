# Spline Interpolated Zoom (SIZ) algorithm for processing of 3D volumetric images
# Reference: Uniformizing Techniques to Process CT scans with 3D CNNs for Tuberculosis Prediction
# GitHub: https://github.com/hasibzunair/uniformizing-3D/blob/master/1_data_process_clef19.ipynb

import torch
from scipy.ndimage import zoom


def spline_interpolated_zoom(img, target_depth=64):
    """Apply Spline Interpolated Zoom (SIZ) to a 3D volumetric image.

    Args:
        - img (torch.Tensor): Input 3D image of shape (W, H, D) where W is width, H is height, and D is the depth.
        - target_depth (int): The target number of slices in the output 3D image.

    Returns:
        - torch.Tensor: The 3D image after applying Spline Interpolated Zoom, with shape (W, H, target_depth).
    """
    current_depth = img.shape[-1]

    # Calculate the depth factor DF = target_depth / current_depth
    depth_factor = target_depth / current_depth

    # Convert to numpy if input is a torch tensor
    if isinstance(img, torch.Tensor):
        img = img.numpy()

    # Apply spline interpolation using the depth factor
    # order=3 cubic interpolation
    # mode='nearest' pads with the nearest value from the edge
    img_resampled = zoom(
        img, (1, 1, depth_factor), order=3, mode="nearest"
    )  # order=3 for spline interpolation

    # Convert back to torch tensor
    img_resampled = torch.tensor(img_resampled, dtype=torch.float32)

    return img_resampled

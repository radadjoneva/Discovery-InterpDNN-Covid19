# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.data.ct_dataset import CovidCTDataset, CovidOutcomeDataset


def get_config():
    from src.train import parse_args

    args = parse_args()
    return vars(args)


def get_mean_std_min_max(loader):
    num_pixels = 0
    sum_pixels = 0.0
    sum_squared_pixels = 0.0
    min_pixel = float("inf")
    max_pixel = float("-inf")

    for batch in loader:
        images = batch["input"]
        batch_size, _, height, width = images.shape
        num_pixels += batch_size * height * width
        sum_pixels += images.sum().item()
        sum_squared_pixels += (images**2).sum().item()
        min_pixel = min(min_pixel, images.min().item())
        max_pixel = max(max_pixel, images.max().item())

    mean = sum_pixels / num_pixels
    std = np.sqrt(sum_squared_pixels / num_pixels) - (mean**2)

    return mean, std, min_pixel, max_pixel


def get_pixel_stats(config, dataset_type):
    if dataset_type == "ct_images":
        train_dataset = CovidCTDataset(config, split="train")
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
    elif dataset_type == "ct_patient_cnn":
        cf_df = pd.read_csv(config["cleaned_cf_data"])
        patient_id_outcome_df = cf_df[["Patient ID", config["outcome"]]]
        train_dataset = CovidOutcomeDataset(config, patient_id_outcome_df, split="train")
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)

    mean, std, min_pixel, max_pixel = get_mean_std_min_max(train_loader)
    stats = {"mean": mean, "std": std, "min_pixel": min_pixel, "max_pixel": max_pixel}
    return stats


if __name__ == "__main__":
    config = get_config()
    config["resize"] = (224, 224)

    print("\nGetting pixel statistics for CT images...")
    print("Resize: ", config["resize"])
    assert (
        config["data_augmentation"] is False
    ), "Data augmentation must be disabled for pixel statistics calculation."
    assert (
        config["normalise_pixels"] is None
    ), "Pixel normalisation must be disabled for pixel statistics calculation."
    assert (
        config["extract_lung_parenchyma"] is False
    ), "Lung parenchyma extraction must be disabled."
    assert config["crop_margin"] is False, "Cropping margins must be disabled."

    ct_images_stats = get_pixel_stats(config, "ct_images")
    # ct_patient_stats = get_pixel_stats(config, "ct_patient")

    save_path = f"research/case_study/biomed/datasets/iCTCF/pixel_stats_{config['resize'][0]}.csv"
    df = pd.DataFrame([ct_images_stats])
    df.to_csv(save_path, index=False)
    print(f"Statistics saved to {save_path}")

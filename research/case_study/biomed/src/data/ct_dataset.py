# ruff: noqa: E402
# ruff: noqa: I001

import os
import re
import sys

import torch
from torch.utils.data import Dataset
import pandas as pd
import wandb

# from data.base_dataset.py import Dataset

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.utils.data_utils import split_data, select_even_slices
from src.utils.utils import read_img_dirs
from src.preprocessing.ct_preprocess import load_and_preprocess_ct_image
from src.preprocessing.cf_preprocess import merge_outcomes_and_filter_rows


class CovidCTDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        assert self.split in ["train", "val", "test"]

        self.root_dir = config["ct_img_dir"]
        self.transform = config["transform"]
        self.classes = config["ct_classes"]  # NiCT, pCT, nCT
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.ratio = config.get(
            "split_ratio", [0.7, 0.2, 0.1]
        )  # default split train:val:test = 7:2:1
        self.seed = config.get("seed", 42)  # default seed for reproducibility

        # Prepare and split data
        self.img_paths = self._prepare_and_split_data()

    def _prepare_and_split_data(self):
        all_paths = []
        for c in self.classes:
            img_paths = read_img_dirs([os.path.join(self.root_dir, c)])
            # Sort the image paths based on the number in the filename
            sorted_img_paths = sorted(img_paths, key=self._extract_number)  # is it necessary?
            # Add class information to each path
            sorted_img_paths_with_class = [(path, c) for path in sorted_img_paths]
            all_paths.extend(sorted_img_paths_with_class)

        # Split data into train, val, test sets
        # return self._split_data(all_paths)
        return split_data(all_paths, seed=self.seed, ratio=self.ratio, split=self.split)

    def _extract_number(self, filepath):
        """Extract the first sequence of digits in the filename."""
        numbers = re.findall(r"\d+", os.path.basename(filepath))
        return int(numbers[-1]) if numbers else 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Get the image, label and file path at the specified index."""
        img_path = self.img_paths[idx][0]

        # Read the image and apply transforms/ preprocessing
        image = load_and_preprocess_ct_image(
            img_path, self.config, self.root_dir, is_train=self.split == "train"
        )

        label = self.img_paths[idx][1]
        label_idx = self.class_to_idx[label]
        label_one_hot = torch.zeros(len(self.classes))
        label_one_hot[label_idx] = 1

        data = {"input": image, "label": label_one_hot, "path": img_path}

        return data

    def class_distribution(self):
        class_counts = {cls: 0 for cls in self.classes}
        for _, label in self.img_paths:
            class_counts[label] += 1
        total_samples = len(self.img_paths)
        class_proportions = {cls: count / total_samples for cls, count in class_counts.items()}
        return class_counts, class_proportions, total_samples


class CovidOutcomeDataset(Dataset):
    def __init__(self, config, df, split="train"):
        self.config = config
        self.split = split
        assert self.split in ["train", "val", "test"]

        self.img_dir = config["ct_patient_dir"]
        self.transform = config["transform"]
        self.outcome = config["outcome"]  # "morbidity" or "mortality"
        self.classes = config["covid_outcome_classes"]  # "Control", "Type I", "Type II"
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.ratio = config.get(
            "split_ratio", [0.7, 0.2, 0.1]
        )  # default split train:val:test = 7:2:1
        self.seed = config.get("seed", 42)  # default seed for reproducibility
        self.nb_imgs = config.get("k_imgs", 10)  # default top k images per patient

        # Prepare and split data (patient directories)
        self.patient_dirs = self._prepare_and_split_data()
        patient_ids = [
            int(re.findall(r"\d+", os.path.basename(patient))[0]) for patient in self.patient_dirs
        ]
        self.patient_df = pd.DataFrame({"Patient ID": patient_ids, "Directory": self.patient_dirs})

        # Merge outcomes and filter rows
        self.df = merge_outcomes_and_filter_rows(
            df.copy(),
            remove_suspected=config["remove_suspected"],
            undis_binary=config["undis_binary"],
            ct_required=config["ct_required"],
        )

        # Merge patient_df <- df based on Patient ID
        self.outcome_dir_df = pd.merge(self.patient_df, self.df, on="Patient ID", how="left")
        self.outcome_dir_df = self.outcome_dir_df.dropna(
            subset=[self.outcome]
        )  # Remove NaN (Suspected)
        self.outcome_dir_df.reset_index(drop=True, inplace=True)  # Reset index
        self.labels_df = self.outcome_dir_df[
            ["Patient ID", self.outcome]
        ]  # preserve all patient dirs order

        # Update patient_df an dir list after filtering
        self.patient_df = self.outcome_dir_df[["Patient ID", "Directory"]]
        self.patient_dirs = self.patient_df["Directory"].tolist()

        # Log patient_df to Wandb
        wandb.log({f"patient_df_{self.split}": wandb.Table(dataframe=self.patient_df)})

    def _prepare_and_split_data(self):
        patient_dirs = [os.path.join(self.img_dir, patient) for patient in os.listdir(self.img_dir)]

        # Split data into train, val, test sets
        return split_data(patient_dirs, seed=self.seed, ratio=self.ratio, split=self.split)

    def _extract_number(self, filepath):
        """Extract the first sequence of digits in the filename."""
        numbers = re.findall(r"\d+", os.path.basename(filepath))
        return int(numbers[-1]) if numbers else 0

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_id = self.patient_df["Patient ID"][idx]

        # Get label from the CSV file
        label = self.labels_df[self.outcome][idx]
        if label not in self.classes:
            print(f"Invalid label: {label}")
            print(f"Patient ID: {patient_id}")
            print(f"Index: {idx}")
        label_idx = self.class_to_idx[label]
        label_one_hot = torch.zeros(len(self.classes))
        label_one_hot[label_idx] = 1

        # Get 10 images from the patient CT directory
        ct_dir = os.path.join(patient_dir, "CT")
        img_paths = [
            os.path.join(ct_dir, img) for img in os.listdir(ct_dir) if img.endswith(".jpg")
        ]
        img_paths = sorted(img_paths, key=self._extract_number)

        # Select the k middle (default: 10) images
        # top_k_imgs = select_middle_images(img_paths, patient_id, top_k=self.nb_imgs)

        # Select the k even slices (default 10 images from 60% of slices in the middle)
        k_imgs = select_even_slices(img_paths, nb_imgs=self.nb_imgs)

        # Read images and apply transforms/ preprocessing
        images = []
        for img_path in k_imgs:
            # Read the image and apply transforms/ preprocessing
            image = load_and_preprocess_ct_image(
                img_path, self.config, self.img_dir, is_train=self.split == "train"
            )
            images.append(image)

        if self.config["single_channel"]:
            if len(images) > 11:
                # Concatenate images along the batch dimension
                images = torch.stack(images, dim=0)  # stack images to create a batch
                label_one_hot = label_one_hot.repeat(images.size(0), 1)
            else:
                # Concatenate images along the width dimension
                images = torch.cat(images, dim=2)
        else:
            images = torch.cat(images, dim=0)  # concatenate images along the channel dimension

        data = {
            "input": images,
            "label": label_one_hot,
            "patient_id": patient_id,
            "kimg_paths": k_imgs,
        }
        return data

    def class_distribution(self):
        class_counts = {cls: 0 for cls in self.classes}
        for label in self.labels_df[self.outcome]:
            if label in class_counts:
                class_counts[label] += 1
        total_samples = len(self.labels_df)
        class_proportions = {cls: count / total_samples for cls, count in class_counts.items()}
        return class_counts, class_proportions, total_samples


if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "split_ratio": [0.7, 0.2, 0.1],
        "ct_img_dir": "research/case_study/biomed/datasets/iCTCF/CT",
        "root_dir": "research/case_study/biomed/datasets/iCTCF/single_images",
        "classes": ["NiCT", "pCT", "nCT"],
        "extract_lung_parenchyma": True,
        "transform": None,  # if None, use load_and_preprocess_ct_image (crop_margin, resize to 200x200, to tensor)
        "crop_margin": True,
    }

    # CovidCT dataset (single images)
    train_dataset = CovidCTDataset(config, split="train")
    val_dataset = CovidCTDataset(config, split="val")
    test_dataset = CovidCTDataset(config, split="test")

    # Per Patient dataset
    # dataset = CTDataset(config)

    # Load data
    # dataset.load_data()

    # Preprocess data
    # dataset.preprocess_data()

    # Print dataset length
    print(f"Number of images in train dataset: {len(train_dataset)}")
    print(f"Number of images in val dataset: {len(val_dataset)}")
    print(f"Number of images in test dataset: {len(test_dataset)}")

    # Number of images in each class in each set
    for split, dataset in zip(["train", "val", "test"], [train_dataset, val_dataset, test_dataset]):
        print(f"\nImages per class in {split} dataset:")
        for c in config["classes"]:
            n = sum([1 for path, label in dataset.img_paths if label == c])
            print(f"{c}: {n}")

    print("Done!")

# ruff: noqa: E402
# ruff: noqa: I001

import os

# Set the environment variable for CuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.data.dataset_load import load_datasets
from src.data.eda import print_and_log_class_distribution
from src.utils.utils import restore_random_state


def load_datasets_and_initialize_loaders(config, init_random_states_path=None):
    seed = config["seed"]
    # Restore the random state used to load datasets
    if init_random_states_path:
        init_random_state = torch.load(init_random_states_path)
        restore_random_state(init_random_state)

    # Load datasets
    print(f"\nLoading datasets for model {config['model']} ...")
    print(f"Device: {config['device']}")
    train_dataset, val_dataset, test_dataset = load_datasets(config)

    # Print class distributions
    print_and_log_class_distribution(train_dataset, "Dataset", "train")
    print_and_log_class_distribution(val_dataset, "Dataset", "val")
    print_and_log_class_distribution(test_dataset, "Dataset", "test")

    # Worker initialization function for DataLoader
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        random.seed(seed + worker_id)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
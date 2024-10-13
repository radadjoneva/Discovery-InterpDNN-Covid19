import os
import random

import numpy as np
import torch

import wandb


def set_seed(seed):
    """Set the random seed for reproducibility.

    Args:
        seed (int): Random seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # might slow down performance!
    torch.use_deterministic_algorithms(True)


def restore_random_state(random_state):
    random.setstate(random_state["python_random_state"])
    np.random.set_state(random_state["numpy_random_state"])
    torch.set_rng_state(random_state["torch_random_state"])
    if torch.cuda.is_available() and "torch_cuda_random_state" in random_state:
        torch.cuda.set_rng_state(random_state["torch_cuda_random_state"])


def log_random_states(config, epoch=0):
    # Get random states
    random_states = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "torch_cuda_random_state": torch.cuda.get_rng_state()
        if torch.cuda.is_available()
        else None,
    }

    # Save the random states to a file
    random_state_path = os.path.join(
        config["save_model_path"], f"random_states_{wandb.run.name}.pth"
    )
    torch.save(random_states, random_state_path)

    # Log the random state file as an artifact in WandB
    artifact = wandb.Artifact(f"random_states_{wandb.run.name}", type="random_states")
    artifact.add_file(random_state_path)
    artifact.metadata["epoch"] = epoch
    wandb.run.log_artifact(artifact)

    print(f"Random states saved and logged to WandB: {random_state_path}")


def log_final_random_state(config):
    random_state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": torch.cuda.get_rng_state()
        if torch.cuda.is_available()
        else None,
    }

    # Save file and log to WandB
    state_path = os.path.join(config["save_model_path"], f"final_random_state_{wandb.run.name}.pth")
    torch.save(random_state, state_path)

    artifact = wandb.Artifact(f"final_random_state_{wandb.run.name}", type="final_random_state")
    artifact.add_file(state_path)
    wandb.run.log_artifact(artifact)

    print(f"Final random state saved and logged to WandB: {state_path}")

# Code Reference: https://ngdc.cncb.ac.cn/ictcf/HUST-19.php
# Adapted: added docstrings, comments

def read_img_dirs(target_dirs):
    """Read all image file paths from the specified directories.

    Args:
        target_dirs (list): List of directories to read image files from.

    Returns:
        list: List of file paths to the images.
    """
    file_list = []
    for target_dir in target_dirs:
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                file_list.append(os.path.join(root, file).replace("\\", "/"))
    return file_list


def is_number(s):
    """Check if a string can be converted to a number.

    Args:
        s (str): Input string.

    Returns:
        bool: True if the string can be converted to a number, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata

        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

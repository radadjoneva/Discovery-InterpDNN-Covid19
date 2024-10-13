import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_real_and_synthetic_data(real_data_dir, synthetic_data_dir, split):
    """Load real and synthetic data for a given split.
    
    Args:
        real_data_dir (str): Directory where the real data CSV files are located.
        synthetic_data_dir (str): Directory where the synthetic data CSV files are located.
        split (str): Which dataset split to load ('train', 'val', 'test').

    Returns:
        pd.DataFrame: real_data (standardized real data).
        pd.DataFrame: real_data_inv (inverse standardized real data).
        pd.DataFrame: synthetic_data_dict (dictionary with different types of synthetic data).
        pd.DataFrame: synthetic_data_inv_dict (dictionary with different types of inverse synthetic data).
    """
    
    # Load real data
    real_data = pd.read_csv(os.path.join(real_data_dir, f"input_features_{split}.csv"))
    real_data_inv = pd.read_csv(os.path.join(real_data_dir, f"input_features_{split}_inverse.csv"))

    # Load synthetic data
    synthetic_data_dict = {
        "detTrue": pd.read_csv(os.path.join(synthetic_data_dir, f"reconstruct_{split}_detTrue.csv")),
        "detFalse": pd.read_csv(os.path.join(synthetic_data_dir, f"reconstruct_{split}_detFalse.csv")),
        "sampled_std1": pd.read_csv(os.path.join(synthetic_data_dir, "sampled_synthetic_std1.csv")),
        "sampled_std3": pd.read_csv(os.path.join(synthetic_data_dir, "sampled_synthetic_std3.csv"))
    }

    synthetic_data_inv_dict = {
        "detTrue": pd.read_csv(os.path.join(synthetic_data_dir, f"inv_reconstruct_{split}_detTrue.csv")),
        "detFalse": pd.read_csv(os.path.join(synthetic_data_dir, f"inv_reconstruct_{split}_detFalse.csv")),
        "sampled_std1": pd.read_csv(os.path.join(synthetic_data_dir, "sampled_inv_synthetic_std1.csv")),
        "sampled_std3": pd.read_csv(os.path.join(synthetic_data_dir, "sampled_inv_synthetic_std3.csv"))
    }

    return real_data, real_data_inv, synthetic_data_dict, synthetic_data_inv_dict


def plot_real_vs_synthetic_distributions(real_data, synthetic_data, real_data_inv, synthetic_data_inv, feature_name, split, save_path=None):
    """Plot and save the distribution comparison between real and synthetic data."""
    
    # Ensure save_path directory exists if provided
    if save_path:
        os.makedirs(os.path.join(save_path, split), exist_ok=True)
        save_path = os.path.join(save_path, split)

    # Plot 1: Real range values comparison
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1a: Real data vs Synthetic data (Real Range)
    bin_edges = np.histogram_bin_edges(real_data_inv[feature_name], bins=30)
    sns.histplot(real_data_inv[feature_name], bins=bin_edges, kde=True, label="Real", color="blue", alpha=0.5, stat="count", ax=axs[0])
    sns.histplot(synthetic_data_inv[feature_name], bins=bin_edges, kde=True, label="Synthetic", color="orange", alpha=0.5, stat="count", ax=axs[0])

    # Customize the first plot
    axs[0].set_title(f"Real vs Synthetic (Real Range) for {feature_name} ({split} data)")
    axs[0].set_ylabel("Frequency", fontsize=14)
    axs[0].set_xlabel("Real range values", fontsize=14)
    axs[0].legend(fontsize=12)
    axs[0].tick_params(axis='both', which='major', labelsize=12)

    # Plot 1b: Real data vs Synthetic data (Standardized)
    bin_edges = np.histogram_bin_edges(real_data[feature_name], bins=30)
    sns.histplot(real_data[feature_name], bins=bin_edges, kde=True, label="Real", color="blue", alpha=0.5, stat="count", ax=axs[1])
    sns.histplot(synthetic_data[feature_name], bins=bin_edges, kde=True, label="Synthetic", color="orange", alpha=0.5, stat="count", ax=axs[1])

    # Customize the second plot
    axs[1].set_title(f"Real vs Synthetic (Standardised) for {feature_name} ({split} data)")
    axs[1].set_ylabel("Frequency", fontsize=14)
    axs[1].set_xlabel("Standardised values", fontsize=14)
    axs[1].legend(fontsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=12)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

    # Save the plot if a save_path is provided
    if save_path:
        file_name = f"{feature_name}_{split}_real_vs_synthetic.png".replace("/", "_")
        plt.savefig(os.path.join(save_path, file_name))
    plt.close()


if __name__ == "__main__":
    real_data_dir = "research/case_study/biomed/datasets/iCTCF/processed_cf/apricot-sweep-24"
    synthetic_data_dir = "research/case_study/biomed/models/VAE/tabular/volcanic-sweep-29"

    save_results_path = "research/case_study/biomed/results/interpretability/VAE/synthetic_vs_real_distributions/volcanic-sweep-29"

    for split in ['train', 'val', 'test']:
        # Load real and synthetic data for the split
        real_data, real_data_inv, synthetic_data_dict, synthetic_data_inv_dict = load_real_and_synthetic_data(real_data_dir, synthetic_data_dir, split)
        
        # Loop through features and plot real vs synthetic distributions for reconstructions
        for det_key in ['detTrue', 'detFalse']:
            reconstruction_save_path = os.path.join(save_results_path, "reconstruction", "deterministic" if det_key == "detTrue" else "non-deterministic")
            for feature_name in real_data.columns:
                plot_real_vs_synthetic_distributions(real_data, synthetic_data_dict[det_key], real_data_inv, synthetic_data_inv_dict[det_key], feature_name, split, save_path=reconstruction_save_path)
        
        if split == 'train':
            # Loop through features and plot real vs sampled synthetic distributions for std1 and std3
            for std_key in ['std1', 'std3']:
                sampled_save_path = os.path.join(save_results_path, "sampled", std_key)
                for feature_name in real_data.columns:
                    plot_real_vs_synthetic_distributions(real_data, synthetic_data_dict[f'sampled_{std_key}'], real_data_inv, synthetic_data_inv_dict[f'sampled_{std_key}'], feature_name, split, save_path=sampled_save_path)

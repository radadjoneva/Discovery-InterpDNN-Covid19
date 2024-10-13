# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)


def load_and_combine_data(data_dir, original_data_path, split='train'):
    """Load and combine data files for a given split (train, val, test). Get NaN mask for imputed values.

    Args:
        data_dir (str): The directory containing the preprocessed data files.
        original_data_path (str): The path to the cleaned data file before processing.
        split (str): The split to load ('train', 'val', 'test').

    Returns:
        pd.DataFrame: Combined DataFrame with patient IDs, input features, and target outcomes.
        pd.DataFrame: Combined DataFrame with patient IDs, inverse standardized input features, and target outcomes.
        pd.DataFrame: Mask of NaN values for imputed features.
    """
    assert split in ['train', 'val', 'test'], "Invalid split parameter. Use 'train', 'val', or 'test'."

    # Load the data files
    X = pd.read_csv(os.path.join(data_dir, f"input_features_{split}.csv"))
    X_inv = pd.read_csv(os.path.join(data_dir, f"input_features_{split}_inverse.csv"))
    Y = pd.read_csv(os.path.join(data_dir, f"target_outcomes_{split}.csv"))
    patient_ids = pd.read_csv(os.path.join(data_dir, f"patient_ids_{split}.csv"))

    # Combine dataframes (patient ids, input features, target outcomes)
    combined_data = pd.concat([patient_ids, X, Y], axis=1)
    combined_data_inv = pd.concat([patient_ids, X_inv, Y], axis=1)

    # Load the original data (before processing)
    original_data = pd.read_csv(original_data_path)
    original_data = original_data.set_index("Patient ID").loc[patient_ids.squeeze()].reset_index()  # extract current patient IDs rows

    # Concatenate additional original columns to the combined dataframes
    original_columns = original_data[["Computed tomography (CT)", "Mortality outcome"]]
    combined_data = pd.concat([combined_data, original_columns], axis=1)
    combined_data_inv = pd.concat([combined_data_inv, original_columns], axis=1)

    # Create a mask of NaN values to identify where the values were imputed
    num_cols = [col for col in combined_data.columns if col.startswith("num__")]  # numerical columns
    # Extract corresponding original column names
    original_cols = original_data.columns.to_list()
    idx_num_cols_in_original = [original_cols.index(col.split("num__")[-1]) for col in num_cols]
    num_cols_in_original = [col for ix, col in enumerate(original_cols) if ix in idx_num_cols_in_original]
    
    # Create a NaN mask for these numerical columns in original_data
    nan_mask_numeric = original_data[num_cols_in_original].isna()
    nan_mask_numeric.columns = ["num__" + col for col in nan_mask_numeric.columns]  # add "num__" prefix to match combined_data columns
    
    # Get NaN mask matching the size and columns of combined_data
    nan_mask = pd.DataFrame(False, index=combined_data.index, columns=combined_data.columns)
    nan_mask.update(nan_mask_numeric)

    return combined_data, combined_data_inv, nan_mask


def plot_feature_distributions(data, data_inv, nan_mask, split='train', save_path=None):
    """Plot and save the distribution of each feature in the dataset.
    
    Args:
        data (pd.DataFrame): The dataset containing features and targets (standardized).
        data_inv (pd.DataFrame): The dataset containing features in real range (inverse standardized).
        nan_mask (pd.DataFrame): Mask DataFrame indicating NaN positions in the original data.
        split (str): Which dataset split to use ('train', 'val', 'test').
        save_path (str): Directory path to save the plots. If None, plots will not be saved.
    """

    # Check for valid split parameter
    assert split in ['train', 'val', 'test'], "Invalid split parameter. Use 'train', 'val', or 'test'."

    # Ensure save_path directory exists if provided
    if save_path:
        os.makedirs(os.path.join(save_path, split), exist_ok=True)
        save_path = os.path.join(save_path, split)

    # Define colors for each class
    cmap = plt.get_cmap("tab10")
    # Extract target classes and create a mapping of class labels to rows
    class_labels = [col for col in data.columns if "Morbidity outcome" in col]
    class_data = {cls: data[cls].values for cls in class_labels}
    class_colors = {cls: cmap(i) for i, cls in enumerate(class_labels)}

    # Loop over each feature in data_inv
    for feature_name in data_inv.columns:
        if feature_name in ['Patient ID', 'Computed tomography (CT)', 'Mortality outcome']:
            continue
        
        # Extract original and standardized data for the feature
        original_data = data_inv[feature_name]
        standardized_data = data[feature_name]

        # Prepare for imputation indication using nan_mask
        imputed_mask = nan_mask[feature_name]

        # Plot 1: Original Values on the x-axis
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1a: Overall distribution
        # Plot the imputed values in grey
        # Define consistent bin edges based on the overall range of the original data
        bin_edges = np.histogram_bin_edges(original_data, bins=30)  # 30 bins with consistent edges
        sns.histplot(original_data[imputed_mask], bins=bin_edges, color="grey", alpha=0.5, stat="count", label="Imputed", ax=axs[0])
        sns.histplot(original_data, kde=True, label="Overall", bins=bin_edges, color="plum", alpha=0.5, stat="count", ax=axs[0])

        # Customize the first plot
        axs[0].set_title(f"Overall Distribution for {feature_name} ({split} data)")
        axs[0].set_ylabel("Frequency", fontsize=14)
        axs[0].set_xlabel("Real range values", fontsize=14)
        axs[0].legend(fontsize=12)
        axs[0].tick_params(axis='both', which='major', labelsize=12)

        
        # Plot 1b: Distribution per class
        for cls, class_values in class_data.items():
            sns.histplot(original_data[class_values == 1], kde=True, label=f"{cls.split('_')[-1]}", bins=bin_edges, color=class_colors.get(cls, "plum"), alpha=0.3, stat="count", ax=axs[1])

        # Customize the second plot
        axs[1].set_title(f"Distribution per Class for {feature_name} ({split} data)")
        axs[1].set_ylabel("Frequency", fontsize=14)
        axs[1].set_xlabel("Real range values", fontsize=14)
        axs[1].legend(fontsize=12)
        axs[1].tick_params(axis='both', which='major', labelsize=12)

        # plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

        # Overall plot title
        # plt.suptitle(f"Distribution of {feature_name} ({split} data)", fontsize=14)
        
        # Save the original plot if a save_path is provided
        if save_path:
            file_name = f"{feature_name}_{split}_distribution.png".replace("/", "_")
            plt.savefig(os.path.join(save_path, file_name))
        plt.close()
        

        # Plot 2: Standardized Values on the x-axis
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # 1 row, 2 columns

        # Define consistent bin edges based on the overall range of the original data
        bin_edges = np.histogram_bin_edges(standardized_data, bins=30)  # 30 bins with consistent edges
        # Plot 2a: Overall distribution for standardized values
        sns.histplot(standardized_data[imputed_mask], bins=bin_edges, color="grey", alpha=0.5, stat="count", label="Imputed", ax=axs[0])
        sns.histplot(standardized_data, kde=True, label="Overall", bins=bin_edges, color="plum", alpha=0.5, stat="count", ax=axs[0])

        # Customize the first subplot for standardized values
        axs[0].set_title(f"Overall Distribution for {feature_name} (Standardised) ({split} data)")
        axs[0].set_ylabel("Frequency", fontsize=14)
        axs[0].set_xlabel("Standardised Values", fontsize=14)
        axs[0].legend(fontsize=12)
        axs[0].tick_params(axis='both', which='major', labelsize=12)

        # Plot 2b: Distribution per class for standardized values
        for cls, class_values in class_data.items():
            sns.histplot(standardized_data[class_values == 1], kde=True, label=f"{cls.split('_')[-1]}", bins=bin_edges, color=class_colors.get(cls, "plum"), alpha=0.3, stat="count", ax=axs[1])

        # Customize the second subplot for standardized values
        axs[1].set_title(f"Distribution per Class for {feature_name} (Standardised) ({split} data)")
        axs[1].set_ylabel("Frequency", fontsize=14)
        axs[1].set_xlabel("Standardised Values", fontsize=14)
        axs[1].legend(fontsize=12)
        axs[1].tick_params(axis='both', which='major', labelsize=12)

        # plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
        
        # # Overall plot title
        # plt.suptitle(f"Distribution of {feature_name} ({split} data)", fontsize=14)

        # Save the standardized plot if a save_path is provided
        if save_path:
            file_name = f"{feature_name}_{split}_distribution_standardised.png".replace("/", "_")
            plt.savefig(os.path.join(save_path, file_name))
        plt.close()


if __name__ == "__main__":
    data_dir = "research/case_study/biomed/datasets/iCTCF/processed_cf/apricot-sweep-24"
    original_data_path = "research/case_study/biomed/datasets/iCTCF/cleaned_cf_data.csv"

    save_path = "research/case_study/biomed/results/eda/distributions"

    for split in ['train', 'val', 'test']:
        # Load and combine data for the split
        data, data_inv, nan_mask = load_and_combine_data(data_dir, original_data_path, split=split)
        # Plot the feature distributions
        plot_feature_distributions(data, data_inv, nan_mask, split=split, save_path=save_path)



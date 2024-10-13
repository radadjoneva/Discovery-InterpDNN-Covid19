# Code Reference: https://ngdc.cncb.ac.cn/ictcf/HUST-19.php
# Adapted: added docstrings, comments, slight chnages (?)
# Reference XAI Covid Ward et al.
"""In this section we will process the data. Since this data was scraped,
we need to remove some impurities in the text and tell the DataFrame which features are numerical.
We also want to encode categorical features and get rid of certain features that were added a posteriori
to the dataset (such as the patient's mortality outcome)."""

# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

import pandas as pd


# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.preprocessing.underlying_diseases import categorise_disease


# PREPROCESSING
def remove_symbols(df):
    """Remove specific symbols from the DataFrame and convert columns to numeric where possible."""
    df[df.columns[4]] = pd.to_numeric(df.iloc[:, 4].str.rstrip("°C")).values
    for i in range(10, len(df.columns)):
        try:
            df[df.columns[i]] = df[df.columns[i]].str.replace("<", "")
            df[df.columns[i]] = df[df.columns[i]].str.replace(">", "")
            df[df.columns[i]] = df[df.columns[i]].str.replace("↓", "")
            df[df.columns[i]] = df[df.columns[i]].str.replace("↑", "")
            df[df.columns[i]] = pd.to_numeric(df[df.columns[i]], errors="raise")
        except AttributeError:
            continue
    return df


def remove_nan_vars(df, percentage=80):
    """Remove columns from the DataFrame where the percentage of missing values exceeds a specified threshold."""
    percent_missing = df.isnull().sum() * 100 / len(df)
    remove_these_vars = percent_missing.loc[percent_missing > percentage].index
    df = df.drop(remove_these_vars, axis=1)
    return df


def merge_outcomes_and_filter_rows(df, remove_suspected=True, undis_binary=True, ct_required=True):
    """Merge categories of morbidity outcomes into broader categories and remove specific unwanted categories.

    This function merges 'Mild' and 'Regular' outcomes into 'Type I', and 'Severe' and 'Critically ill' outcomes into 'Type II'.
    It also renames 'Control (Community-acquired pneumonia)' to 'Control'.
    Optionally, it can remove rows with 'Suspected' or 'Suspected (COVID-19-confirmed later)' outcomes.
    Optionally, it can remove rows with missing CT data.
    Optionally, it can convert the 'Underlying diseases' column to binary categorical.

    Args:
        df (pandas.DataFrame): The input DataFrame with morbidity outcome categories to be merged and cleaned.
        remove_suspected (bool): Whether to remove rows with 'Suspected' or 'Suspected (COVID-19-confirmed later)' outcomes.
        undis_binary (bool): Whether to convert the 'Underlying diseases' column to binary categorical.
        ct_required (bool): Whether to remove rows with missing CT data.

    Returns:
        pandas.DataFrame: The DataFrame with merged and cleaned morbidity outcome categories.
    """
    # Merge 'Mild' and 'Regular' into 'Type I', and 'Severe' and 'Critically ill' into 'Type II'
    df["Morbidity outcome"] = df["Morbidity outcome"].replace(["Mild", "Regular"], "Type I")
    df["Morbidity outcome"] = df["Morbidity outcome"].replace(
        ["Severe", "Critically ill"], "Type II"
    )

    # Replace "Control (Community-acquired pneumonia)" with "Control"
    df["Morbidity outcome"] = df["Morbidity outcome"].replace(
        "Control (Community-acquired pneumonia)", "Control"
    )

    # Drop rows with "Suspected" or "Suspected (COVID-19-confirmed later)" outcomes
    if remove_suspected:
        df = df[
            ~df["Morbidity outcome"].isin(["Suspected", "Suspected (COVID-19-confirmed later)"])
        ]

    # Drop rows with missing CT data
    if ct_required and "Computed tomography (CT)" in df.columns:
        df = df.dropna(subset=["Computed tomography (CT)"])

    # Turn "Underlying diseases" to binary categorical
    if undis_binary and "Underlying diseases" in df.columns:
        df.loc[:, "Underlying diseases"] = df["Underlying diseases"].apply(
            lambda x: "No" if x == "No underlying disease" else "Yes"
        )
    elif "Underlying diseases" in df.columns:
        # Underlying diseases is multi-categorical
        def process_diseases(disease_string):
            diseases = disease_string.split(", ")
            categories = set()
            for disease in diseases:
                category = categorise_disease(disease.strip())
                categories.add(category)
            return list(categories)

        # Convert "Underlying diseases" to lists of categories
        df["Underlying diseases"] = df["Underlying diseases"].apply(process_diseases)

    return df


def get_save_path(config, split):
    """
    Construct the save path for the processed data based on the configuration.

    Args:
        config (dict): Configuration dictionary containing keys like 'cleaned_cf_data', 'split_value', 'normalization_value', 'ct_required', and 'undis_binary'.

    Returns:
        str: The constructed save path for the processed data.
    """
    # Extract the root directory from config["cleaned_cf_data"]
    root_dir = os.path.dirname(config["cleaned_cf_data"])

    # Start building the filename with "processed_cf_data"
    filename = "processed_cf_data"

    # Append the split value and normalisation value to the filename
    filename += f"_{split}_{config['normalisation']}"

    # Append "ct_required" if it is True
    if config.get("ct_required", False):
        filename += "_ct_required"

    # Append "undis_binary" if it is True
    if config.get("undis_binary", False):
        filename += "_undis_binary"

    # Add the file extension
    filename += ".csv"

    # Combine the root directory with the filename to form the full path
    full_path = os.path.join(root_dir, filename)

    return full_path


if __name__ == "__main__":
    # Preprocess the raw cf_data file
    config = {
        "raw_cf_data": "research/case_study/biomed/datasets/iCTCF/cf_data",
        "cleaned_cf_data": "research/case_study/biomed/datasets/iCTCF/cleaned_cf_data.csv",
        "outcome": "Morbidity outcome",
    }

    cf_data = pd.read_csv(config["raw_cf_data"])

    # PREPROCESSING
    # 1. Removing symbols from the numbers
    # 2. Removing variables with a large number of NaNs
    cf_data = remove_nan_vars(remove_symbols(cf_data))
    cf_data.rename(columns={cf_data.columns[0]: "Patient ID"}, inplace=True)

    # Save the cleaned data to a new file
    save_path = os.path.join(os.path.dirname(config["raw_cf_data"]), "cleaned_cf_data.csv")
    cf_data.to_csv(save_path, index=False)

    print("Done!")

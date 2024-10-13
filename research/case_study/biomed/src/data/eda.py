import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wandb

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)


def print_and_log_class_distribution(dataset, dataset_name, split):
    class_counts, class_proportions, total_samples = dataset.class_distribution()

    # Print the distribution nicely
    print(f"\nDataset | Split: {split}")
    print(f"Total samples: {total_samples}")
    for cls, count in class_counts.items():
        proportion = class_proportions[cls]
        print(f"Class: {cls}, Count: {count}, Proportion: {proportion:.2f}")

    # Log to WandB
    table_data = [[cls, count, class_proportions[cls]] for cls, count in class_counts.items()]
    table = wandb.Table(columns=["Class", "Count", "Proportion"], data=table_data)
    wandb.log({f"{dataset_name}_{split}_class_distribution": table})


def get_nb_images_per_patient():
    # Define the base directory path
    base_dir = "research/case_study/biomed/datasets/iCTCF/CT/"

    # Initialize variables to store the minimum and maximum number of images
    min_images = float("inf")
    max_images = float("-inf")

    all_num_images = []
    patient_subdirs = []

    # Iterate through each subdirectory in the base directory
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir, "CT")
        if os.path.isdir(subdir_path):
            num_images = len(os.listdir(subdir_path))
            # print(f"Patient ID: {subdir}, Number of images: {num_images}")
            all_num_images.append(num_images)
            patient_subdirs.append(subdir)

            # Update the minimum and maximum counts
            if num_images < min_images:
                min_images = num_images
            if num_images > max_images:
                max_images = num_images

    mean_num_images = sum(all_num_images) / len(all_num_images)
    std_num_images = (
        sum([(num_images - mean_num_images) ** 2 for num_images in all_num_images])
        / len(all_num_images)
    ) ** 0.5

    # Print the minimum and maximum number of images
    print(f"Minimum number of images: {min_images}")
    print(f"Maximum number of images: {max_images}")

    print(f"Mean number of images: {mean_num_images}")
    print(f"Standard deviation of number of images: {std_num_images}")

    # Plot the distribution of image sizes
    plt.figure(figsize=(10, 6))
    sns.histplot(all_num_images, bins=30, kde=True)
    plt.title("Distribution of CT slices per patient")
    plt.xlabel("Number of Images")
    plt.ylabel("Frequency")
    if not os.path.exists("research/case_study/biomed/results/eda"):
        os.makedirs("research/case_study/biomed/results/eda")
    plt.savefig("research/case_study/biomed/results/eda/image_distribution.png")


def analyse_nan_values(df, target_col, output_csv="nan_analysis.csv"):
    columns = df.columns
    nan_analysis_data = []

    for col in columns:
        nan_mask = df[col].isna()
        total_missing = sum(nan_mask)

        if total_missing == 0:
            continue

        percent_missing = (total_missing / len(df)) * 100

        outcome = {
            "Control": sum(df[target_col][~nan_mask] == "Control"),
            "Control %": sum(df[target_col][~nan_mask] == "Control") / sum(~nan_mask) * 100,
            "Type I": sum(df[target_col][~nan_mask] == "Type I"),
            "Type I %": sum(df[target_col][~nan_mask] == "Type I") / sum(~nan_mask) * 100,
            "Type II": sum(df[target_col][~nan_mask] == "Type II"),
            "Type II %": sum(df[target_col][~nan_mask] == "Type II") / sum(~nan_mask) * 100,
        }

        underlying_disease = (
            sum(df["Underlying diseases"][~nan_mask] != "No underlying disease")
            / sum(~nan_mask)
            * 100
        )
        high_body_temp = sum(df["Body temperature"][~nan_mask] >= 37.5) / sum(~nan_mask) * 100

        nan_analysis_data.append(
            {
                "Column": col,
                "Total Missing": total_missing,
                "Percentage Missing": percent_missing,
                "Control": outcome["Control"],
                "Control %": outcome["Control %"],
                "Type I": outcome["Type I"],
                "Type I %": outcome["Type I %"],
                "Type II": outcome["Type II"],
                "Type II %": outcome["Type II %"],
                "With Underlying Disease": underlying_disease,
                "With High Body Temperature": high_body_temp,
            }
        )

    nan_analysis_df = pd.DataFrame(nan_analysis_data)

    # Save the DataFrame to a CSV file
    nan_analysis_df.to_csv(output_csv, index=False)

    # Figure 1: Total Missing and Percentage Missing
    fig1, axes1 = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    nan_analysis_df.plot(x="Column", y="Total Missing", kind="bar", ax=axes1[0], color="skyblue")
    axes1[0].set_title("Total Missing Values by Column")
    axes1[0].set_ylabel("Total Missing")
    axes1[0].set_xticklabels(nan_analysis_df["Column"], rotation=45, ha="right", fontsize=8)

    nan_analysis_df.plot(
        x="Column", y="Percentage Missing", kind="bar", ax=axes1[1], color="orange"
    )
    axes1[1].set_title("Percentage Missing Values by Column")
    axes1[1].set_ylabel("Percentage Missing")
    axes1[1].set_xticklabels(nan_analysis_df["Column"], rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_csv.replace(".csv", "_missing.png"))
    plt.show()

    # Figure 2: Morbidity Outcome Distributionq
    fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    nan_analysis_df.plot(
        x="Column",
        y=["Control", "Type I", "Type II"],
        kind="bar",
        stacked=True,
        ax=ax2[0],
        color=["blue", "green", "red"],
    )
    ax2[0].set_title("Morbidity Outcome Distribution by Column")
    ax2[0].set_ylabel("Count")
    ax2[0].set_xticklabels(nan_analysis_df["Column"], rotation=45, ha="right", fontsize=8)

    nan_analysis_df.plot(
        x="Column",
        y=["Control %", "Type I %", "Type II %"],
        kind="bar",
        stacked=True,
        ax=ax2[1],
        color=["blue", "green", "red"],
    )
    ax2[1].set_title("Morbidity Outcome Distribution by Column")
    ax2[1].set_ylabel("Percentage")
    ax2[1].set_xticklabels(nan_analysis_df["Column"], rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_csv.replace(".csv", "_morbidity.png"))
    plt.show()

    # Figure 3: With Underlying Disease and With High Body Temperature
    fig3, axes3 = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    nan_analysis_df.plot(
        x="Column", y="With Underlying Disease", kind="bar", ax=axes3[0], color="purple"
    )
    axes3[0].set_title("With Underlying Disease by Column")
    axes3[0].set_ylabel("Percentage")
    axes3[0].set_xticklabels(nan_analysis_df["Column"], rotation=45, ha="right", fontsize=8)

    nan_analysis_df.plot(
        x="Column", y="With High Body Temperature", kind="bar", ax=axes3[1], color="pink"
    )
    axes3[1].set_title("With High Body Temperature by Column")
    axes3[1].set_ylabel("Percentage")
    axes3[1].set_xticklabels(nan_analysis_df["Column"], rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_csv.replace(".csv", "_underlying_temp.png"))
    plt.show()

    return nan_analysis_df


def analyse_outcome_diseases_temperature(df, target_col, output_csv="outcome_analysis.csv"):
    # Filter out rows where target_col is NaN
    df_filtered = df.dropna(subset=[target_col])

    # Calculate percentages for underlying diseases
    outcomes = ["Control", "Type I", "Type II"]
    disease_data = []
    temperature_data = []

    for outcome in outcomes:
        outcome_mask = df_filtered[target_col] == outcome

        total_outcome = sum(outcome_mask)
        no_disease = sum(
            (df_filtered["Underlying diseases"] == "No underlying disease") & outcome_mask
        )
        other_disease = total_outcome - no_disease
        high_temp = sum((df_filtered["Body temperature"] >= 37.5) & outcome_mask)
        low_temp = total_outcome - high_temp

        disease_data.append(
            {
                "Outcome": outcome,
                "No Underlying Disease": no_disease / total_outcome * 100,
                "Other Underlying Disease": other_disease / total_outcome * 100,
            }
        )

        temperature_data.append(
            {
                "Outcome": outcome,
                "High Temperature": high_temp / total_outcome * 100,
                "Low Temperature": low_temp / total_outcome * 100,
            }
        )

    disease_df = pd.DataFrame(disease_data)
    temperature_df = pd.DataFrame(temperature_data)

    # Save the DataFrames to a CSV file
    disease_df.to_csv(output_csv.replace(".csv", "_disease.csv"), index=False)
    temperature_df.to_csv(output_csv.replace(".csv", "_temperature.csv"), index=False)

    # Plotting the results
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    disease_df.plot(
        x="Outcome",
        y=["No Underlying Disease", "Other Underlying Disease"],
        kind="bar",
        stacked=True,
        ax=ax1,
        color=["blue", "red"],
    )
    ax1.set_title("Percentage of Patients with Underlying Diseases by Outcome")
    ax1.set_ylabel("Percentage")
    ax1.set_xticklabels(disease_df["Outcome"], rotation=0, ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_csv.replace(".csv", "_disease.png"))
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    temperature_df.plot(
        x="Outcome",
        y=["High Temperature", "Low Temperature"],
        kind="bar",
        stacked=True,
        ax=ax2,
        color=["orange", "green"],
    )
    ax2.set_title("Percentage of Patients with High Body Temperature by Outcome")
    ax2.set_ylabel("Percentage")
    ax2.set_xticklabels(temperature_df["Outcome"], rotation=0, ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_csv.replace(".csv", "_temperature.png"))
    plt.show()

    return disease_df, temperature_df


def visualise_and_save_correlation_heatmaps(train_file, val_file, test_file, results_dir):
    """ 
    Function to visualise and save correlation heatmaps for the train, validation,
    and test datasets, with customized font sizes for labels and title. It removes
    certain columns and reorders others.
    
    Args:
        train_file (str): Path to the training dataset CSV file.
        val_file (str): Path to the validation dataset CSV file.
        test_file (str): Path to the test dataset CSV file.
        results_dir (str): Directory where the results (heatmap images) will be saved.
    """

    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load the CSV files
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)

    # Dictionary to map dataset names to data
    datasets = {'train': train_data, 'val': val_data, 'test': test_data}

    # List of essential columns to retain
    essential_columns = [
        "cat_Mortality outcome_Deceased",
        "cat_Mortality outcome_Cured"
        "cat_Morbidity outcome_Control",
        "cat_Morbidity outcome_Type I",
        "cat_Morbidity outcome_Type II",
        "cat_undis-Cardiovascular",
        "cat_undis-Diabetes",
        "cat_undis-No underlying disease",
    ]

    # Iterate over each dataset, generate heatmap, and save the figure
    for dataset_name, data in datasets.items():
        # Remove columns containing "Patient ID", "Hospital", "CT", or "SARS"
        filtered_data = data.drop(columns=[col for col in data.columns if any(keyword in col for keyword in ["Patient ID", "Hospital", "CT", "SARS"])])

        # Remove all 'cat_undis' columns except the specified one
        filtered_data = filtered_data.drop(columns=[col for col in filtered_data.columns if "cat__undis" in col and col not in essential_columns])

        # Ensure the essential columns are kept if they are in the data
        filtered_data = filtered_data[[col for col in filtered_data.columns] + [col for col in essential_columns if col in filtered_data.columns]]

        # Reorder columns: move "Mortality" columns to the end, then "Morbidity" columns
        mortality_columns = [col for col in filtered_data.columns if "Mortality" in col]
        morbidity_columns = [col for col in filtered_data.columns if "Morbidity" in col]
        other_columns = [col for col in filtered_data.columns if col not in mortality_columns + morbidity_columns]

        # Create the new column order: other columns first, then mortality, then morbidity
        reordered_data = filtered_data[other_columns + mortality_columns + morbidity_columns]

        # Compute the correlation matrix
        correlation_matrix = reordered_data.corr()

        # Increase the figure size to accommodate large numbers of columns
        plt.figure(figsize=(20, 18))  # Increased size to accommodate more columns
        heatmap = sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', 
                              vmin=-1, vmax=1, cbar=True)
        
        # Customize the font sizes and rotation of the tick labels
        heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=12, rotation=45, ha="right")
        heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=12)

        # Set the title with the specified font size
        plt.title(f'Correlation Heatmap of {dataset_name.capitalize()} Data', fontsize=14)

        # Adjust layout to prevent label cropping
        plt.tight_layout()

        # Save the figure to the results directory
        figure_path = os.path.join(results_dir, f'{dataset_name}_correlation_heatmap.png')
        plt.savefig(figure_path)
        plt.close()

    print(f'Heatmaps saved to {results_dir}')


def visualise_specific_correlation_heatmap(data_file, results_dir):
    """ Function to visualise and save a specific correlation heatmap for the given dataset.
    
    Args:
        data_file (str): Path to the dataset CSV file.
        results_dir (str): Directory where the results (heatmap image) will be saved.
    """

    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load the CSV file
    data = pd.read_csv(data_file)

    # List of specific columns to retain
    specific_columns = [
        "num__Age",
        "num__Body temperature",
        "num__Erythrocyte sedimentation rate", "num__C-reactive protein", "num__Procalcitonin",
        "num__Lymphocyte count",
        "num__Neutrophil count",
        "num__CD3+ T cell", "num__CD4+ T cell", "num__CD8+ T cell", "num__CD4/CD8 ratio", 
        "num__B lymphocyte", "num__Natural killer cell",
        "num__Interleukin-4", "num__Interleukin-10", "num__Interleukin-2", 
        "num__Interleukin-6", "num__TNF-α", "num__IFN-γ",
        "num__Albumin", "num__Calcium",
        "cat__Morbidity outcome_Control", "cat__Morbidity outcome_Type I", 
        "cat__Morbidity outcome_Type II"
    ]

    # Filter data to keep only the specific columns
    filtered_data = data[specific_columns]

    # Compute the correlation matrix
    correlation_matrix = filtered_data.corr()

    plt.figure(figsize=(16, 12))

    # Plot the heatmap
    heatmap = sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', 
                          vmin=-1, vmax=1, cbar=True)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14, rotation=45, ha="right")
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)
    heatmap.figure.axes[-1].tick_params(labelsize=14)
    plt.tight_layout()

    figure_path = os.path.join(results_dir, 'specific_correlation_heatmap.png')
    plt.savefig(figure_path)
    plt.close()


def count_positive_values(file_path):
    df = pd.read_csv(file_path)
    
    # Filter for columns that start with 'cat__undis-'
    cat_columns = [col for col in df.columns if col.startswith('cat__undis-')]
    
    # Create a dictionary to store the counts and percentages
    results = {"Column": [], "Count": [], "Percentage": []}
    
    # Iterate through filtered columns and count 1's
    for col in cat_columns:
        count = df[col].sum()
        total = df[col].count()
        percentage = (count / total) * 100
        
        results["Column"].append(col)
        results["Count"].append(count)
        results["Percentage"].append(percentage)
    
    return pd.DataFrame(results)




if __name__ == "__main__":
    # config = {
    #     "raw_cf_data": "research/case_study/biomed/datasets/iCTCF/cf_data",
    #     "cleaned_cf_data_with_suspect": "research/case_study/biomed/datasets/iCTCF/cleaned_cf_data_with_suspect.csv",
    #     "cleaned_cf_data": "research/case_study/biomed/datasets/iCTCF/cleaned_cf_data.csv",
    #     # "preprocessed_cf_data": "research/case_study/biomed/datasets/iCTCF/preprocessed_cf_data_standard_undis_binary.csv",
    #     "outcome": "Morbidity outcome",
    #     "drop_columns": [
    #         "Patient ID",
    #         "Hospital",
    #         "SARS-CoV-2 nucleic acids",
    #         "Computed tomography (CT)",
    #         "Morbidity outcome",
    #         "Mortality outcome",
    #     ],  # drop columns for training
    #     "ct_required": False,
    #     "normalisation": "standardise",
    #     "undis_binary": True,
    # }

    # df = pd.read_csv(config["cleaned_cf_data"])
    # outcome = config["outcome"]

    # nan_analysis_df = analyse_nan_values(
    #     df, outcome, "research/case_study/biomed/results/eda/cf_nan_analysis.csv"
    # )

    # disease_df, temperature_df = analyse_outcome_diseases_temperature(df, outcome, 'research/case_study/biomed/results/eda/cf_undis_temp_analysis.csv')

    # Get the number of images per patient
    # num_imgs_df = get_nb_images_per_patient()


    # Correlations - Processed CF data (train/ val/ test)
    # data_dir = "research/case_study/biomed/datasets/iCTCF/"
    # visualise_and_save_correlation_heatmaps(
    #     os.path.join(data_dir, "processed_cf_data_train_standardise.csv"),
    #     os.path.join(data_dir, "processed_cf_data_val_standardise.csv"),
    #     os.path.join(data_dir, "processed_cf_data_test_standardise.csv"),
    #     "research/case_study/biomed/results/eda/"
    #     )
    
    # Selected CFs heatmap
    # visualise_specific_correlation_heatmap(os.path.join(data_dir, "processed_cf_data_train_standardise.csv"), "research/case_study/biomed/results/eda/")


    # Number of values for Underlying diseases columns
    data_dir = "research/case_study/biomed/datasets/iCTCF/"
    train_file = os.path.join(data_dir, "processed_cf_data_train_standardise.csv")
    val_file = os.path.join(data_dir, "processed_cf_data_val_standardise.csv")
    test_file = os.path.join(data_dir, "processed_cf_data_test_standardise.csv")

    train_results = count_positive_values(train_file)
    val_results = count_positive_values(val_file)
    test_results = count_positive_values(test_file)

    # Combine all results into one dataframe
    combined_results = pd.concat([train_results, val_results, test_results], keys=['Train', 'Validation', 'Test'], names=['Dataset'])

    output_dir = "research/case_study/biomed/results/eda/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cat_undis_positive_values.csv")
    combined_results.to_csv(output_file)
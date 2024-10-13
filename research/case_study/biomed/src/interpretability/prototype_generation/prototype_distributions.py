import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_prototype_distributions(control_df, type_i_df, type_ii_df, save_path=None):
    """
    Plots and saves the distribution comparison between Control, Type I, and Type II prototypes.
    
    Parameters:
    - control_df: DataFrame containing the control prototypes.
    - type_i_df: DataFrame containing the Type I prototypes.
    - type_ii_df: DataFrame containing the Type II prototypes.
    - save_path: Directory path where the plots will be saved (default is None).
    """
    
    # Filter the data for the last epoch (999)
    control_df = control_df[control_df['epoch'] == 999]
    type_i_df = type_i_df[type_i_df['epoch'] == 999]
    type_ii_df = type_ii_df[type_ii_df['epoch'] == 999]
    
    # Remove 'epoch' and 'prototype_number' columns
    control_df = control_df.drop(columns=['epoch', 'prototype_number'])
    type_i_df = type_i_df.drop(columns=['epoch', 'prototype_number'])
    type_ii_df = type_ii_df.drop(columns=['epoch', 'prototype_number'])

    # Plot distributions for each feature
    for feature in control_df.columns:
        plt.figure(figsize=(10, 6))

        # Calculate bin edges based on all data combined to ensure same bin width
        combined_data = np.concatenate([control_df[feature], type_i_df[feature], type_ii_df[feature]])
        bin_edges = np.histogram_bin_edges(combined_data, bins=30)
        
        # Plotting distributions for Control, Type I, and Type II
        sns.histplot(control_df[feature], bins=bin_edges, kde=True, color='blue', label='Control', stat="count", alpha=0.5)
        sns.histplot(type_i_df[feature], bins=bin_edges, kde=True, color='orange', label='Type I', stat="count", alpha=0.5)
        sns.histplot(type_ii_df[feature], bins=bin_edges, kde=True, color='green', label='Type II', stat="count", alpha=0.5)
        
        # Customize the plot
        plt.title(f'Distribution of {feature} across Prototypes', fontsize=16)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        
        # Save the plot if save_path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            feature = feature.replace("/", "_")
            plt.savefig(os.path.join(save_path, f'{feature}_distribution.png'))
        
        # Show plot
        # plt.show()
        plt.close()


def plot_prototype_distributions_in_single_figure(control_df, type_i_df, type_ii_df, features, figsize=(15, 15), save_path=None):
    """
    Plots and saves a single figure comparing the distribution of prototypes for Control, Type I, and Type II classes.

    Parameters:
    - control_df: DataFrame containing the control prototypes.
    - type_i_df: DataFrame containing the Type I prototypes.
    - type_ii_df: DataFrame containing the Type II prototypes.
    - features: List of features to plot.
    - figsize: Tuple specifying the size of the figure.
    - save_path: Path to save the figure (default is None).
    """
    
    # Filter the data for the last epoch (999)
    control_df = control_df[control_df['epoch'] == 999]
    type_i_df = type_i_df[type_i_df['epoch'] == 999]
    type_ii_df = type_ii_df[type_ii_df['epoch'] == 999]
    
    # Remove 'epoch' and 'prototype_number' columns
    control_df = control_df.drop(columns=['epoch', 'prototype_number'])
    type_i_df = type_i_df.drop(columns=['epoch', 'prototype_number'])
    type_ii_df = type_ii_df.drop(columns=['epoch', 'prototype_number'])

    # Calculate the number of rows and columns for the subplots
    num_features = len(features)
    num_cols = 3  # Define the number of columns
    num_rows = (num_features + num_cols - 1) // num_cols  # Calculate the number of rows

    # Create the figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case of multiple rows/columns

    # Plot distributions for each feature
    for i, feature in enumerate(features):
        if feature in control_df.columns:  # Only plot features that exist in the dataframe
            ax = axes[i]

            # Calculate bin edges based on all data combined to ensure same bin width
            combined_data = np.concatenate([control_df[feature], type_i_df[feature], type_ii_df[feature]])
            bin_edges = np.histogram_bin_edges(combined_data, bins=30)

            # Plotting distributions for Control, Type I, and Type II
            sns.histplot(control_df[feature], bins=bin_edges, kde=True, color='blue', label='Control', stat="count", alpha=0.5, ax=ax)
            sns.histplot(type_i_df[feature], bins=bin_edges, kde=True, color='orange', label='Type I', stat="count", alpha=0.5, ax=ax)
            sns.histplot(type_ii_df[feature], bins=bin_edges, kde=True, color='green', label='Type II', stat="count", alpha=0.5, ax=ax)

            # Customize the plot
            ax.set_title(f'{feature}', fontsize=16) 
            ax.set_xlabel('', fontsize=12) 
            ax.set_ylabel('Frequency', fontsize=14) 
            # ax.legend(loc='upper right', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout for spacing
    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path)
    
    # plt.show()
    plt.close()



def plot_correlation_matrix_per_class(control_df, type_i_df, type_ii_df, features=None, figsize=(12, 10), save_path=None):
    """
    Plots and saves the correlation matrix of the specified features for each class (Control, Type I, and Type II).
    
    Parameters:
    - control_df: DataFrame containing the control prototypes.
    - type_i_df: DataFrame containing the Type I prototypes.
    - type_ii_df: DataFrame containing the Type II prototypes.
    - features: List of features to include in the correlation matrix.
    - figsize: Size of the correlation matrix figure.
    - save_path: Path to save the figure (default is None).
    """
    
    if features is not None:
        control_df = control_df[features]
        type_i_df = type_i_df[features]
        type_ii_df = type_ii_df[features]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Correlation matrix for Control class
    sns.heatmap(control_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
    axes[0].set_title('Control Class Correlation Matrix')
    
    # Correlation matrix for Type I class
    sns.heatmap(type_i_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
    axes[1].set_title('Type I Class Correlation Matrix')
    
    # Correlation matrix for Type II class
    sns.heatmap(type_ii_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[2])
    axes[2].set_title('Type II Class Correlation Matrix')

    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path)
    # plt.show()
    plt.close()



if __name__ == "__main__":
    dir = "research/case_study/biomed/results/interpretability/prototypes/vae_tab_prototypes/"
    control_prototype_path = os.path.join(dir, "Control_inv_prototype_1000.csv")
    type_i_prototype_path = os.path.join(dir, "Type I_inv_prototype_1000.csv")
    type_ii_prototype_path = os.path.join(dir, "Type II_inv_prototype_1000.csv")

    control_df = pd.read_csv(control_prototype_path)
    type_i_df = pd.read_csv(type_i_prototype_path)
    type_ii_df = pd.read_csv(type_ii_prototype_path)

    save_path = os.path.join(dir, "prototype_distributions")
    os.makedirs(save_path, exist_ok=True)

    # plot_prototype_distributions(control_df, type_i_df, type_ii_df, save_path=save_path)

    # List of features to plot
    list_features = ["num__Age", "num__Body temperature", "num__Platelet count", "num__Eosinophil count", "num__Lymphocyte count",
                     "num__Neutrophil count", "num__Erythrocyte sedimentation rate", "num__C-reactive protein", "num__Procalcitonin", 
                     "num__D-Dimer", "num__Albumin/Globulin ratio", "num__Albumin", "num__Alkaline phosphatase", 
                     "num__Alanine aminotransferase", "num__Urea nitrogen", "num__Calcium", "num__Creatinine", 
                     "num__Potassium", "num__Magnesium", "num__Sodium", "num__Phosphorus", "num__CD3+ T cell", 
                     "num__CD4+ T cell", "num__CD8+ T cell", "num__B lymphocyte", "num__Natural killer cell", 
                     "num__CD4/CD8 ratio", "num__Interleukin-2", "num__Interleukin-4", "num__Interleukin-6", 
                     "num__Interleukin-10", "num__TNF-α", "num__IFN-γ"]

    save_path = os.path.join(dir, "prototype_distributions", "prototype_distributions_all_features.png")
    # plot_prototype_distributions_in_single_figure(control_df, type_i_df, type_ii_df, list_features, figsize=(20, 20), save_path=save_path)


    # Plot correlation matrix for each class separately
    plot_correlation_matrix_per_class(control_df, type_i_df, type_ii_df, features=list_features, save_path=os.path.join(dir, "correlation_per_class.png"))

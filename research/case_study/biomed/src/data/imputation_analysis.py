# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from missforest.missforest import MissForest

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)


def plot_distributions(df_before, df_after, columns, method_name, save_dir):
    for col in columns:
        plt.figure(figsize=(12, 6))
        plt.hist(df_before[col].dropna(), bins=30, alpha=0.5, label="Before Imputation")
        plt.hist(df_after[col].dropna(), bins=30, alpha=0.5, label="After Imputation")
        plt.title(f"Distribution of {col} - {method_name}", fontsize=16)
        plt.xlabel(col, fontsize=14)
        plt.xticks(fontsize=12)
        plt.ylabel("Frequency", fontsize=14)
        plt.yticks(fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        col_name = col.replace("/", "_").replace(" ", "_")
        if not os.path.exists(os.path.join(save_dir, method_name)):
            os.makedirs(os.path.join(save_dir, method_name), exist_ok=True)
        save_path = os.path.join(save_dir, method_name, f"{col_name}.png")
        plt.savefig(save_path)
        plt.close()


def impute_and_plot(cf_df, num_cols, cat_cols, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    df_original = cf_df.copy()

    # Define the configurations for each imputation method
    imputation_methods = {
        "Iterative_BayesianRidge": IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            random_state=42,
            initial_strategy="mean",
            n_nearest_features=None,
            tol=1e-3,
            verbose=2,
        ),
        "Iterative_RandomForest": IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            max_iter=20,
            random_state=42,
            initial_strategy="mean",
            n_nearest_features=None,
            tol=1e-3,
            verbose=2,
        ),
        "Iterative_KNeighbors": IterativeImputer(
            estimator=KNeighborsRegressor(n_neighbors=20),
            max_iter=10,
            random_state=42,
            initial_strategy="mean",
            n_nearest_features=None,
            tol=1e-3,
            verbose=2,
        ),
        "KNNImputer": KNNImputer(n_neighbors=20, weights="uniform"),
        "MissForest": MissForest(),
    }

    # Apply each imputation method and plot results
    for method_name, imputer in imputation_methods.items():
        # Impute the data
        if method_name == "MissForest":
            df_imputed = imputer.fit_transform(cf_df, categorical=cat_cols)
            df_imputed = pd.DataFrame(df_imputed, columns=cf_df.columns)
        else:
            df_imputed = imputer.fit_transform(cf_df[num_cols])
            df_imputed = pd.DataFrame(df_imputed, columns=num_cols)

        # Plot the distributions
        plot_distributions(df_original, df_imputed, num_cols, method_name, save_dir)
        print(f"Plots saved for method: {method_name}")


if __name__ == "__main__":
    # Load training data
    cf_df = pd.read_csv("research/case_study/biomed/datasets/iCTCF/cleaned_cf_data.csv")

    # Define directory to save plots
    save_dir = "research/case_study/biomed/results/eda/"

    # Define columns to drop
    drop_cols = [
        "Patient ID",
        "Hospital",
        "SARS-CoV-2 nucleic acids",
        "Computed tomography (CT)",
        "Morbidity outcome",
        "Mortality outcome",
    ]

    # Drop unnecessary columns
    cf_df = cf_df.drop(drop_cols, axis=1)

    # Separate numerical and categorical columns
    num_cols = [col for col in cf_df.select_dtypes(include=["float64", "int64"]).columns]
    cat_cols = [col for col in cf_df.columns if col not in num_cols]

    # Impute and plot distributions
    impute_and_plot(cf_df, num_cols, cat_cols, save_dir)

    print("Done!")

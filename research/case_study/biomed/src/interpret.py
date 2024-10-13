# ruff: noqa: E402
# ruff: noqa: I001

import os

# Set the environment variable for CuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import torch
import pandas as pd

import yaml

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import f1_score

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.utils.model_utils import load_pretrained_model
from src.utils.utils import restore_random_state, set_seed
from src.interpretability.global_feature_importance import compute_pdp_feature_importances, compute_permutation_feature_importances, compute_ale_feature_importances, plot_feature_importances, plot_pdp_grid


def run_feature_importance_analyses(model_pipeline, X, Y, X_real, config, results_dir, class_list, num_cols):
    # Create a directory to save the results if it does not exist
    os.makedirs(results_dir, exist_ok=True)
    analyses = ["ALE"]

    ### 1. Compute and plot PDP-based feature importances
    if "PDP" in analyses:
        pdp_importances = compute_pdp_feature_importances(model_pipeline, X, num_cols, grid_resolution=100, classes=class_list)

        pdp_results_path = os.path.join(results_dir, "PDP")
        os.makedirs(pdp_results_path, exist_ok=True)

        for class_label in class_list:
            plot_feature_importances(
                feature_importances=pdp_importances,
                class_label=class_label,
                title="PDP-Based Feature Importance for numeric features",
                importance_fraction=1.0,
                save_path=pdp_results_path
            )

        # Plot PDP grid for all numerical features
        plot_pdp_grid(
            model_pipeline, 
            X, 
            X_real,  # Use inverse transformed values for the x-axis (real value range)
            features=num_cols, 
            grid_resolution=100, 
            classes=class_list, 
            save_path=pdp_results_path
        )

    ### 2. Compute and plot permutation-based feature importances
    if "Permutation" in analyses:
        # Overall importances - F1 Score (macro average)
        permutation_results_path = os.path.join(results_dir, "Permutation")
        os.makedirs(permutation_results_path, exist_ok=True)

        permutation_importances_macro = compute_permutation_feature_importances(
            model_pipeline, 
            X_train, 
            Y_train, 
            scoring_fn=f1_score, 
            n_repeats=5, 
            classes=class_list, 
            mode="macro",
        )

        plot_feature_importances(
            feature_importances=permutation_importances_macro, 
            class_label="All Classes (Macro)", 
            title="Permutation-Based Feature Importance", 
            importance_fraction=1.0,
            classes=class_list,
            save_path=permutation_results_path
        )

        # One-vs-all importances
        permutation_importances_one_vs_all = compute_permutation_feature_importances(
            model_pipeline, 
            X_train, 
            Y_train, 
            scoring_fn=f1_score, 
            n_repeats=5, 
            classes=class_list, 
            mode="one-vs-all",
        )

        for class_label in class_list:
            plot_feature_importances(
                feature_importances=permutation_importances_one_vs_all[class_label], 
                class_label=class_label, 
                title="Permutation-Based Feature Importance", 
                importance_fraction=1.0,
                classes=class_list,
                save_path=permutation_results_path
            )
        
        # Type I & Type II vs Control importances
        permutation_importances_custom = compute_permutation_feature_importances(
            model_pipeline, 
            X_train, 
            Y_train, 
            scoring_fn=f1_score, 
            n_repeats=5, 
            classes=class_list, 
            mode="custom",
            custom_class_group=[1, 2]
        )

        plot_feature_importances(
            feature_importances=permutation_importances_custom, 
            class_label="Type I & Type II (vs Control)", 
            title="Permutation-Based Feature Importance", 
            classes=class_list,
            importance_fraction=1.0
        )

    ### 3. Compute and plot ALE-based feature importances
    if "ALE" in analyses:
        ale_results_path = os.path.join(results_dir, "ALE")
        os.makedirs(ale_results_path, exist_ok=True)
        
        ale_importances = compute_ale_feature_importances(
            model_pipeline, 
            X, 
            X_real, 
            Y, 
            classes=class_list, 
            proba=True, plot=True, 
            save_path=ale_results_path)

        for class_label in class_list:
            plot_feature_importances(
                feature_importances=ale_importances,
                class_label=class_label,
                title="ALE-Based Feature Importance",
                importance_fraction=1.0,
                proba=True,
                classes=class_list,
                save_path=ale_results_path
            )


def run_interp_analyses(model_path, config, X, Y, X_real, results_dir, eval_random_states_path=None, split="all"):
    # Set random seed for reproducibility (based on config)
    seed = config["seed"]
    set_seed(seed)

    if eval_random_states_path:
        eval_random_state = torch.load(eval_random_states_path)
        restore_random_state(eval_random_state)

    # Parameters 
    class_list = config["covid_outcome_classes"]
    # TODO check if it works with "cuda" device
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    num_cols = [col for col in X.columns if col.startswith("num")]
    input_dim = len(X.columns)

    # Load the model using skorch and create a scikit-learn pipeline
    skorch_model = load_pretrained_model(model_path, config, input_dim=input_dim, skorch=True)
    model_pipeline = Pipeline([
        ("preprocessor", None),
        ("to_tensor", FunctionTransformer(lambda x: torch.tensor(x.values, dtype=torch.float32))),  # convert DataFrame to tensor
        ("model", skorch_model),
    ])

    # Run the feature importance analyses
    run_feature_importance_analyses(model_pipeline, X, Y, X_real, config, results_dir, class_list, num_cols)



if __name__ == "__main__":
    # CF_DNN: apricot-sweep-24
    run_name = "apricot-sweep-24"
    run_id = "radadjoneva-icl/covid-outcome-classification/2sspqbzv"  # apricot-sweep-24
    model_dir = f"research/case_study/biomed/models/CF_DNN/{run_name}"
    model_path = os.path.join(model_dir, f"f1_cf_dnn_{run_name}.pth")
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    eval_random_states_path = os.path.join(model_dir, f"random_states_{run_name}-best-epoch.pth")

    # api = wandb.Api()
    # run = api.run(run_id)
    # config = run.config

    # Load the datasets (preprocessed)
    data_dir = f"research/case_study/biomed/datasets/iCTCF/processed_cf/{run_name}"
    X_train = pd.read_csv(os.path.join(data_dir, "input_features_train.csv"))
    Y_train = pd.read_csv(os.path.join(data_dir, "target_outcomes_train.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "input_features_val.csv"))
    Y_val = pd.read_csv(os.path.join(data_dir, "target_outcomes_val.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "input_features_test.csv"))
    Y_test = pd.read_csv(os.path.join(data_dir, "target_outcomes_test.csv"))
    # Load datasets (preprocessed & inverse standardised - real range values)
    X_train_real = pd.read_csv(os.path.join(data_dir, "input_features_train_inverse.csv"))
    X_val_real = pd.read_csv(os.path.join(data_dir, "input_features_val_inverse.csv"))
    X_test_real = pd.read_csv(os.path.join(data_dir, "input_features_test_inverse.csv"))

    # Combine the datasets
    X_all = pd.concat([X_train, X_val, X_test])
    Y_all = pd.concat([Y_train, Y_val, Y_test])
    X_all_real = pd.concat([X_train_real, X_val_real, X_test_real])

    # Directory to save all interpretability results
    results_dir = "research/case_study/biomed/results/interpretability"

    run_interp_analyses(model_path, config, X_all, Y_all, X_all_real, results_dir, eval_random_states_path=eval_random_states_path, split="all")

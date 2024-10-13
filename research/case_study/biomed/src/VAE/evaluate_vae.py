# ruff: noqa: E402
# ruff: noqa: I001

import os

# Set the environment variable for CuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import torch
import torch.nn as nn
import yaml
import pandas as pd

import wandb


# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.utils.utils import restore_random_state, set_seed
from src.utils.model_utils import load_pretrained_model
from src.evaluation.performance_metrics import process_outputs, compute_metrics
from src.evaluation.plotting import plot_confusion_matrix
from src.utils.evaluation_utils import print_performance


def evaluate_on_data(model, data, labels, target_classes, device="cpu", split="train", save_path="", type_data=" real"):
        with torch.no_grad():
            # Forward pass
            outputs = model(data.to(device))
            correct_predictions = (outputs.argmax(1) == labels.argmax(1)).sum().item()
            total_samples = labels.size(0)

            accuracy = correct_predictions / total_samples
            y_true, y_pred, probs = process_outputs([labels], [outputs])

            metrics = compute_metrics(y_true, y_pred, probs, target_classes, total_samples, split=split, best_model=type_data)
            metrics[f"{split}_Accuracy"] = accuracy
            metrics[f"{split}_Correct_Predictions"] = correct_predictions

            if isinstance(type_data, str):
                metrics = {f"{key}_{type_data}": value for key, value in metrics.items()}

            wandb.log(metrics)

            conf_key = (
                f"{split}_Confusion_Matrix_{type_data}"
                if isinstance(type_data, str)
                else f"{split}_Confusion_Matrix"
            )
            plot_confusion_matrix(metrics[conf_key], target_classes, split, save_path, best_model=type_data)
            return metrics


def evaluate_classifier_on_reconstructed_data(model_path, config, eval_random_states_path, all_real_data, all_labels, all_reconstructed_data, save_path=""):
    # Set random seed for reproducibility
    seed = config["seed"]
    set_seed(seed)

    # Restore the random state used to evaluate the model/ inference
    if eval_random_states_path:
        eval_random_state = torch.load(eval_random_states_path)
        restore_random_state(eval_random_state)
    
    # Load the pretrained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = all_real_data["train"].values.shape[1]
    classifier = load_pretrained_model(model_path, config, input_dim=input_dim)
    classifier.to(device)
    classifier.eval()
    target_classes = config["covid_outcome_classes"]

    # wandb.init(project="covid-vae-eval", config=config)

    # Evaluate the classifier on the synthetic and real data
    split_names = ["train", "val", "test"]
    all_metrics = {}
    for split in split_names:
        real_data = torch.tensor(all_real_data[split].values, device=config["device"], dtype=torch.float32)
        labels = torch.tensor(all_labels[split].values, device=config["device"], dtype=torch.long)
        reconstructed_data = torch.tensor(all_reconstructed_data[split].values, device=config["device"], dtype=torch.float32)

        print(f"\nEvaluating performance on REAL {split} data...")
        save_path_real = os.path.join(save_path, "real_metrics")
        real_metrics = evaluate_on_data(classifier, real_data, labels, target_classes, device=device, split=split, save_path=save_path_real, type_data=" real")
        # Print performance metrics
        print_performance(real_metrics, f"Real {split}")

        print(f"\nEvaluating performance on SYNTHETIC {split} data...")
        save_path_reconstructed = os.path.join(save_path, "reconstruct_metrics")
        reconstructed_metrics = evaluate_on_data(classifier, reconstructed_data, labels, target_classes, device=device, split=split, save_path=save_path_reconstructed, type_data=" reconstructed")
        print_performance(reconstructed_metrics, f"Synthetic {split}")
    
    # wandb.finish()

    return all_metrics


if __name__ == "__main__":
    # CF_DNN: apricot-sweep-24
    run_name = "apricot-sweep-24"
    run_id = "radadjoneva-icl/covid-outcome-classification/2sspqbzv"  # apricot-sweep-24
    model_dir = f"research/case_study/biomed/models/CF_DNN/{run_name}"
    model_path = os.path.join(model_dir, f"f1_cf_dnn_{run_name}.pth")
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    init_random_states_path = os.path.join(model_dir, f"random_states_{run_name}.pth")
    eval_random_states_path = os.path.join(model_dir, f"random_states_{run_name}-best-epoch.pth")

    # Load real datasets (preprocessed)
    data_dir = f"research/case_study/biomed/datasets/iCTCF/processed_cf/{run_name}"
    X_train = pd.read_csv(os.path.join(data_dir, "input_features_train.csv"))
    Y_train = pd.read_csv(os.path.join(data_dir, "target_outcomes_train.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "input_features_val.csv"))
    Y_val = pd.read_csv(os.path.join(data_dir, "target_outcomes_val.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "input_features_test.csv"))
    Y_test = pd.read_csv(os.path.join(data_dir, "target_outcomes_test.csv"))
    # Load synthetic datasets
    vae_run_name = "hearty-shadow-95"
    syn_data_dir = f"research/case_study/biomed/models/VAE/{vae_run_name}"
    X_train_synthetic = pd.read_csv(os.path.join(syn_data_dir, "synthetic_train.csv"))
    X_val_synthetic = pd.read_csv(os.path.join(syn_data_dir, "synthetic_val.csv"))
    X_test_synthetic = pd.read_csv(os.path.join(syn_data_dir, "synthetic_test.csv"))

    real_data_X = {
        "train": X_train,
        "val": X_val,
        "test": X_test,
    }

    labels_Y = {
        "train": Y_train,
        "val": Y_val,
        "test": Y_test
    }

    synthetic_data = {
        "train": X_train_synthetic,
        "val": X_val_synthetic,
        "test": X_test_synthetic,
    }

    evaluate_classifier_on_reconstructed_data(model_path, config, eval_random_states_path, real_data_X, labels_Y, synthetic_data, syn_data_dir) 

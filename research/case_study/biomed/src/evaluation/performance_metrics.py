# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import wandb

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.evaluation.plotting import plot_confusion_matrix


def process_outputs(all_labels, all_outputs):
    # All true labels and predicted outputs
    all_labels = torch.cat(all_labels)  # all true labels
    all_outputs = F.softmax(
        torch.cat(all_outputs), dim=1
    ).cpu()  # apply softmax to convert to probabilities
    # Convert one-hot encoded labels to class indices
    y_true = all_labels.argmax(axis=1).cpu().numpy()
    y_pred = all_outputs.argmax(axis=1).cpu().numpy()
    return y_true, y_pred, all_outputs


def compute_roc_auc(y_true, all_outputs, target_classes):
    """Compute the ROC AUC score for each class.

    True Positive Rate (TPR) vs. False Positive Rate (FPR)
    probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance

    Args:
        - y_true (np.ndarray): true labels
        - all_outputs (torch.Tensor): predicted outputs
        - target_classes (list): list of target classes
    """
    roc_auc = {}
    # roc_curves = {}
    for i, class_name in enumerate(target_classes):
        if len(set(y_true == i)) > 1:
            roc_auc[class_name] = roc_auc_score(
                (y_true == i).astype(int), all_outputs[:, i].detach().cpu().numpy()
            )
            # fpr, tpr, _ = roc_curve((y_true == i).astype(int), all_outputs[:, i])
            # roc_curves[class_name] = (fpr, tpr)
        else:
            roc_auc[class_name] = "N/A"
            # roc_curves[class_name] = ([], [])
    return roc_auc


def compute_metrics(y_true, y_pred, all_outputs, target_classes, total_samples, split, best_model=""):
    # weighted average: take into account class imbalances
    # TP / (TP + FP); higher -> minimise false positives (e.g. incorrctly diagnosed)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    # TP / (TP + FN);  higher -> minimise false negatives (e.g. missed diagnoses)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
    # 2 * (precision * recall) / (precision + recall); harmonic mean of precision and recall
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)

    # Compute confusion matrix: actual vs. predicted labels
    conf_matrix = confusion_matrix(
        y_true, y_pred, labels=range(len(target_classes)), normalize="true"
    )
    conf_matrix_df = pd.DataFrame(conf_matrix, index=target_classes, columns=target_classes)

    # Compute ROC AUC Score for each class
    # True Positive Rate (TPR) vs. False Positive Rate (FPR)
    roc_auc = compute_roc_auc(y_true, all_outputs, target_classes)
    # Compute overall ROC AUC Score if all classes have ROC AUC Scores
    overall_roc_auc = "N/A"
    if all(auc != "N/A" for auc in roc_auc.values()):
        overall_roc_auc = roc_auc_score(
            y_true, all_outputs.detach().cpu().numpy(), multi_class="ovr"
        )

    metrics = {
        f"{split}_Precision": precision,
        f"{split}_Recall": recall,
        f"{split}_F1_Score": f1,
        f"{split}_ROC_AUC": roc_auc,
        f"{split}_Overall_ROC_AUC": overall_roc_auc,
        f"{split}_Confusion_Matrix": conf_matrix_df,
    }

    # Log PR and ROC curves
    wandb.log(
        {
            f"{split}_pr_curve"+best_model: wandb.plot.pr_curve(
                y_true,
                all_outputs.detach().cpu(),
                labels=target_classes,
                title=f"{split} Precision v. Recall"+best_model,
            )
        }
    )
    wandb.log(
        {
            f"{split}_roc_curve"+best_model: wandb.plot.roc_curve(
                y_true, all_outputs.detach().cpu(), labels=target_classes, title=f"{split} ROC"+best_model
            )
        }
    )

    # Log confusion matrix
    cm = wandb.plot.confusion_matrix(
        y_true=y_true, preds=y_pred, class_names=target_classes, title=f"{split} Confusion Matrix" + best_model
    )
    wandb.log({f"{split}_conf_matrix"+best_model: cm})

    return metrics


def evaluate_performance(
    config, model, data_loader, criterion, target_classes, split="test", plots=True, best_model=None
):
    """Evaluate the performance of the model on the given data_loader.

    Calculate the loss, accuracy, precision, recall, F1 score, ROC AUC, and confusion matrix of the model on the given data_loader.
    Save results into a CSV file and plot the ROC curve if specified.

    Args:
        - config (dict): dictionary containing configuration parameters
        - model (torch.nn.Module): the model to evaluate
        - data_loader (torch.utils.data.DataLoader): the data loader to evaluate the model on
        - criterion (torch.nn.Module): the loss function to use
        - target_classes (list): list of target classes
        - split (str): the split to evaluate the model on (default: "test")
        - plots (bool): whether to create plots (default: True)
        - best_model (str): "best_val_loss", "best_val_auc" or "best_val_f1" (default: None)

    Returns:
        - results (dict): dictionary containing the performance metrics
    """
    device = config["device"]
    save_path = os.path.join(config["save_model_path"], config["model"] + "_" + split)

    model.eval()
    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    all_labels, all_outputs = [], []

    with torch.no_grad():
        for batch in data_loader:
            if batch["input"].dim() == 5:  # Check if input has an extra batch dimension
                inputs = batch["input"].squeeze(0)
            else:
                inputs = batch["input"]
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = (inputs[0].to(device), inputs[1].to(device))
            else:
                inputs = inputs.to(device)

            if batch["label"].dim() == 3:  # Check if label has an extra batch dimension
                labels = batch["label"].squeeze(0).to(device)
            else:
                labels = batch["label"].to(device)

            outputs = model(inputs)
            if split == "train":
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct_predictions += (outputs.argmax(1) == labels.argmax(1)).sum().item()
            total_samples += labels.size(0)

            all_labels.append(labels)
            all_outputs.append(outputs)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    y_true, y_pred, all_outputs = process_outputs(all_labels, all_outputs)

    metrics = compute_metrics(y_true, y_pred, all_outputs, target_classes, total_samples, split, best_model=best_model)
    metrics[f"{split}_Loss"] = avg_loss
    metrics[f"{split}_Accuracy"] = accuracy
    metrics[f"{split}_Correct_Predictions"] = correct_predictions

    # Check if `best_model` is a string and not None, and append it to each key
    if isinstance(best_model, str):
        metrics = {f"{key}_{best_model}": value for key, value in metrics.items()}

    wandb.log(metrics)

    if plots:
        # plot_roc_curve(metrics["ROC_Curves"], metrics["ROC_AUC"], target_classes, save_path)
        conf_key = (
            f"{split}_Confusion_Matrix_{best_model}"
            if isinstance(best_model, str)
            else f"{split}_Confusion_Matrix"
        )
        plot_confusion_matrix(metrics[conf_key], target_classes, split, save_path, best_model)

    # save_metrics_to_csv(metrics, save_path)
    return metrics

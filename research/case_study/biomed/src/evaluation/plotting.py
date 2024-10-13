# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

import wandb

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)


def plot_roc_curve(roc_curves, roc_auc, target_classes, save_path):
    plt.figure()
    for class_name in target_classes:
        if roc_auc[class_name] != "N/A":
            fpr, tpr = roc_curves[class_name]
            plt.plot(fpr, tpr, label=f"{class_name} (area = {roc_auc[class_name]:0.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plot_roc_path = os.path.join(save_path + "_" + "roc_curve.png")
    plt.savefig(plot_roc_path)
    plt.close()
    wandb.log({"roc_curve": wandb.Image(plot_roc_path)})


def plot_confusion_matrix(conf_matrix, target_classes, split, save_path, best_model=None):
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=target_classes,
        yticklabels=target_classes,
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)

    # Append best_model to the title if it's a string
    title_suffix = f" ({best_model})" if isinstance(best_model, str) else ""
    plt.title(f"Confusion Matrix {split}{title_suffix}", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    file_suffix = f"_{best_model}" if isinstance(best_model, str) else ""
    plot_confusion_path = os.path.join(save_path + f"{split}_confusion_matrix{file_suffix}.png")
    plt.savefig(plot_confusion_path)
    plt.close()
    wandb.log({f"{split}_confusion_matrix{file_suffix}": wandb.Image(plot_confusion_path)})

# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys
import torch
import wandb
from skorch import NeuralNetClassifier

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.model.cf_dnn import DNNSimple
from src.model.resnet import ResNet50Adapted


def load_pretrained_model(
    model_path,
    config,
    input_dim=None,
    skorch=False,
):
    """Load a pretrained model from a given path.

    Args:
        - model_path (str): The path to the saved model's state_dict.
        - config (dict): Configuration dictionary.
        - input_dim (int, optional): Input dimension (for models like DNN).
        - skorch (bool, optional): Whether to load the model as a skorch model.

    Returns:
        - model (nn.Module): The loaded model.
    """
    # Parameters
    num_classes = len(config["covid_outcome_classes"])
    num_channels = config["k_imgs"]
    dropout = config["dropout"]
    pretrained = config["pretrained"]
    batch_norm = config["batch_norm"]

    if config["model"] == "cf_dnn":
        if not skorch:
            model = DNNSimple(input_dim, dropout=dropout, batch_norm=batch_norm)
        else:
            # Wrap the DNNSimple model with NeuralNetClassifier
            net = NeuralNetClassifier(
                DNNSimple,
                module__input_dim=input_dim,
                module__dropout=dropout,
                module__batch_norm=batch_norm,
                classes=config["covid_outcome_classes"],
            )
            net.initialize()

    elif config["model"] == "resnet50":
        if config["single_channel"]:
            num_channels = 1
        model = ResNet50Adapted(num_channels, num_classes, pretrained)

    # Load the state_dict into the model
    state_dict = torch.load(model_path, map_location=config["device"])
    if not skorch:
        model.load_state_dict(state_dict, strict=True)
        model.to(config["device"])
        model.eval()
        return model
    else:
        net.module_.load_state_dict(state_dict, strict=True)
        net.module_.to(config["device"])
        net.module_.eval()
        return net


def log_to_terminal(
    epoch,
    num_epochs,
    epoch_loss,
    epoch_acc,
    val_loss,
    val_acc,
    train_auc,
    val_auc,
    overall_val_auc,
    val_f1_score,
):
    print(f"{'-'*40}")
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"{'-'*40}")
    print(f"Train Loss: {epoch_loss:.4f}")
    print(f"Train Acc: {epoch_acc:.4f}")
    print(f"Train AUC: {train_auc}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    print(f"Val AUC: {val_auc}")
    print(f"Overall Val AUC: {overall_val_auc:.4f}")
    print(f"Val F1 Score: {val_f1_score:.4f}")
    print(f"{'-'*40}\n")


def plot_loss_curves(history, num_epochs, title="Model"):
    # Extract data for the plots
    epochs = list(range(1, num_epochs + 1))

    # Plot and log accuracy curves
    wandb.log(
        {
            f"{title}_accuracy": wandb.plot.line_series(
                xs=epochs,
                ys=[history["train_acc"], history["val_acc"]],
                keys=["Train Accuracy", "Validation Accuracy"],
                title=f"{title} Accuracy Over Epochs",
                xname="Epoch",
            )
        }
    )

    # Plot and log loss curves
    wandb.log(
        {
            f"{title}_loss": wandb.plot.line_series(
                xs=epochs,
                ys=[history["train_loss"], history["val_loss"]],
                keys=["Train Loss", "Validation Loss"],
                title=f"{title} Loss Over Epochs",
                xname="Epoch",
            )
        }
    )


# def get_custom_head(head_layers):
#     layers = []
#     for layer in head_layers:
#         if layer["type"] == "linear":
#             layers.append(nn.Linear(layer["in_features"], layer["out_features"]))
#         elif layer["type"] == "relu":
#             layers.append(nn.ReLU())
#         elif layer["type"] == "leaky_relu":
#             layers.append(nn.LeakyReLU())
#         else:
#             raise ValueError(f"Layer type {layer['type']} not supported")

#     return nn.Sequential(*layers)


# def initialise_linear_layers(layers, std=2e-5):
#     if isinstance(layers, nn.Linear):
#         trunc_normal_(layers.weight, std=std)
#     elif isinstance(layers, nn.Sequential):
#         for layer in layers:
#             if isinstance(layer, nn.Linear):
#                 trunc_normal_(layer.weight, std=std)


# def validate_input(input_data, expected_shape):
#     # Check if input_data is of the right shape and type
#     if not isinstance(input_data, torch.Tensor):
#         raise TypeError("Input data must be a torch.Tensor")

#     if input_data.shape != expected_shape:
#         raise ValueError(f"Input data must have shape {expected_shape}, but got {input_data.shape}")

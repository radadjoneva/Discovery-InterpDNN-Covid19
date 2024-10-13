# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

import torch
import numpy as np

import wandb
from sklearn.metrics import f1_score

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.utils.model_utils import plot_loss_curves
from src.utils.model_utils import log_to_terminal
from src.evaluation.performance_metrics import (
    evaluate_performance,
    compute_roc_auc,
    process_outputs,
)
from src.utils.evaluation_utils import print_performance
from src.utils.utils import log_random_states


class ModelTrainer:
    def __init__(self, config, model, criterion, optimizer, train_loader, val_loader):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = config["num_epochs"]
        self.device = config["device"]
        # Move the model to the device
        self.model.to(self.device)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_auc": [],
            "val_auc": [],
        }

        self.target_classes = self.train_loader.dataset.classes

        self.save_path = config["save_model_path"]
        os.makedirs(self.save_path, exist_ok=True)  # Create the directory if it doesn't exist

        # Initialise the learning rate scheduler if specified in config
        self.scheduler = None
        if config["lr_scheduler"] == "plateau":
            # ReduceLROnPlateau
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config["lr_factor"],
                patience=self.config["lr_patience"],
                verbose=True,
            )
        elif config["lr_scheduler"] == "step":
            # StepLR
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config["lr_step_size"], gamma=0.1
            )
        elif config["lr_scheduler"] == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.config["lr_gamma"]
            )

    def train(self):
        # Weights, gradients and parameters logging
        wandb.watch(self.model, log="all", log_freq=10)

        # Initialise best AUC score
        best_val_auc = 0.0
        lowest_val_loss = float("inf")  # Lowest validation loss observed
        highest_f1_score = 0.0

        for epoch in range(self.num_epochs):
            self.model.train()

            # Log random states
            log_random_states(self.config, epoch + 1)

            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            all_labels = []
            all_outputs = []

            for batch in self.train_loader:
                inputs = batch["input"]
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
                else:
                    inputs = inputs.to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                # forward pass
                outputs = self.model(inputs)
                # compute loss
                loss = self.criterion(outputs, labels)
                # backward pass
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * labels.size(0)
                correct_predictions += (outputs.argmax(1) == labels.argmax(1)).sum().item()
                total_samples += labels.size(0)

                # Collect true labels and predicted probabilities for AUC
                all_labels.append(labels)
                all_outputs.append(outputs)

                # Log wandb running loss
                wandb.log({"running_loss": loss.item()})

            epoch_loss = running_loss / total_samples
            epoch_acc = correct_predictions / total_samples

            # Calculate AUC for training set
            y_true, y_pred, all_outputs_processed = process_outputs(all_labels, all_outputs)
            train_auc = compute_roc_auc(y_true, all_outputs_processed, self.target_classes)

            # Evaluate the model on the validation set
            val_loss, val_acc, val_auc, val_f1 = self.validate()

            # Convert val_auc dictionary to a list of scores, replacing NaN with zero
            val_auc_values = [val_auc.get(cls, 0.0) for cls in self.target_classes]
            val_auc_values = [0.0 if np.isnan(auc) else auc for auc in val_auc_values]
            # Calculate the overall AUC score for validation set
            overall_val_auc = np.mean(val_auc_values)

            # Check if the current AUC or loss or f1 is the best we've seen so far
            if (
                overall_val_auc > best_val_auc
                or val_loss < lowest_val_loss
                or val_f1 > highest_f1_score
            ):
                if overall_val_auc > best_val_auc:
                    best_val_auc = overall_val_auc
                    print(f"New best AUC: {best_val_auc:.4f}")
                    self.save_best_model("auc", best_val_auc, epoch, val_loss, val_acc, overall_val_auc, val_f1)

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    print(f"New lowest validation loss: {lowest_val_loss:.4f}")
                    self.save_best_model("loss", lowest_val_loss, epoch, val_loss, val_acc, overall_val_auc, val_f1)

                if val_f1 > highest_f1_score:
                    highest_f1_score = val_f1
                    print(f"New highest F1 score: {highest_f1_score:.4f}")
                    self.save_best_model("f1", highest_f1_score, epoch, val_loss, val_acc, overall_val_auc, val_f1)

            # Logging to terminal
            log_to_terminal(
                epoch,
                self.num_epochs,
                epoch_loss,
                epoch_acc,
                val_loss,
                val_acc,
                train_auc,
                val_auc,
                overall_val_auc,
                val_f1,
            )

            if epoch % 10 == 0:
                # Get all metrics on the validation set
                val_metrics = evaluate_performance(
                    self.config,
                    self.model,
                    self.val_loader,
                    self.criterion,
                    self.target_classes,
                    split="val",
                    plots=True,
                )
                print_performance(val_metrics, "Validation")

            # Logging to wandb
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "train_acc": epoch_acc,
                    "train_auc": train_auc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "overall_val_auc": overall_val_auc,
                    "val_f1_score": val_f1,
                }
            )

            # Update history
            self.history["train_loss"].append(epoch_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(epoch_acc)
            self.history["val_acc"].append(val_acc)
            self.history["train_auc"].append(train_auc)
            self.history["val_auc"].append(val_auc)
            self.history["overall_val_auc"] = overall_val_auc
            self.history["val_f1_score"] = val_f1

            # Update the learning rate scheduler
            if self.scheduler is not None:
                if self.config["lr_scheduler"] == "plateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        # Plot loss curves
        plot_loss_curves(self.history, self.num_epochs, title=self.config["model"])

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["input"]
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
                else:
                    inputs = inputs.to(self.device)
                labels = batch["label"].to(self.device)
                # forward pass & compute loss
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                correct_predictions += (outputs.argmax(1) == labels.argmax(1)).sum().item()
                total_samples += labels.size(0)

                # Collect true labels and predicted probabilities for AUC
                all_labels.append(labels)
                all_outputs.append(outputs)

        # average validation loss
        val_loss = val_loss / total_samples
        val_acc = correct_predictions / total_samples

        # Calculate AUC for validation set
        y_true, y_pred, all_outputs_processed = process_outputs(all_labels, all_outputs)
        val_auc = compute_roc_auc(y_true, all_outputs_processed, self.target_classes)

        # Calculate F1 score
        val_f1_score = f1_score(y_true, y_pred, average="macro")
        return val_loss, val_acc, val_auc, val_f1_score


    def save_best_model(self, metric_name, metric_value, epoch, val_loss, val_acc, overall_val_auc, val_f1):
        model_path = os.path.join(self.save_path, f"{metric_name}_{self.config['model']}_{wandb.run.name}.pth")
        
        # Save the model state_dict and checkpoint
        torch.save(self.model.state_dict(), model_path)

        # Save the model to WandB
        wandb.save(model_path)
        artifact = wandb.Artifact(f"{metric_name}_{self.config['model']}_{wandb.run.name}", type="model")
        artifact.add_file(model_path)

        # Add metadata to the artifact
        artifact.metadata["epoch"] = epoch + 1
        artifact.metadata["overall_val_auc"] = overall_val_auc
        artifact.metadata["val_loss"] = val_loss
        artifact.metadata["val_acc"] = val_acc
        artifact.metadata["val_f1"] = val_f1

        wandb.run.log_artifact(artifact)
        print(
            f"Model saved with {metric_name}: {metric_value:.4f}, loss: {val_loss:.4f}, AUC: {overall_val_auc:.4f}, F1 score: {val_f1:.4f}"
        )


if __name__ == "__main__":
    config = {
        "model": "ct_images_cnn",
        "criterion": "cross_entropy",
        "learning_rate": 0.001,
        "weight_decay": 0.05,
        "num_epochs": 300,
        "save_model_path": "research/case_study/biomed/models",
        "device": "cpu",
    }

    # if config["model"] == "ct_images_cnn":
    #     model = ct_images_cnn()
    # if config["criterion"] == "cross_entropy":
    #     criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(
    #     model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    # )

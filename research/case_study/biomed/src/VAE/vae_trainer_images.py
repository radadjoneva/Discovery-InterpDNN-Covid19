# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

from typing import Callable

import torch
import pandas as pd
import wandb
import torch.nn.functional as F

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.VAE.losses import kldiv, l2_dist, weighted_l2_dist, l1_dist
from src.evaluation.performance_metrics import process_outputs, compute_metrics
from src.evaluation.plotting import plot_confusion_matrix


class ImageVAETrainer:
    def __init__(
        self,
        classifier: torch.nn.Module,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        prep: Callable[[torch.Tensor], torch.Tensor],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimiser: torch.optim.Optimizer,
        act_fn: Callable[[torch.Tensor,], torch.Tensor]=None,
        epochs: int=20,
        recon_coeff: int=1,
        recon_loss_type="l2",
        perceptual_loss=False,
        kl_coeff: int=1,
        perceptual_coeff: int=1,
        percept_weighted: bool=False,
        percept_layers: list[str]=['layer1', 'layer2', 'layer3', 'layer4'],
        log_freq: int=10,
        device: torch.device=None,
        target_classes: list[str]=[],
        class_weights: list[float]=[10/3, 2, 5],
        save_path: str="",
    ):
        """Train a VAE for input data reconstruction with feature representations as input to the encoder."""
        self.classifier = classifier
        self.encoder = encoder
        self.decoder = decoder
        self.prep = prep
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimiser = optimiser
        self.act_fn = act_fn
        self.epochs = epochs
        self.recon_coeff = recon_coeff
        self.recon_loss_type = recon_loss_type
        self.kl_coeff = kl_coeff
        self.perceptual_loss = perceptual_loss
        self.perceptual_coeff = perceptual_coeff
        self.percept_weighted = percept_weighted
        self.percept_layers = percept_layers
        self.log_freq = log_freq
        self.device = device
        self.save_path = save_path
        self.target_classes = target_classes
        self.class_weights = torch.Tensor(class_weights).to(device)  # Class weights for reconstruction and perceptual loss

        # Move the models to the device
        self.classifier.to(self.device)
        self.classifier.eval()  # Ensure classifier is in evaluation mode
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # Print loss params
        print("\n--- Loss Parameters ---")
        print(f"Reconstruction Loss Type: {self.recon_loss_type}")
        print(f"Reconstruction Loss Coefficient: {self.recon_coeff}")
        print(f"KL Divergence Loss Coefficient: {self.kl_coeff}")
        if self.perceptual_loss:
            print(f"Perceptual Loss Coefficient: {self.perceptual_coeff}")
            print(f"Perceptual Layers: {self.percept_layers}")
        print("------------------------\n")

        # Initialize history for tracking metrics
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_kl_loss": [],
            "val_kl_loss": [],
            "train_reconstruction_loss": [],
            "val_reconstruction_loss": [],
            "train_perceptual_loss": [],
            "val_perceptual_loss": [],
        }

    def calculate_perceptual_loss(self, real_img, gen_img, labels=None):
        # Extract features from both real and generated images
        real_features = self.classifier.get_features(real_img, layer='all')
        gen_features = self.classifier.get_features(gen_img, layer='all')

        # Initialise total loss
        total_perceptual_loss = 0.0

        # Calculate the feature reconstruction loss summed across all layers
        for layer in self.percept_layers:
            real_feat = real_features[layer]
            gen_feat = gen_features[layer]
            layer_loss = F.mse_loss(gen_feat, real_feat)
            total_perceptual_loss += layer_loss
        
        if self.percept_weighted and labels is not None:
            # Get the weights corresponding to the class/label of each sample
            class_indices = torch.argmax(labels, dim=1)
            weights = self.class_weights[class_indices]
            weights = weights.view(-1, 1)

            # Apply the weights to the perceptual loss
            total_perceptual_loss = total_perceptual_loss * weights
        
        return total_perceptual_loss

    
    def train(self):
        # Weights, gradients and parameters logging
        wandb.watch(self.encoder, log="all", log_freq=10)
        wandb.watch(self.decoder, log="all", log_freq=10)

        # Initialise best validation loss for saving best model
        best_val_loss = float("inf")
        best_val_f1_diff = float("inf")
        best_val_percept_loss = float("inf")

        for epoch in range(self.epochs):
            print(f"\nTraining Epoch {epoch+1}/{self.epochs}")
            self.encoder.train()
            self.decoder.train()

            train_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            train_perceptual_loss = 0.0
            total_samples = 0

            for batch in self.train_loader:
                if batch["input"].dim() == 5:  # Check if input has an extra batch dimension
                    x = batch["input"].view(-1, 1, 224, 224)
                else:
                    x = batch["input"]

                if batch["label"].dim() == 3:  # Check if label has an extra batch dimension
                    labels = batch["label"].view(-1, 3).to(self.device)
                else:
                    labels = batch["label"].to(self.device)

                self.optimiser.zero_grad()
                if self.device:
                    if isinstance(x, list) or isinstance(x, tuple):
                        # handle multimodal case: (tabular, image)
                        x = (x[0].to(self.device), x[1].to(self.device))
                    else:
                        x = x.to(self.device)
                
                if self.prep:
                    x = self.prep(x)

                latent_repr = self.classifier.get_features(x)
                mu, sigma = self.encoder(latent_repr)
                z = self.encoder.sampling(mu, sigma)
                gen = self.decoder(z)

                # Activation function for generated samples
                if self.act_fn:
                    gen = self.act_fn(gen)

                # Reconstruction Loss
                if self.recon_loss_type == "l2":
                    recon_loss = l2_dist(x, gen)
                elif self.recon_loss_type == "weighted_l2":
                    recon_loss = weighted_l2_dist(x, gen, labels, self.class_weights)
                elif self.recon_loss_type == "l1":
                    recon_loss = l1_dist(x, gen)
                else:
                    raise ValueError(f"Invalid reconstruction loss type: {self.recon_loss_type}")

                # KL Divergence Loss
                kl_loss = kldiv(mu, sigma)

                # Calculate Perceptual Loss
                if self.perceptual_loss:
                    perceptual_loss = self.calculate_perceptual_loss(x, gen, labels)
                else:
                    perceptual_loss = torch.zeros_like(recon_loss)

                loss = (
                    self.recon_coeff * recon_loss.mean()
                    + self.kl_coeff * kl_loss.mean()
                    + self.perceptual_coeff * perceptual_loss.mean()
                )
                    
                loss.backward()
                self.optimiser.step()

                train_loss += loss.item() * x.size(0)
                train_recon_loss += recon_loss.mean().item() * x.size(0)
                train_kl_loss += kl_loss.mean().item() * x.size(0)
                train_perceptual_loss += perceptual_loss.mean().item() * x.size(0)
                total_samples += x.size(0)

                # Log running loss to wandb
                wandb.log({"running_loss": loss.item(),
                            "running_reconstruction_loss": recon_loss.mean().item(),
                           "running_kl_loss": kl_loss.mean().item(),
                            "running_perceptual_loss": perceptual_loss.mean().item(),
                            "mu_mean": mu.mean().item(),
                           "sigma_mean": sigma.mean().item(),
                           "z_mean": z.mean().item()
                           })

                wandb.log({"z": wandb.Histogram(z.mean(axis=0).cpu().detach().numpy()),
                            "mu": wandb.Histogram(mu.mean(axis=0).cpu().detach().numpy()),
                           "sigma": wandb.Histogram(sigma.mean(axis=0).cpu().detach().numpy()),
                        })
            
            # Calculate average losses for the epoch
            train_loss = train_loss / total_samples
            train_recon_loss = train_recon_loss / total_samples
            train_kl_loss = train_kl_loss / total_samples
            train_perceptual_loss = train_perceptual_loss / total_samples

            # Validate after each epoch
            val_loss, val_recon_loss, val_kl_loss, val_perceptual_loss = self.validate()

            # Log to wandb
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_reconstruction_loss": train_recon_loss,
                    "train_kl_loss": train_kl_loss,
                    "train_perceptual_loss": train_perceptual_loss,
                    "val_loss": val_loss,
                    "val_reconstruction_loss": val_recon_loss,
                    "val_kl_loss": val_kl_loss,
                    "val_perceptual_loss": val_perceptual_loss
                }
            )

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                # Evaluate and log classifier performance metric differences between real and reconstructed data
                # Training data
                train_metric_differences = self._get_classifier_performance_differences(self.train_loader, "train")
                wandb.log({"train_metric_differences": train_metric_differences})
                # Validation data
                val_metric_differences = self._get_classifier_performance_differences(self.val_loader, "val")
                wandb.log({"val_metric_differences": val_metric_differences})

                # Save the best model based on F1 score difference between real and reconstructed data
                if val_metric_differences["val_F1_Score_diff"] < best_val_f1_diff:
                    best_val_f1_diff = val_metric_differences["val_F1_Score_diff"]
                    model_path = os.path.join(self.save_path, f"f1_vae_resnet50_{wandb.run.name}.pth")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "encoder_state_dict": self.encoder.state_dict(),
                            "decoder_state_dict": self.decoder.state_dict(),
                            "val_loss": val_loss,
                            "val_f1_diff": val_metric_differences["val_F1_Score_diff"],
                            "hidden_dim": self.encoder.fc1.out_features,
                            "latent_dim": self.encoder.fc_mu.out_features,
                        },
                        model_path,
                    )
                    wandb.save(model_path)

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(self.save_path, f"val_vae_resnet50_{wandb.run.name}.pth")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "decoder_state_dict": self.decoder.state_dict(),
                        "val_loss": val_loss,
                        "hidden_dim": self.encoder.fc1.out_features,
                        "latent_dim": self.encoder.fc_mu.out_features,
                    },
                    model_path,
                )
                wandb.save(model_path)
            if val_perceptual_loss < best_val_percept_loss:
                best_val_percept_loss = val_perceptual_loss
                model_path = os.path.join(self.save_path, f"val_percept_vae_resnet50_{wandb.run.name}.pth")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "decoder_state_dict": self.decoder.state_dict(),
                        "val_loss": val_loss,
                        "val_perceptual_loss": val_perceptual_loss,
                        "hidden_dim": self.encoder.fc1.out_features,
                        "latent_dim": self.encoder.fc_mu.out_features,
                    },
                    model_path,
                )
                wandb.save(model_path)

            if epoch == 0 or (epoch + 1) % self.log_freq == 0:
                log_message = (
                    f"{'-'*50}\n"
                    f"Epoch {epoch+1}/{self.epochs}:\n"
                    f"Train Losses:      Total: {train_loss:.4f} | "
                    f"Reconstruction: {train_recon_loss:.4f} | "
                    f"KL Divergence: {train_kl_loss:.4f} | "
                    f"Perceptual: {train_perceptual_loss:.4f}\n"
                    f"Validation Losses: Total: {val_loss:.4f} | "
                    f"Reconstruction: {val_recon_loss:.4f} | "
                    f"KL Divergence: {val_kl_loss:.4f} | "
                    f"Perceptual: {val_perceptual_loss:.4f}\n"
                )
                print(log_message)
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_reconstruction_loss"].append(train_recon_loss)
            self.history["val_reconstruction_loss"].append(val_recon_loss)
            self.history["train_kl_loss"].append(train_kl_loss)
            self.history["val_kl_loss"].append(val_kl_loss)
            self.history["train_perceptual_loss"].append(train_perceptual_loss)
            self.history["val_perceptual_loss"].append(val_perceptual_loss)

        return self.encoder, self.decoder
    

    def validate(self):
        print("Validating...")
        self.encoder.eval()
        self.decoder.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        val_perceptual_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if batch["input"].dim() == 5:  # Check if input has an extra batch dimension
                    x = batch["input"].view(-1, 1, 224, 224)
                else:
                    x = batch["input"]

                if batch["label"].dim() == 3:  # Check if label has an extra batch dimension
                    labels = batch["label"].view(-1, 3).to(self.device)
                else:
                    labels = batch["label"].to(self.device)

                if self.device:
                    if isinstance(x, list) or isinstance(x, tuple):
                        x = (x[0].to(self.device), x[1].to(self.device))
                    else:
                        x = x.to(self.device)

                if self.prep:
                    x = self.prep(x)

                latent_repr = self.classifier.get_features(x)
                mu, sigma = self.encoder(latent_repr)
                z = self.encoder.sampling(mu, sigma)
                gen = self.decoder(z)

                if self.act_fn:
                    gen = self.act_fn(gen)

                # Reconstruction Loss
                if self.recon_loss_type == "l2":
                    recon_loss = l2_dist(x, gen)
                elif self.recon_loss_type == "weighted_l2":
                    recon_loss = weighted_l2_dist(x, gen, labels, self.class_weights)
                elif self.recon_loss_type == "l1":
                    recon_loss = l1_dist(x, gen)
                else:
                    raise ValueError(f"Invalid reconstruction loss type: {self.recon_loss_type}")

                kl_loss = kldiv(mu, sigma)

                # Calculate Perceptual Loss
                if self.perceptual_loss:
                    perceptual_loss = self.calculate_perceptual_loss(x, gen, labels)
                else:
                    perceptual_loss = torch.zeros_like(recon_loss)

                loss = (
                    self.recon_coeff * recon_loss.mean()
                    + self.kl_coeff * kl_loss.mean()
                    + self.perceptual_coeff * perceptual_loss.mean()
                )

                val_loss += loss.item() * x.size(0)
                val_recon_loss += recon_loss.mean().item() * x.size(0)
                val_kl_loss += kl_loss.mean().item() * x.size(0)
                val_perceptual_loss += perceptual_loss.mean().item() * x.size(0)
                total_samples += x.size(0)

        # Calculate average losses
        val_loss = val_loss / total_samples
        val_recon_loss = val_recon_loss / total_samples
        val_kl_loss = val_kl_loss / total_samples
        val_perceptual_loss = val_perceptual_loss / total_samples

        return val_loss, val_recon_loss, val_kl_loss, val_perceptual_loss

    def _get_classifier_predictions(self, data_loader):
        """Generate classifier predictions for both real and reconstructed data for the given data loader."""
        self.encoder.eval()
        self.decoder.eval()

        real_all_labels = []
        real_all_outputs = []
        reconstructed_all_labels = []
        reconstructed_all_outputs = []
        
        correct_predictions_real = 0
        correct_predictions_reconstructed = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                if batch["input"].dim() == 5:  # Check if input has an extra batch dimension
                    x = batch["input"].view(-1, 1, 224, 224)
                else:
                    x = batch["input"]

                if batch["label"].dim() == 3:  # Check if label has an extra batch dimension
                    labels = batch["label"].view(-1, 3).to(self.device)
                else:
                    labels = batch["label"].to(self.device)
                
                # Move input data to device
                if isinstance(x, list) or isinstance(x, tuple):
                    x = (x[0].to(self.device), x[1].to(self.device))
                else:
                    x = x.to(self.device)

                # Preprocess the input if needed
                if self.prep:
                    x = self.prep(x)

                # REAL Data Predictions
                real_outputs = self.classifier(x)
                batch_correct_predictions_real = (real_outputs.argmax(1) == labels.argmax(1)).sum().item()

                # Accumulate the results for real data
                real_all_labels.append(labels)
                real_all_outputs.append(real_outputs)
                correct_predictions_real += batch_correct_predictions_real

                # RECONSTRUCTED Data Predictions
                latent_repr = self.classifier.get_features(x)
                mu, sigma = self.encoder(latent_repr)
                z = self.encoder.sampling(mu, sigma, deterministic=True)
                gen = self.decoder(z)

                if self.act_fn:
                    gen = self.act_fn(gen)

                reconstructed_outputs = self.classifier(gen)
                batch_correct_predictions_reconstructed = (reconstructed_outputs.argmax(1) == labels.argmax(1)).sum().item()

                # Accumulate the results for reconstructed data
                reconstructed_all_labels.append(labels)
                reconstructed_all_outputs.append(reconstructed_outputs)
                correct_predictions_reconstructed += batch_correct_predictions_reconstructed

                # Update total samples
                total_samples += labels.size(0)

        # Process outputs
        real_y_true, real_y_pred, real_probs = process_outputs(real_all_labels, real_all_outputs)
        reconstructed_y_true, reconstructed_y_pred, reconstructed_probs = process_outputs(reconstructed_all_labels, reconstructed_all_outputs)
        
        return (
            real_y_true, real_y_pred, real_probs, correct_predictions_real, 
            reconstructed_y_true, reconstructed_y_pred, reconstructed_probs, correct_predictions_reconstructed,
            total_samples
        )
    
    def _get_classifier_performance_differences(self, data_loader, split):
        """
        Evaluate classifier performance on real and reconstructed data,
        and compute the metric differences for a given split (train/val).
        """
        print(f"Evaluating classifier performance differences on {split} data...")
        # Get the classifier predictions for both real and reconstructed data
        (
            real_y_true, real_y_pred, real_probs, correct_predictions_real, 
            reconstructed_y_true, reconstructed_y_pred, reconstructed_probs, correct_predictions_reconstructed,
            total_samples
        ) = self._get_classifier_predictions(data_loader)

        # Compute and log metrics for real data
        real_metrics = self._compute_metrics(
            real_y_true, real_y_pred, real_probs,
            correct_predictions_real, total_samples,
            split, data_type="real"
        )

        # Compute and log metrics for reconstructed data
        reconstructed_metrics = self._compute_metrics(
            reconstructed_y_true, reconstructed_y_pred, reconstructed_probs,
            correct_predictions_reconstructed, total_samples,
            split, data_type="reconstructed"
        )

        # Compute the differences between real and reconstructed metrics
        metric_differences = {
            key.replace("real", "diff"): real_metrics[key] - reconstructed_metrics.get(key.replace("real", "reconstructed"), 0.0) for key, value in real_metrics.items() if f"{split}_" in key if not isinstance(value, dict)
        }

        return metric_differences
    
    def _compute_metrics(self, y_true, y_pred, probs, correct_predictions, total_samples, split, data_type):
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, probs, self.target_classes, total_samples, split=split, best_model=data_type)
        accuracy = correct_predictions / total_samples
        metrics[f"{split}_Accuracy_{data_type}"] = accuracy
        metrics[f"{split}_Correct_Predictions_{data_type}"] = correct_predictions
        metrics = {f"{key}_{data_type}": value for key, value in metrics.items()}

        # Log metrics to wandb
        wandb.log(metrics)

        # Plot confusion matrix
        conf_key = f"{split}_Confusion_Matrix_{data_type}"
        plot_confusion_matrix(metrics[conf_key], self.target_classes, split, self.save_path, best_model=data_type)
        return metrics
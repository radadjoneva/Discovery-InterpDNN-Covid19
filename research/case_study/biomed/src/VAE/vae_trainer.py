# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

from typing import Callable

import torch
import pandas as pd
import numpy as np
import wandb

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.VAE.losses import kldiv, l2_dist
from src.VAE.evaluate_vae import evaluate_on_data


class VAETrainer:
    def __init__(
        self,
        classifier: torch.nn.Module,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        prep: Callable[[torch.Tensor], torch.Tensor],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimiser: torch.optim.Optimizer,
        categorical_column_ix: list[int],
        continuous_column_ix: list[int],
        act_fn: Callable[[torch.Tensor,], torch.Tensor]=None,
        epochs: int=20,
        categorical_coeff: int=1,
        continuous_coeff: int=1,
        kl_coeff: int=1,
        log_freq: int=10,
        device: torch.device=None,
        column_names: list[str]=None,
        target_classes: list[str]=None,
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
        self.categorical_column_ix = categorical_column_ix
        self.continuous_column_ix = continuous_column_ix
        self.act_fn = act_fn
        self.epochs = epochs
        self.categorical_coeff = categorical_coeff
        self.continuous_coeff = continuous_coeff
        self.kl_coeff = kl_coeff
        self.log_freq = log_freq
        self.device = device
        self.column_names = column_names
        self.target_classes = target_classes
        self.save_path = save_path

        # Move the models to the device
        self.classifier.to(self.device)
        self.classifier.eval()  # Ensure classifier is in evaluation mode
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # Initialize history for tracking metrics
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_kl_loss": [],
            "val_kl_loss": [],
            "train_categorical_loss": [],
            "val_categorical_loss": [],
            "train_continuous_loss": [],
            "val_continuous_loss": [],
        }
    
    def train(self):
        # Weights, gradients and parameters logging
        wandb.watch(self.encoder, log="all", log_freq=10)
        wandb.watch(self.decoder, log="all", log_freq=10)

        # Initialise best validation loss for saving best model
        best_val_loss = float("inf")
        best_val_f1_diff = float("inf")

        for epoch in range(self.epochs):
            self.encoder.train()
            self.decoder.train()

            train_loss = 0.0
            train_categorical_loss = 0.0
            train_continuous_loss = 0.0
            train_kl_loss = 0.0
            total_samples = 0

            # Per column MAE
            train_cat_mae = {col_name: 0.0 for ix, col_name in enumerate(self.column_names) if ix in self.categorical_column_ix}
            train_num_mae = {col_name: 0.0 for ix, col_name in enumerate(self.column_names) if ix in self.continuous_column_ix}

            for batch in self.train_loader:
                x = batch["input"]

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

                # Categorical Loss
                if len(self.categorical_column_ix) > 0:
                    categorical_loss = l2_dist(
                        x[:, self.categorical_column_ix], gen[:, self.categorical_column_ix]
                    )
                else:
                    categorical_loss = torch.tensor(0.0)

                # Continuous Loss
                if len(self.continuous_column_ix) > 0:
                    continuous_loss = l2_dist(
                        x[:, self.continuous_column_ix], gen[:, self.continuous_column_ix]
                    )
                else:
                    continuous_loss = torch.tensor(0.0)

                # KL Divergence Loss
                kl_loss = kldiv(mu, sigma)

                loss = (
                    self.categorical_coeff * categorical_loss.mean()
                    + self.continuous_coeff * continuous_loss.mean()
                    + self.kl_coeff * kl_loss.mean()
                )
                loss.backward()
                self.optimiser.step()

                train_loss += loss.item() * x.size(0)
                train_categorical_loss += categorical_loss.mean().item() * x.size(0)
                train_continuous_loss += continuous_loss.mean().item() * x.size(0)
                train_kl_loss += kl_loss.mean().item() * x.size(0)
                total_samples += x.size(0)

                # Column-wise MAE (accumulate absolute error)
                for i, j in enumerate(self.categorical_column_ix):
                    train_cat_mae[self.column_names[j]] += categorical_loss[:, i].sqrt().sum(dim=0).item()
                for i, j in enumerate(self.continuous_column_ix):
                    train_num_mae[self.column_names[j]] += continuous_loss[:, i].sqrt().sum(dim=0).item()

                # Log running loss to wandb
                wandb.log({"running_loss": loss.item(),
                           "running_categorical_loss": categorical_loss.mean().item(),
                           "running_continuous_loss": continuous_loss.mean().item(),
                           "running_kl_loss": kl_loss.mean().item(),
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
            train_categorical_loss = train_categorical_loss / total_samples
            train_continuous_loss = train_continuous_loss / total_samples
            train_kl_loss = train_kl_loss / total_samples
            # MAE per column
            train_cat_mae = {k: v / total_samples for k, v in train_cat_mae.items()}
            train_num_mae = {k: v / total_samples for k, v in train_num_mae.items()}

            # Validate after each epoch
            val_loss, val_categorical_loss, val_continuous_loss, val_kl_loss, val_cat_mae, val_num_mae = self.validate()

            # Log to wandb
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_categorical_loss": train_categorical_loss,
                    "train_continuous_loss": train_continuous_loss,
                    "train_kl_loss": train_kl_loss,
                    "train_cat_mae": train_cat_mae,
                    "train_num_mae": train_num_mae,
                    "val_loss": val_loss,
                    "val_categorical_loss": val_categorical_loss,
                    "val_continuous_loss": val_continuous_loss,
                    "val_kl_loss": val_kl_loss,
                    "val_cat_mae": val_cat_mae,
                    "val_num_mae": val_num_mae,
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
                    model_path = os.path.join(self.save_path, f"f1_vae_cf_dnn_{wandb.run.name}.pth")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "encoder_state_dict": self.encoder.state_dict(),
                            "decoder_state_dict": self.decoder.state_dict(),
                            "val_loss": val_loss,
                            "val_f1_diff": val_metric_differences["val_F1_Score_diff"],
                            "hidden_dim": self.encoder.fc1.out_features,
                            "latent_dim": self.encoder.fc3.out_features,
                        },
                        model_path,
                    )
                    wandb.save(model_path)

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(self.save_path, f"val_vae_cf_dnn_{wandb.run.name}.pth")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "decoder_state_dict": self.decoder.state_dict(),
                        "val_loss": val_loss,
                        "val_f1_diff": val_metric_differences["val_F1_Score_diff"],
                        "hidden_dim": self.encoder.fc1.out_features,
                        "latent_dim": self.encoder.fc3.out_features,
                    },
                    model_path,
                )
                wandb.save(model_path)

            if epoch == 0 or (epoch + 1) % self.log_freq == 0:
                log_message = (
                    f"{'-'*50}\n"
                    f"Epoch {epoch+1}/{self.epochs}:\n"
                    f"Train Losses:      Total: {train_loss:.4f} | "
                    f"Categorical: {train_categorical_loss:.4f} | "
                    f"Continuous: {train_continuous_loss:.4f} | "
                    f"KL Divergence: {train_kl_loss:.4f}\n"
                    f"Validation Losses: Total: {val_loss:.4f} | "
                    f"Categorical: {val_categorical_loss:.4f} | "
                    f"Continuous: {val_continuous_loss:.4f} | "
                    f"KL Divergence: {val_kl_loss:.4f}"
                )
                print(log_message)
                
            if epoch == self.epochs - 1:
                # Compute and log the final MAE per column at the end of training
                train_mae = {**train_cat_mae, **train_num_mae}
                val_mae = {**val_cat_mae, **val_num_mae}

                data = []
                for feature_name in train_mae.keys():
                    data.append({
                        "Feature Name": feature_name,
                        "Train MAE": train_mae[feature_name],
                        "Val MAE": val_mae[feature_name],
                    })

                # Create DataFrame and log the final MAE per column to wandb
                mae_df = pd.DataFrame(data)
                print(mae_df)
                wandb.log({
                    "Final Column MAE": wandb.Table(dataframe=mae_df)
                })
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_categorical_loss"].append(train_categorical_loss)
            self.history["val_categorical_loss"].append(val_categorical_loss)
            self.history["train_continuous_loss"].append(train_continuous_loss)
            self.history["val_continuous_loss"].append(val_continuous_loss)
            self.history["train_kl_loss"].append(train_kl_loss)
            self.history["val_kl_loss"].append(val_kl_loss)

        return self.encoder, self.decoder, mae_df
    

    def validate(self):
        self.encoder.eval()
        self.decoder.eval()
        val_loss = 0.0
        val_categorical_loss = 0.0
        val_continuous_loss = 0.0
        val_kl_loss = 0.0
        total_samples = 0

        # Per column MAE
        val_cat_mae = {col_name: 0.0 for ix, col_name in enumerate(self.column_names) if ix in self.categorical_column_ix}
        val_num_mae = {col_name: 0.0 for ix, col_name in enumerate(self.column_names) if ix in self.continuous_column_ix}

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["input"]
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

                if len(self.categorical_column_ix) > 0:
                    categorical_loss = l2_dist(
                        x[:, self.categorical_column_ix], gen[:, self.categorical_column_ix]
                    )
                else:
                    categorical_loss = torch.tensor(0.0)

                if len(self.continuous_column_ix) > 0:
                    continuous_loss = l2_dist(
                        x[:, self.continuous_column_ix], gen[:, self.continuous_column_ix]
                    )
                else:
                    continuous_loss = torch.tensor(0.0)

                kl_loss = kldiv(mu, sigma)

                loss = (
                    self.categorical_coeff * categorical_loss.mean()
                    + self.continuous_coeff * continuous_loss.mean()
                    + self.kl_coeff * kl_loss.mean()
                )

                val_loss += loss.item() * x.size(0)
                val_categorical_loss += categorical_loss.mean().item() * x.size(0)
                val_continuous_loss += continuous_loss.mean().item() * x.size(0)
                val_kl_loss += kl_loss.mean().item() * x.size(0)
                total_samples += x.size(0)

                # Column-wise MAE (accumulate absolute error)
                for i, j in enumerate(self.categorical_column_ix):
                    val_cat_mae[self.column_names[j]] += categorical_loss[:, i].sqrt().sum(dim=0).item()

                for i, j in enumerate(self.continuous_column_ix):
                    val_num_mae[self.column_names[j]] += continuous_loss[:, i].sqrt().sum(dim=0).item()

        # Calculate average losses
        val_loss = val_loss / total_samples
        val_categorical_loss = val_categorical_loss / total_samples
        val_continuous_loss = val_continuous_loss / total_samples
        val_kl_loss = val_kl_loss / total_samples
        # MAE per column
        val_cat_mae = {k: v / total_samples for k, v in val_cat_mae.items()}
        val_num_mae = {k: v / total_samples for k, v in val_num_mae.items()}

        return val_loss, val_categorical_loss, val_continuous_loss, val_kl_loss, val_cat_mae, val_num_mae


    def _get_reconstructed_data(self, data_loader):
        """Generate reconstructed data for the given data loader."""
        self.encoder.eval()
        self.decoder.eval()

        real_data_list = []
        labels_list = []
        reconstructed_data_list = []

        with torch.no_grad():
            for batch in data_loader:
                x = batch["input"]
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
                z = self.encoder.sampling(mu, sigma, deterministic=True)
                gen = self.decoder(z)

                if self.act_fn:
                    gen = self.act_fn(gen)

                real_data_list.append(x.cpu())
                labels_list.append(labels.cpu())
                reconstructed_data_list.append(gen.cpu())
        
        # Concatenate the lists into tensors
        real_data = torch.cat(real_data_list, dim=0).to(self.device)
        labels = torch.cat(labels_list, dim=0).to(self.device)
        reconstructed_data = torch.cat(reconstructed_data_list, dim=0).to(self.device)
        
        return real_data, labels, reconstructed_data
    

    def _get_classifier_performance_differences(self, data_loader, split):
        """
        Evaluate classifier performance on real and reconstructed data,
        and compute the metric differences for a given split (train/val).
        """
        # Generate real and reconstructed data tensors
        real_data, labels, reconstructed_data = self._get_reconstructed_data(data_loader)

        # Evaluate classifier performance on real and reconstructed data
        real_metrics = evaluate_on_data(self.classifier, real_data, labels, self.target_classes, device=self.device, split=split, save_path=self.save_path, type_data="real")
        reconstructed_metrics = evaluate_on_data(self.classifier, reconstructed_data, labels, self.target_classes, device=self.device, split=split, save_path=self.save_path, type_data="reconstructed")

        # Compute the differences between real and reconstructed metrics
        metric_differences = {
            key.replace("real", "diff"): real_metrics[key] - reconstructed_metrics.get(key.replace("real", "reconstructed"), 0.0) for key, value in real_metrics.items() if f"{split}_" in key if not isinstance(value, dict)
        }
        return metric_differences
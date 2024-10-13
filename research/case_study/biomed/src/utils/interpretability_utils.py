# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys
import yaml
import torch
import pandas as pd

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.utils.model_utils import load_pretrained_model
from src.VAE.cf_dnn_vae import VariationalEncoder, Decoder
from src.VAE.ct_resnet_vae import ImageVariationalEncoder, ImageDecoder, ResImageDecoder, ResImageDecoder2

# For protype generation
def calculate_diversity(a):
    """Calculate the diversity of a given tensor.

    Args:
        a (torch.Tensor): Input tensor with shape (n_samples/ prototypes, n_features).

    Returns:
        torch.Tensor: Negative mean pairwise distance as a tensor.
    """
    sim_matrix = torch.cdist(a, a, p=2)
    return -sim_matrix.mean() if a.size(0) > 1 else torch.zeros(1, 1)


def initialise_vae_models(classifier_path, config_path, vae_path, input_dim=75, image=False):
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pretrained classifier model
    classifier = load_pretrained_model(classifier_path, config, input_dim=input_dim)

    # Load VAE encoder and decoder state dicts
    checkpoint = torch.load(vae_path, map_location=config["device"])
    encoder_state_dict = checkpoint["encoder_state_dict"]
    decoder_state_dict = checkpoint["decoder_state_dict"]
    best_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["val_loss"]
    best_val_f1_diff = checkpoint["val_f1_diff"] if "val_f1_diff" in checkpoint else None
    hidden_dim = checkpoint.get("hidden_dim", 32)
    latent_dim = checkpoint.get("latent_dim", 16)

    print(f"\nBest model: {vae_path}")
    print(f"Epoch: {best_epoch}   |   Validation loss: {best_val_loss}   |  F1 diff: {best_val_f1_diff}")

    # Initialise VAE Encoder and Decoder and load state dicts
    if not image:
        # Tabular VAE
        encoder = VariationalEncoder(feature_dim=classifier.fc5.out_features, hidden_dim=hidden_dim, latent_dim=latent_dim)
        decoder = Decoder(num_inputs=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    else:
        encoder = ImageVariationalEncoder(feature_dim=classifier.resnet50.fc.in_features, hidden_dim=hidden_dim, latent_dim=latent_dim)
        decoder = ResImageDecoder2(latent_dim=latent_dim, base_channels=64, output_channels=1)

    # Load state dicts into encoder and decoder
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    # Move models to the appropriate device and set to evaluation mode
    classifier.to(config["device"])
    classifier.eval()
    encoder.to(config["device"])
    encoder.eval()
    decoder.to(config["device"])
    decoder.eval()

    return classifier, encoder, decoder, config, latent_dim, hidden_dim




if __name__ == "__main__":
    # Example
    # CF_DNN: apricot-sweep-24
    run_name = "apricot-sweep-24"
    run_id = "radadjoneva-icl/covid-outcome-classification/2sspqbzv"  # apricot-sweep-24
    model_dir = f"research/case_study/biomed/models/CF_DNN/{run_name}"
    classifier_path = os.path.join(model_dir, f"f1_cf_dnn_{run_name}.pth")
    config_path = os.path.join(model_dir, "config.yaml")

    # VAE: woven-snow-97
    vae_run_name = "woven-snow-97"
    vae_path = f"research/case_study/biomed/models/VAE/{vae_run_name}/vae_cf_dnn_{vae_run_name}.pth"

    # Load classifier and VAE models
    classifier, encoder, decoder, config = initialise_vae_models(classifier_path, config_path, vae_path)

    print("Models initialised successfully!")



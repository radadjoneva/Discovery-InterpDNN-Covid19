# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys
import wandb

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.VAE.train_vae import train_vae

vae_tab_sweep_config = {
    "name": "exp8_VAE_tabular",
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "run_name": {"values": ["apricot-sweep-24"]},
        "run_id": {"values": ["radadjoneva-icl/covid-outcome-classification/2sspqbzv"]},
        "model_dir": {"values": ["research/case_study/biomed/models/CF_DNN/apricot-sweep-24"]},
        "save_vae_path": {"values": ["research/case_study/biomed/results/VAE_training/tabular/"]},
        "device": {"values": ["cuda"]},
        "latent_dim": {"values": [16]},
        "hidden_dim": {"values": [32]},
        "learning_rate": {"values": [1e-3]},
        "num_epochs": {"values": [500]},
        "categorical_coeff": {"values": [0.001]},
        "continuous_coeff": {"values": [2]},
        "kl_coeff": {"values": [0, 0.1, 0.8, 2]},
    }
}

vae_image_sweep_config = {
    "name": "exp19_VAE_finetune_dashing-daily-2",
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "run_name": {"values": ["worldly-sweep-4"]},
        "run_id": {"values": ["radadjoneva-icl/covid-outcome-classification/c3hpiq15"]},
        "model_dir": {"values": ["research/case_study/biomed/models/CT_CNN/worldly-sweep-4"]},
        "save_vae_path": {"values": ["research/case_study/biomed/results/VAE_training/image/"]},
        "device": {"values": ["cuda"]},
        "latent_dim": {"values": [256]},
        "hidden_dim": {"values": [1024]},
        "output_channels": {"values": [1]},
        "image_size": {"values": [224]},
        "learning_rate": {"values": [1e-5]},
        "weight_decay": {"values": [0]},
        "num_epochs": {"values": [100]},
        "recon_loss_type": {"values": ["l2"]},
        "reconstruction_coeff": {"values": [0.01]},
        "kl_coeff": {"values": [0.001]},
        "perceptual_loss": {"values": [True]},
        "perceptual_coeff": {"values": [1]},
        "percept_layers": {"values": [["layer4"]]},
        "percept_weighted": {"values": [True]},
        "class_weights": {"values": [[0.5, 0.5, 10]]},
        "k_imgs": {"values": [20]},
        "batch_size": {"values": [5]},
        "data_augmentation": {"values": [False]},
        "fine_tune": {"values": [True]},
        "vae_path": {"values": ["research/case_study/biomed/models/VAE/image/f1_vae_resnet50_scarlet-sweep-3.pth"]},
    }
}

if __name__ == "__main__":
    # Initialise the VAE sweep
    # TABULAR
    # sweep_type = vae_tab_sweep_config
    # sweep_id = wandb.sweep(sweep=sweep_type, project="covid-vae")

    # IMAGE
    sweep_type = vae_image_sweep_config
    sweep_id = wandb.sweep(sweep=sweep_type, project="covid-vae-image")

    # Run the sweep agent
    wandb.agent(sweep_id, function=train_vae)
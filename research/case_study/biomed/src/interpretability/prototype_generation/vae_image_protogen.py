# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

import torch
import pandas as pd
import wandb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

from src.utils.interpretability_utils import initialise_vae_models, calculate_diversity
# from src.interpretability.prototype_generation.protogen_tabular import StandardPreprocessor
from src.VAE.train_vae import inverse_normalize


def protogen_loop(
    device,
    epochs,
    classifier,
    decoder,
    init_input,
    target_ix,
    mean_std,
    objective='ce',
    lr=0.01,
    diversity_weight=0,
    grad_mask=None,
    target_classes = [],
    log_interim=False,
    plot=False,
):
    classifier.eval()
    decoder.eval()

    wandb.log({"diversity_weight": diversity_weight})

    mean, std = mean_std
    prototype_class = target_classes[target_ix[0]]

    input = torch.nn.Parameter(init_input)

    if grad_mask is None:
        grad_mask = torch.ones_like(input)

    optimiser = torch.optim.Adam([input], lr=lr)

    for e in range(epochs):
        gen_image = decoder(input)

        logits = classifier(gen_image)
        probs = torch.softmax(logits, dim=-1)

        if objective == 'ce':
            logit_loss = torch.nn.functional.cross_entropy(logits, target_ix)
        elif objective == 'logit':
            logit_loss = -logits[:, target_ix].mean()

        diversity_loss = diversity_weight * calculate_diversity(input)

        loss = (logit_loss + diversity_loss).sum()

        # Log metrics to WandB
        logits_dict = {f"{c}_{j}": logits[j, i].item() for j in range(logits.size(0)) for i, c in enumerate(target_classes)}
        probs_dict = {f"{c}_{j}": probs[j, i].item() for j in range(probs.size(0)) for i, c in enumerate(target_classes)}
        mean_logits_dict = {f"{c}_mean": logits[:, i].mean().item() for i, c in enumerate(target_classes)}
        mean_probs_dict = {f"{c}_mean": probs[:, i].mean().item() for i, c in enumerate(target_classes)}

        metrics = {
            "epoch": e + 1,
            f"{prototype_class}_logits": logits_dict,
            f"{prototype_class}_probs": probs_dict,
            f"{prototype_class}_mean_logits": mean_logits_dict,
            f"{prototype_class}_mean_probs": mean_probs_dict,
            f"{prototype_class}_logit_loss": logit_loss.item(),
            f"{prototype_class}_diversity_loss": diversity_loss.item(),
            f"{prototype_class}_total_loss": loss.item(),
        }

        wandb.log(metrics)

        optimiser.zero_grad()
        loss.backward()
        input.grad *= grad_mask
        optimiser.step()

        if (log_interim and e % (epochs // 10) == 0) or e == epochs - 1:
            prototype_numbers = torch.arange(probs.size(0)).unsqueeze(1).to(device)
            # Log original images
            inv_gen_image = inverse_normalize(gen_image.clone(), mean, std)
            wandb.log({f"prototype_{prototype_class}": wandb.Image(inv_gen_image, caption=f"Class: {prototype_class}")})

            # Log to terminal
            print(f"{'-'*40}")
            print(f"Epoch {e+1}  |  Prototype Class: {prototype_class}")
            print("Mean Probabilities:")
            for label, prob in mean_probs_dict.items():
                print(f"    {label}: {prob:.4f}")
            print("Mean Logits:")
            for label, logit in mean_logits_dict.items():
                print(f"    {label}: {logit:.4f}")
            print(f"Total Loss: {loss.item()}")
            print(f"Logit Loss: {logit_loss.item()}")
            print(f"Diversity Loss: {diversity_loss.item()}")
            print(f"{'-'*40}\n")
            print(
                f"\n{e}: Total Loss: {loss.data} Logit Loss: {logit_loss.data} Diversity Loss: {diversity_loss.item()}"
            )

    return gen_image



def run_vae_protogen_image():
    vae_protogen_config = {
        "run_name": "worldly-sweep-4",
        "run_id": "radadjoneva-icl/covid-outcome-classification/c3hpiq15",
        "model_dir": "research/case_study/biomed/models/CT_CNN/worldly-sweep-4",
        # "vae_run_name": "spring-cherry-43",
        "vae_run_name": "scarlet-sweep-3",
        "vae_dir": "research/case_study/biomed/models/VAE/image",
        "pixel_stats_path": "research/case_study/biomed/datasets/iCTCF/pixel_stats_224.csv",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 500,
        "target_ix": [2],
        "num_prototypes": 1,
        "objective": "ce",  # 'ce' or 'logit'
        "lr": 0.01,
        "diversity_weight": 0,
        "grad_mask": None,
    }
    
    # Initialise WandB
    wandb.init(project="vae-protogen-image", config=vae_protogen_config)
    config = wandb.config

    # Get classifier and VAE paths
    run_name = config["run_name"]
    model_dir = config["model_dir"]
    classifier_path = os.path.join(model_dir, f"f1_resnet50_{run_name}.pth")
    classifier_config_path = os.path.join(model_dir, "config.yaml")
    vae_run_name = config["vae_run_name"]
    vae_dir = config["vae_dir"]
    vae_path = os.path.join(vae_dir, f"f1_vae_resnet50_{vae_run_name}.pth")
    # vae_path = os.path.join(vae_dir, f"val_percept_vae_resnet50_{vae_run_name}.pth")

    # Load classifier and VAE models
    classifier, encoder, decoder, classifier_config, latent_dim, hidden_dim = initialise_vae_models(classifier_path, classifier_config_path, vae_path, image=True)

    # Update WandB config with classifier config, preserving original WandB config values
    for key, value in classifier_config.items():
        if key not in wandb.config:  # Only update if key is not already in WandB config
            wandb.config[key] = value
    
    # Prototype generation loop
    num_p = config["num_prototypes"]
    device = config["device"]
    target_classes = config["covid_outcome_classes"]

    # Randomly initialise prototypes (all different)
    mu = torch.zeros(num_p, latent_dim)
    sigma = torch.ones(num_p, latent_dim)
    init_input = torch.normal(mean=mu, std=sigma).to(device)

    # Randomly initialise prototypes (all the same)
    # init_input = torch.normal(mean=0, std=1, size=(1, latent_dim)).to(device)
    # init_input = init_input.repeat(num_p, 1)

    target_ix = torch.tensor(config["target_ix"] * num_p).to(device)

    # Load your pixel stats
    pixel_stats = pd.read_csv(config["pixel_stats_path"])
    # Convert to torch and add to device
    mean = torch.tensor(pixel_stats["mean"].values).view(1, 1, 1).to(config["device"])
    std = torch.tensor(pixel_stats["std"].values).view(1, 1, 1).to(config["device"])

    # Generate prototypes
    prototypes = protogen_loop(
        device=device,
        epochs=config["epochs"],
        classifier=classifier,
        decoder=decoder,
        init_input=init_input,
        target_ix = target_ix,
        mean_std = (mean, std),
        objective=config["objective"],
        lr=config["lr"],
        diversity_weight=config["diversity_weight"],
        grad_mask=config["grad_mask"],
        target_classes=target_classes,
        log_interim=True,
        plot=True,
    )

    print("Prototypes generated successfully!")

    wandb.finish()


if __name__ == '__main__':
    run_vae_protogen_image()
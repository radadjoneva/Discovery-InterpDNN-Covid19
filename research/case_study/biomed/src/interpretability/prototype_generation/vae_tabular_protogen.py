# Arush's code for prototype generation
# discovery/research/automl/predictive_feature_reconstruction/protogen.py

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
from src.interpretability.prototype_generation.protogen_tabular import StandardPreprocessor


def protogen_loop(
    device,
    epochs,
    classifier,
    decoder,
    init_input,
    target_ix,
    preprocessing,
    standardiser,
    coercion,
    objective='ce',
    lr=0.01,
    diversity_weight=0,
    grad_mask=None,
    clamp_min=None,
    clamp_max=None,
    categorical_groups=None,
    columns=None,
    target_classes = [],
    log_interim=False,
    plot=False,
):
    classifier.eval()
    decoder.eval()

    wandb.log({"diversity_weight": diversity_weight})

    prototype_class = target_classes[target_ix[0]]
    # Initialise table to store prototype types for logging
    prototype_table = wandb.Table(columns=["epoch", "prototype_number"] + columns + target_classes)
    inv_prototype_table = wandb.Table(columns=["epoch", "prototype_number"] + columns + target_classes)

    continuous_columns_ix = [i for i,c in enumerate(columns) if c.startswith("num")]

    if clamp_min is not None and clamp_max is not None:
        input = torch.randn_like(init_input) * (clamp_max - clamp_min) + clamp_min

    input = torch.nn.Parameter(init_input)

    if grad_mask is None:
        grad_mask = torch.ones_like(input)

    optimiser = torch.optim.Adam([input], lr=lr)

    for e in range(epochs):
        if preprocessing:
            input = preprocessing(input)
        if coercion:
            input = coercion(input)

        if clamp_min is not None and clamp_max is not None:
            c_input = input.clamp(clamp_min, clamp_max)
            gen_data = decoder(c_input) 
        else:
            gen_data = decoder(input)

        logits = classifier(gen_data)
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
            epoch = torch.Tensor([e]* probs.size(0)).unsqueeze(1).to(device)
            cat = torch.cat([epoch, prototype_numbers, gen_data, probs], dim=-1)
            df = pd.DataFrame(data=cat.detach().cpu().numpy(), columns=["epoch", "prototype_number"] + columns + target_classes)
            for row in df.itertuples(index=False, name=None):
                prototype_table.add_data(*row)
            
            # Inverse standardise the generated data to get real range values
            inv_gen_data = gen_data.clone()
            inv_gen_data[:, continuous_columns_ix] = standardiser.reverse(inv_gen_data[:, continuous_columns_ix])
            inv_cat = torch.cat([epoch, prototype_numbers, inv_gen_data, probs], dim=-1)
            inv_df = pd.DataFrame(data=inv_cat.detach().cpu().numpy(), columns=["epoch", "prototype_number"] + columns + target_classes)
            for row in inv_df.itertuples(index=False, name=None):
                inv_prototype_table.add_data(*row)

            if e == epochs - 1:
                wandb.log({f"prototype_probs_{prototype_class}": prototype_table})  # log table at the end of training
                wandb.log({f"inv_prototype_probs_{prototype_class}": inv_prototype_table})  # log table at the end of training

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

    return gen_data



def run_vae_protogen_tabular():
    vae_protogen_config = {
        "run_name": "apricot-sweep-24",
        "run_id": "radadjoneva-icl/covid-outcome-classification/2sspqbzv",
        "vae_run_name": "volcanic-sweep-29",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 1000,
        "target_ix": [1],
        "num_prototypes": 1000,
        # "coercion": None,
        # "preprocessing": None,
        "objective": "ce",  # 'ce' or 'logit'
        "lr": 0.01,
        "diversity_weight": 0.1,
        "grad_mask": None,
        "clamp_min": None,
        "clamp_max": None,
    }
    
    # Initialise WandB
    wandb.init(project="vae-protogen-tabular", config=vae_protogen_config)
    config = wandb.config

    # Get classifier and VAE paths
    run_name = config["run_name"]
    model_dir = f"research/case_study/biomed/models/CF_DNN/{run_name}"
    classifier_path = os.path.join(model_dir, f"f1_cf_dnn_{run_name}.pth")
    classifier_config_path = os.path.join(model_dir, "config.yaml")
    vae_run_name = config["vae_run_name"]
    vae_path = f"research/case_study/biomed/models/VAE/tabular/{vae_run_name}/f1_vae_cf_dnn_{vae_run_name}.pth"

    # Load real datasets (preprocessed)
    data_dir = f"research/case_study/biomed/datasets/iCTCF/processed_cf/{run_name}"
    X_train = pd.read_csv(os.path.join(data_dir, "input_features_train.csv"))
    column_names = X_train.columns.to_list()

    # Load classifier and VAE models
    classifier, encoder, decoder, classifier_config, latent_dim, hidden_dim = initialise_vae_models(classifier_path, classifier_config_path, vae_path)

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

    preprocessor = StandardPreprocessor.load_stats("research/case_study/biomed/datasets/iCTCF/standardise_stats.pkl")

    # Generate prototypes
    prototypes = protogen_loop(
        device=device,
        epochs=config["epochs"],
        classifier=classifier,
        decoder=decoder,
        init_input=init_input,
        target_ix = target_ix,
        preprocessing=None,
        standardiser = preprocessor,
        coercion=None,
        objective=config["objective"],
        lr=config["lr"],
        diversity_weight=config["diversity_weight"],
        grad_mask=config["grad_mask"],
        clamp_min=config["clamp_min"],
        clamp_max=config["clamp_max"],
        categorical_groups=None,
        columns=column_names,
        target_classes=target_classes,
        log_interim=True,
        plot=True,
    )

    print("Prototypes generated successfully!")

    wandb.finish()


if __name__ == '__main__':
    run_vae_protogen_tabular()
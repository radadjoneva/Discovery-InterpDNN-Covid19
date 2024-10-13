# ruff: noqa: E402
# ruff: noqa: I001

import os

# Set the environment variable for CuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import torch
import torch.nn as nn
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import umap

import wandb


# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

from src.interpretability.latent_space.dimensionality_reduction import apply_pca, apply_tsne, apply_umap
from src.utils.interpretability_utils import initialise_vae_models
from src.utils.utils import set_seed
from src.interpretability.latent_space.plot_latent import plot_latent_space, plot_latent_feature_distributions


def generate_latent_representations(encoder, classifier, X, mode=None, sampled_std=None, num_samples=100):
    """
    Generate the latent representations (z) and sampled latent representations (z_sampled)
    based on the mode (deterministic or non-deterministic) and the sampling standard deviation.

    Args:
        encoder (nn.Module): The encoder model.
        classifier (nn.Module): The classifier model.
        X (torch.Tensor): The input features.
        mode (str): The mode, either 'detTrue' for deterministic or 'detFalse' for non-deterministic.
        sampled_std (float): The standard deviation for the sampling.
        num_samples (int): The number of samples to generate for each data point in non-deterministic mode.

    Returns:
        z (torch.Tensor): The latent representation (num_samples x batch_size x latent_dim).
        z_sampled (torch.Tensor): The sampled latent representation.
    """
    latent_repr = classifier.get_features(X)
    mu, sigma = encoder(latent_repr)
    
    if mode == 'detTrue':
        z = mu  # Deterministic
    elif mode == 'detFalse':
        z_list = []
        for _ in range(num_samples):
            z_sample = encoder.sampling(mu, sigma)  # Non-deterministic sampling
            z_list.append(z_sample)
        z = torch.stack(z_list)  # Shape: (num_samples, batch_size, latent_dim)
        z = z.view(-1, z.shape[-1])  # Reshape to (num_samples * batch_size, latent_dim)
    else:
        z = None
    
    # Generate samples from the normal distribution for comparison
    if sampled_std:
        num_samples_X = X.shape[0] * num_samples
        latent_dim = encoder.fc3.out_features
        mean = torch.zeros(num_samples_X, latent_dim).to(X.device)
        std = torch.ones(num_samples_X, latent_dim).to(X.device) * sampled_std
        z_sampled = torch.normal(mean=mean, std=std)
    else:
        z_sampled = None
    
    return z, z_sampled


def analyse_latent_vectors(encoder, classifier, X, labels_real, modes, sampled_stds, num_samples=100):
    """
    Analyse latent z vectors by plotting their distributions.

    Args:
        encoder (nn.Module): The encoder model.
        classifier (nn.Module): The classifier model.
        X (torch.Tensor): The input features.
        labels_real (np.array): The true class labels for X.
        modes (list): The modes to process ('detTrue' and 'detFalse').
        sampled_stds (list): List of standard deviations for sampling.
    """
    # Process the deterministic and non-deterministic modes
    for mode in modes:
        if mode == 'detFalse':
            labels_real = labels_real.repeat(num_samples)

        # Generate z based on the mode
        z, _ = generate_latent_representations(encoder, classifier, X, mode, sampled_std=None)
        
        # Plot latent space distributions for z
        plot_latent_feature_distributions(z, labels_real, title=f"Latent Feature Distribution ({mode})")

    # Process the sampled standard deviations
    for sampled_std in sampled_stds:
        # Generate z_sampled with the given standard deviation
        _, z_sampled = generate_latent_representations(encoder, classifier, X, mode=None, sampled_std=sampled_std)
        
        # Predict labels for z_sampled
        with torch.no_grad():
            gen_z_sampled = classifier(decoder(z_sampled)).argmax(dim=1).cpu().numpy()
        
        # Plot latent space distributions for z_sampled
        plot_latent_feature_distributions(z_sampled, gen_z_sampled, title=f"Latent Feature Distribution (sampled std {sampled_std})")


def visualise_latent_space_dim_reduction(classifier, encoder, decoder, X, labels_real, reduction_method, reduction_params=None, title="Latent Space", mode='detTrue', sampled_std=1, n_components=2, num_samples=5):
    # Get latent representations
    latent_repr = classifier.get_features(X)
    mu, sigma = encoder(latent_repr)
    
    if mode == 'detTrue':
        z = mu  # Deterministic
    elif mode == 'detFalse':
        z = encoder.sampling(mu, sigma)  # Non-deterministic
    
    # Generate samples from the normal distribution for comparison
    num_samples = len(z)
    latent_dim = z.shape[1]
    mean = torch.zeros(num_samples, latent_dim)
    std = mean + sampled_std  # Use std provided in the function argument
    device = X.device
    mean = mean.to(device)
    std = std.to(device)
    z_sampled = torch.normal(mean=mean, std=std)

    # Predict labels for the sampled data
    with torch.no_grad():
        labels_pred = classifier(X).argmax(dim=1).cpu().numpy()  # Predict directly from real X
        gen_z_sampled = decoder(z_sampled)
        labels_sampled = classifier(gen_z_sampled).argmax(dim=1).cpu().numpy()

    # Apply dimensionality reduction to both real z and sampled z
    if reduction_params is None:
        reduction_params = {}
    z_reduced, z_sampled_reduced = reduction_method(z.detach().cpu().numpy(), z_sampled.detach().cpu().numpy(), n_components=n_components, **reduction_params)
    
    # Determine the label for sampled points
    sampled_label = f'z_sampled (std {sampled_std})'
    
    # Plot the latent space
    plot_latent_space(z_reduced, z_sampled_reduced, labels_real, labels_pred, labels_sampled, title, config["covid_outcome_classes"], sampled_label)


def overlay_prototypes_on_latent_space(encoder, classifier, prototypes, ax, color, label):
    """
    Pass prototypes through the encoder to get their latent representations
    and plot them as scatter points on the latent space plots.
    """
    # Convert prototypes to tensor
    prototypes_tensor = torch.Tensor(prototypes).to(next(encoder.parameters()).device)
    
    # Get latent representation of prototypes
    z_prototype, _ = generate_latent_representations(encoder, classifier, prototypes_tensor, mode='detTrue')

    z_prototype = z_prototype.cpu().detach().numpy()

    # Overlay the prototype values on the violin plot
    latent_dim = z_prototype.shape[1]  # Number of latent features
    x_values = np.arange(0, latent_dim)  # Latent feature indices
    # Scatter plot the prototypes in latent space
    for i in range(z_prototype.shape[0]):
        ax.scatter(x_values, z_prototype[i], color=color, label=label if i==0 else "", s=100, marker='x', alpha=0.7)

    # ax.scatter(x_values, z_prototype.flatten(), color=color, label=label, s=100, marker='x', alpha=0.7)
    # ax.scatter(z_prototype[:, 0], z_prototype[:, 1], color=color, s=100, marker='x', label=label)


def analyse_latent_vectors_with_prototypes(encoder, classifier, X, labels_real, modes, sampled_stds, control_prototype, type_i_prototype, type_ii_prototype, num_samples=100):
    """
    Analyse latent z vectors by plotting their distributions and overlay prototypes on the plot.
    """
    # fig, ax = plt.subplots(figsize=(8, 6))

    # Process the deterministic and non-deterministic modes
    for mode in modes:
        fig, ax = plt.subplots(figsize=(8, 6))
        if mode == 'detFalse':
            labels_real = labels_real.repeat(num_samples)

        # Generate z based on the mode
        z, _ = generate_latent_representations(encoder, classifier, X, mode, sampled_std=None)
        
        # Plot latent space distributions for z
        plot_latent_feature_distributions(z, labels_real, title=f"Latent Feature Distribution ({mode})", ax=ax)

        # Overlay prototypes on the plot
        overlay_prototypes_on_latent_space(encoder, classifier, control_prototype, ax, 'royalblue', 'Control Prototype')
        overlay_prototypes_on_latent_space(encoder, classifier, type_i_prototype, ax, 'darkorange', 'Type I Prototype')
        overlay_prototypes_on_latent_space(encoder, classifier, type_ii_prototype, ax, 'forestgreen', 'Type II Prototype')

        # Show the plot with the overlaid prototypes
        plt.legend(fontsize=14)
        dir = "research/case_study/biomed/results/interpretability/latent_space/z_distributions/"
        plt.savefig(os.path.join(dir, f"z_vec_prototypes_{mode}.png"))
        plt.close()



def visualise_latent_space_with_prototypes(encoder, classifier, X, control_prototype, type_i_prototype, type_ii_prototype, 
                                           reduction_method='PCA', n_components=2, mode='detTrue', title="Reduced Latent Space", 
                                           save_path=None, reduction_params=None):
    """
    Visualise latent space with dimensionality reduction and overlay the prototype z vectors.
    
    Parameters:
    - encoder: The encoder model to get latent representations.
    - classifier: The classifier model.
    - X: Input data to generate latent vectors.
    - control_prototype, type_i_prototype, type_ii_prototype: Prototype data for Control, Type I, and Type II.
    - reduction_method: 'PCA' or 'UMAP' for dimensionality reduction.
    - n_components: Number of dimensions to reduce the latent space to.
    - mode: 'detTrue' for deterministic latent vector, 'detFalse' for sampling latent vectors.
    - title: Title of the plot.
    - save_path: Path to save the plot (if provided).
    - reduction_params: Optional parameters for dimensionality reduction methods (UMAP).
    """
    
    if reduction_params is None:
        reduction_params = {}

    # Get latent representations for the input data
    latent_repr = classifier.get_features(X)
    mu, sigma = encoder(latent_repr)
    
    # Use deterministic or non-deterministic latent vectors
    if mode == 'detTrue':
        z = mu.detach().cpu().numpy()  # Deterministic latent representation
    else:
        z = encoder.sampling(mu, sigma).detach().cpu().numpy()  # Non-deterministic sampling
    
    # Get the prototype latent vectors by passing through the encoder
    control_tensor = torch.Tensor(control_prototype).to(next(encoder.parameters()).device)
    type_i_tensor = torch.Tensor(type_i_prototype).to(next(encoder.parameters()).device)
    type_ii_tensor = torch.Tensor(type_ii_prototype).to(next(encoder.parameters()).device)
    
    z_control, _ = generate_latent_representations(encoder, classifier, control_tensor, mode='detTrue')
    z_type_i, _ = generate_latent_representations(encoder, classifier, type_i_tensor, mode='detTrue')
    z_type_ii, _ = generate_latent_representations(encoder, classifier, type_ii_tensor, mode='detTrue')

    z_control = z_control.cpu().detach().numpy()
    z_type_i = z_type_i.cpu().detach().numpy()
    z_type_ii = z_type_ii.cpu().detach().numpy()
    
    # Combine all prototypes into one array
    prototypes_z = np.vstack([z_control, z_type_i, z_type_ii])
    
    # Apply dimensionality reduction
    if reduction_method == 'PCA':
        z_reduced, z_prototypes_reduced = apply_pca(z, prototypes_z, n_components=n_components)
    elif reduction_method == 'UMAP':
        z_reduced, z_prototypes_reduced = apply_umap(z, prototypes_z, n_components=n_components, **reduction_params)
    else:
        raise ValueError("Invalid reduction method. Use 'PCA' or 'UMAP'.")
    
    # Separate prototypes by class
    z_control_reduced = z_prototypes_reduced[:z_control.shape[0]]
    z_type_i_reduced = z_prototypes_reduced[z_control.shape[0]:z_control.shape[0] + z_type_i.shape[0]]
    z_type_ii_reduced = z_prototypes_reduced[z_control.shape[0] + z_type_i.shape[0]:]

    # Plot the reduced latent space
    plt.figure(figsize=(10, 8))
    
    # Plot the data points in the reduced latent space
    plt.scatter(z_reduced[:, 0], z_reduced[:, 1], c='gray', alpha=0.5, label='Data')
    
    # Overlay the prototypes
    plt.scatter(z_control_reduced[:, 0], z_control_reduced[:, 1], color='royalblue', label='Control Prototype', marker='x', s=100)
    plt.scatter(z_type_i_reduced[:, 0], z_type_i_reduced[:, 1], color='darkorange', label='Type I Prototype', marker='x', s=100)
    plt.scatter(z_type_ii_reduced[:, 0], z_type_ii_reduced[:, 1], color='forestgreen', label='Type II Prototype', marker='x', s=100)
    
    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel("Component 1", fontsize=14)
    plt.ylabel("Component 2", fontsize=14)
    plt.legend(fontsize=12)
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()





if __name__ == "__main__":
    # CF_DNN: apricot-sweep-24
    run_name = "apricot-sweep-24"
    run_id = "radadjoneva-icl/covid-outcome-classification/2sspqbzv"  # apricot-sweep-24
    model_dir = f"research/case_study/biomed/models/CF_DNN/{run_name}"
    classifier_path = os.path.join(model_dir, f"f1_cf_dnn_{run_name}.pth")
    config_path = os.path.join(model_dir, "config.yaml")
    init_random_states_path = os.path.join(model_dir, f"random_states_{run_name}.pth")
    eval_random_states_path = os.path.join(model_dir, f"random_states_{run_name}-best-epoch.pth")

    # VAE: woven-snow-97
    vae_run_name = "volcanic-sweep-29"
    vae_path = f"research/case_study/biomed/models/VAE/tabular/{vae_run_name}/f1_vae_cf_dnn_{vae_run_name}.pth"

    # Load data
    # Load real datasets (preprocessed)
    data_dir = f"research/case_study/biomed/datasets/iCTCF/processed_cf/{run_name}"
    X_train = pd.read_csv(os.path.join(data_dir, "input_features_train.csv"))
    Y_train = pd.read_csv(os.path.join(data_dir, "target_outcomes_train.csv"))
    input_dim = X_train.values.shape[1]

    # Load classifier and VAE models and get config
    classifier, encoder, decoder, config, latent_dim, hidden_dim = initialise_vae_models(classifier_path, config_path, vae_path)

    target_classes = config["covid_outcome_classes"]

    # Set random seed for reproducibility
    seed = config["seed"]
    set_seed(seed)

    # Visualise latent space
    X = torch.Tensor(X_train.values).to(config["device"])
    labels = np.argmax(Y_train.values, axis=1)

    # Analyse latent vectors
    modes = ['detTrue', 'detFalse']
    sampled_stds = [0.5, 1, 3]
    
    # analyse_latent_vectors(encoder, classifier, X, labels, modes, sampled_stds, num_samples=100)

    # Define paths to prototype files
    dir = "research/case_study/biomed/results/interpretability/prototypes/vae_tab_prototypes/"
    control_prototype_path = os.path.join(dir, "Control_inv_prototype_100.csv")
    type_i_prototype_path = os.path.join(dir, "Type I_inv_prototype_100.csv")
    type_ii_prototype_path = os.path.join(dir, "Type II_inv_prototype_100.csv")

    # Load prototypes
    control_prototype = pd.read_csv(control_prototype_path)
    type_i_prototype = pd.read_csv(type_i_prototype_path)
    type_ii_prototype = pd.read_csv(type_ii_prototype_path)

    # Filter for epoch 999 and drop irrelevant columns
    control_prototype = control_prototype[control_prototype['epoch'] == 999].drop(columns=['epoch', 'prototype_number', 'Control', 'Type I', 'Type II']).iloc[:5].values
    type_i_prototype = type_i_prototype[type_i_prototype['epoch'] == 999].drop(columns=['epoch', 'prototype_number', 'Control', 'Type I', 'Type II']).iloc[:5].values
    type_ii_prototype = type_ii_prototype[type_ii_prototype['epoch'] == 999].drop(columns=['epoch', 'prototype_number', 'Control', 'Type I', 'Type II']).iloc[:5].values


    # Analyse latent vectors and overlay prototypes
    analyse_latent_vectors_with_prototypes(encoder, classifier, X, labels, modes, sampled_stds, control_prototype, type_i_prototype, type_ii_prototype, num_samples=100)

    # Visualise latent space with prototypes
    # save_path = dir = "research/case_study/biomed/results/interpretability/latent_space/latent_space_with_prototypes.png"
    # visualise_latent_space_with_prototypes(encoder, classifier, X, control_prototype, type_i_prototype, type_ii_prototype, reduction_method='UMAP', n_components=2, mode='detTrue', title="Reduced Latent Space", save_path=save_path)

    # # PCA Visualization (2D and 3D)
    # for mode in ['detTrue', 'detFalse']:
    #     for sampled_std in [0.5, 1, 3]:
    #         # 2D Visualization
    #         visualise_latent_space_dim_reduction(classifier, encoder, decoder, X, labels, apply_pca, title=f"PCA of Latent Space ({mode} with std{sampled_std})", mode=mode, sampled_std=sampled_std, n_components=2)
    #         # 3D Visualization
    #         visualise_latent_space_dim_reduction(classifier, encoder, decoder, X, labels, apply_pca, title=f"PCA of Latent Space 3D ({mode} with std{sampled_std})", mode=mode, sampled_std=sampled_std, n_components=3)

    # # t-SNE Visualization (2D and 3D)
    # for mode in ['detTrue', 'detFalse']:
    #     for sampled_std in [0.5, 1, 3]:
    #         # 2D Visualization
    #         visualise_latent_space_dim_reduction(classifier, encoder, decoder, X, labels, apply_tsne, title=f"t-SNE of Latent Space ({mode} with std{sampled_std})", mode=mode, sampled_std=sampled_std, n_components=2)
    #         # 3D Visualization
    #         visualise_latent_space_dim_reduction(classifier, encoder, decoder, X, labels, apply_tsne, title=f"t-SNE of Latent Space 3D ({mode} with std{sampled_std})", mode=mode, sampled_std=sampled_std, n_components=3)

    # # UMAP Visualization (2D and 3D)
    # for mode in ['detTrue', 'detFalse']:
    #     for sampled_std in [0.5, 1, 3]:
    #         # 2D Visualization
    #         visualise_latent_space_dim_reduction(classifier, encoder, decoder, X, labels, apply_umap, title=f"UMAP of Latent Space ({mode} with std{sampled_std})", mode=mode, sampled_std=sampled_std, n_components=2)
    #         # 3D Visualization
    #         visualise_latent_space_dim_reduction(classifier, encoder, decoder, X, labels, apply_umap, title=f"UMAP of Latent Space 3D ({mode} with std{sampled_std})", mode=mode, sampled_std=sampled_std, n_components=3)
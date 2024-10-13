# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

from src.interpretability.latent_space.visualise_latent_space import generate_latent_representations
from src.utils.interpretability_utils import initialise_vae_models
from src.utils.utils import set_seed


def hierarchical_clustering(z, method="ward", metric="euclidean", num_clusters=None):
    """
    Perform hierarchical clustering on the latent vectors.

    Args:
        z (np.array): Latent vectors (n_samples x latent_dim).
        method (str): Linkage method ('ward', 'complete', 'average', 'single').
        metric (str): Distance metric ('euclidean', 'cosine', etc.).
        num_clusters (int): Number of clusters to extract.

    Returns:
        clustering: Cluster labels for each data point.
    """

    # # Compute the pairwise distance matrix
    # if metric in ['pearson', 'spearman']:
    #     z = z.T  # Transpose for correlation-based metrics
    #     dist_matrix = 1 - np.corrcoef(z.detach().cpu().numpy())
    # else:
    #     dist_matrix = pairwise_distances(z.detach().cpu().numpy(), metric=metric)
    
    # Perform hierarchical clustering using AgglomerativeClustering
    clustering_model = AgglomerativeClustering(n_clusters=num_clusters, metric=metric, linkage=method, compute_full_tree=True, compute_distances=True)
    clustering = clustering_model.fit(z.detach().cpu().numpy())
    cluster_labels = clustering.labels_
    # clustering = clustering_model.fit_predict(z)

    # # Extract the hierarchical tree (children) and distances
    # children = clustering.children_  # Linkage structure (hierarchical tree)
    # distances = clustering.distances_  # Distances between merged clusters
    
    return clustering, cluster_labels

def plot_dendrogram(z, method="ward", metric="euclidean"):
    """
    Plot a dendrogram for hierarchical clustering.

    Args:
        z (np.array): Latent vectors (n_samples x latent_dim).
        method (str): Linkage method ('ward', 'complete', 'average', 'single').
        metric (str): Distance metric ('euclidean', 'correlation', 'cosine', etc.).
    """
    z = z.detach().cpu().numpy()

    # if metric in ['pearson', 'spearman']:
    #     z = z.T  # Transpose for correlation-based metrics
    #     dist_matrix = 1 - np.corrcoef(z)
    #     dist_matrix = squareform(dist_matrix)
    # else:
    #     dist_matrix = pdist(z, metric=metric)

    # Perform hierarchical clustering
    Z = linkage(z, metric=metric, method=method)
    
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dn = dendrogram(Z, truncate_mode='level', p=4)
    plt.title(f"Hierarchical Clustering Dendrogram (Linkage: {method}, Metric: {metric})")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()


def plot_dendrogram_with_feature_values(z, features, feature_names, method="ward", metric="euclidean"):
    """
    Plot a dendrogram for hierarchical clustering with feature values plotted below the dendrogram.

    Args:
        z (np.array): Latent vectors (n_samples x latent_dim).
        features (np.array): The feature matrix (n_samples x n_features) to plot below the dendrogram.
        feature_names (list): List of feature names corresponding to the columns in 'features'.
        method (str): Linkage method ('ward', 'complete', 'average', 'single').
        metric (str): Distance metric ('euclidean', 'correlation', 'cosine', etc.).
    """
    z = z.detach().cpu().numpy()
    features = features.detach().cpu().numpy()

    # Perform hierarchical clustering
    Z = linkage(z, metric=metric, method=method)
    
    # Plot the dendrogram
    fig, ax_dendro = plt.subplots(figsize=(10, 7))
    
    # Create the dendrogram
    dn = dendrogram(Z, ax=ax_dendro)
    # dn = dendrogram(Z, truncate_mode='level', p=4, ax=ax_dendro)
    
    # Get the order of the leaves from the dendrogram
    leaf_order = dn['leaves']
    
    # Reorder the feature matrix based on the dendrogram leaf order
    reordered_features = features[leaf_order, :]
    
    # Plot the feature values below the dendrogram
    fig, (ax_dendro, ax_heatmap) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]})
    
    # Plot the dendrogram
    dendrogram(Z, ax=ax_dendro, truncate_mode='level', p=4)
    
    # Plot heatmap of features below the dendrogram
    sns.heatmap(reordered_features.T, cmap="coolwarm", ax=ax_heatmap, cbar=True, 
                xticklabels=False, yticklabels=feature_names)
    
    # Set axis labels and title
    ax_dendro.set_title(f"Hierarchical Clustering Dendrogram (Linkage: {method}, Metric: {metric})")
    ax_dendro.set_xlabel("Samples")
    ax_dendro.set_ylabel("Distance")
    
    ax_heatmap.set_title("Feature Values Heatmap")
    ax_heatmap.set_ylabel("Features")
    ax_heatmap.set_xlabel("Samples (in dendrogram order)")
    
    plt.tight_layout()
    plt.show()


def plot_clustered_latent_space(z, cluster_labels, method, metric):
    """
    Plot the clustered data in a 2D space using UMAP for dimensionality reduction.

    Args:
        z (np.array): Latent vectors (n_samples x latent_dim).
        cluster_labels (np.array): Cluster labels for each point.
        method (str): Linkage method used for clustering.
        metric (str): Distance metric used for clustering.
    """
    # Apply UMAP to reduce the dimensionality to 2D
    reducer = umap.UMAP(n_components=2, random_state=42)
    # PCA
    # reducer = PCA(n_components=2)
    z_reduced = reducer.fit_transform(z)

    # Plot the clustered data in 2D
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=z_reduced[:, 0], y=z_reduced[:, 1], hue=cluster_labels, palette='Set1', s=60)
    plt.title(f"Clustered Latent Space (Method: {method}, Metric: {metric})")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.show()

def analyze_hierarchical_clustering(z, labels, method="ward", metric="euclidean", num_clusters=3):
    """
    Perform hierarchical clustering on latent vectors and visualize the clusters.

    Args:
        z (np.array): Latent vectors (n_samples x latent_dim).
        labels (np.array): Real class labels for the latent vectors.
        method (str): Linkage method for clustering.
        metric (str): Distance metric for clustering.
        num_clusters (int): Number of clusters to extract.
    """

    # Perform clustering and get cluster labels
    clustering, cluster_labels = hierarchical_clustering(z, method=method, metric=metric, num_clusters=num_clusters)

    # Visualize clustering using a dendrogram
    plot_dendrogram(z, method=method, metric=metric)

    # Plot the clustered data in 2D using UMAP
    plot_clustered_latent_space(z.detach().cpu().numpy(), cluster_labels, method, metric)

    # # Plot the clustered data
    # z = z.detach().cpu().numpy()
    # plt.figure(figsize=(10, 7))
    # sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=cluster_labels, palette='Set1')
    # plt.title(f"Clustered Latent Space (Method: {method}, Metric: {metric})")
    # plt.xlabel("Latent Dimension 1")
    # plt.ylabel("Latent Dimension 2")
    # plt.show()


if __name__ == "__main__":
    # CF_DNN: apricot-sweep-24
    run_name = "apricot-sweep-24"
    run_id = "radadjoneva-icl/covid-outcome-classification/2sspqbzv"  # apricot-sweep-24
    model_dir = f"research/case_study/biomed/models/CF_DNN/{run_name}"
    classifier_path = os.path.join(model_dir, f"f1_cf_dnn_{run_name}.pth")
    config_path = os.path.join(model_dir, "config.yaml")
    init_random_states_path = os.path.join(model_dir, f"random_states_{run_name}.pth")
    eval_random_states_path = os.path.join(model_dir, f"random_states_{run_name}-best-epoch.pth")

    # VAE: volcanic-sweep-29
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

    z, _ = generate_latent_representations(encoder, classifier, X, mode='detFalse', num_samples=10)

    # Perform hierarchical clustering (ward linkage, euclidean distance)
    analyze_hierarchical_clustering(z, labels, method="ward", metric="euclidean", num_clusters=6)

    # For Pearson or Spearman correlation distance
    # analyze_hierarchical_clustering(z, labels, method="average", metric="pearson", num_clusters=3)
    # feature_names = ["num__Age", "num__Body temperature"]
    # plot_dendrogram_with_feature_values(z, X, feature_names=feature_names)
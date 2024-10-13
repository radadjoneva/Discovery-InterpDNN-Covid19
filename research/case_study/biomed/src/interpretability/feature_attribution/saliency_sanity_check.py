# ruff: noqa: E402
# ruff: noqa: I001

import os

# Set the environment variable for CuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

from src.interpretability.feature_attribution.saliency_maps import apply_grad_cam, apply_captum_interp, apply_rise, apply_hierarchical_perturbation, select_random_images_by_class
from src.utils.model_utils import load_pretrained_model
from RISE.explanations import RISE

def randomize_model_weights(model):
    """ Randomize weights of a given model. """
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def compute_spearman_correlation(map1, map2):
    """ Compute Spearman rank correlation between two saliency maps. """
    # Flatten the saliency maps and compute Spearman rank correlation
    map1_flat = map1.flatten()
    map2_flat = map2.flatten()
    correlation, _ = spearmanr(map1_flat, map2_flat)
    return correlation

def saliency_randomization_test(model, random_model, selected_img_paths, class_names, saliency_methods, config):
    """ Compare saliency maps from the trained model vs. random model using Spearman correlation. """
    correlations = {method: [] for method in saliency_methods}

    for img_path in selected_img_paths:
        # Load and preprocess the image
        image = load_and_preprocess_ct_image(img_path, config)

        # Prepare input tensor
        input_tensor = image.unsqueeze(0).to(config["device"])

        # Loop through saliency methods
        for method in saliency_methods:
            # Apply saliency method to the trained model
            saliency_map_trained = apply_saliency_method(model, input_tensor, method)

            # Apply the same saliency method to the randomly initialized model
            saliency_map_random = apply_saliency_method(random_model, input_tensor, method)

            # Compute Spearman rank correlation
            correlation = compute_spearman_correlation(saliency_map_trained, saliency_map_random)
            correlations[method].append(correlation)

    return correlations

def apply_saliency_method(model, input_tensor, method):
    """ Apply a saliency method to a model and return the saliency map. """
    if method in ["gradcam", "gradcam++", "scorecam"]:
        return apply_grad_cam(model, input_tensor, method=method)
    elif method in ["ig", "occlusion", "gs"]:
        return apply_captum_interp(model, input_tensor, method=method)
    elif method == "rise":
        explainer = RISE(model, (224, 224), gpu_batch=32)
        return apply_rise(model, input_tensor, explainer)
    elif method == "hipe":
        return apply_hierarchical_perturbation(model, input_tensor)
    else:
        raise ValueError(f"Invalid saliency method: {method}")

def plot_correlation_results(correlations, saliency_methods):
    """ Plot Spearman rank correlation results. """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to numpy array for easier handling
    corr_data = {method: np.array(corr_list) for method, corr_list in correlations.items()}

    # Boxplot for each saliency method
    ax.boxplot([corr_data[method] for method in saliency_methods], labels=saliency_methods)
    ax.set_title('Spearman Rank Correlation: Trained vs Random Model Saliency Maps')
    ax.set_ylabel('Spearman Rank Correlation')
    ax.set_xlabel('Saliency Methods')

    plt.show()

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "resize": (224, 224),
        "normalise_pixels": "standardise",
        "k_imgs": 10,
    }

    ct_image_dir = "path/to/ct_image_dir"
    class_names = ["Control", "Type I", "Type II"]

    data_dir = "research/case_study/biomed/datasets/iCTCF/processed_cf/apricot-sweep-24"
    # Load patient IDs (CF)
    patient_ids_train = pd.read_csv(os.path.join(data_dir, "patient_ids_train.csv"))
    patient_ids_val = pd.read_csv(os.path.join(data_dir, "patient_ids_val.csv"))
    patient_ids_test = pd.read_csv(os.path.join(data_dir, "patient_ids_test.csv"))

    # Load patient labels
    labels_train = pd.read_csv(os.path.join(data_dir, "target_outcomes_train.csv"))
    labels_val = pd.read_csv(os.path.join(data_dir, "target_outcomes_val.csv"))
    labels_test = pd.read_csv(os.path.join(data_dir, "target_outcomes_test.csv"))

    # Add a column to specify the split
    patient_ids_train["split"] = "train"
    patient_ids_val["split"] = "val"
    patient_ids_test["split"] = "test"

    all_patient_ids = pd.concat([patient_ids_train, patient_ids_val, patient_ids_test], ignore_index=True)
    all_labels = pd.concat([labels_train, labels_val, labels_test], ignore_index=True)

    # Combine patient IDs and labels
    all_patient_id_labels = pd.concat([all_patient_ids, all_labels], axis=1)

    model_dir = "research/case_study/biomed/models/CT_CNN/worldly-sweep-4"
    # Load patient IDs (CT)
    patient_ct_train = pd.read_csv(os.path.join(model_dir, "patient_df_train.csv"))["Patient ID"]
    patient_ct_val = pd.read_csv(os.path.join(model_dir, "patient_df_val.csv"))["Patient ID"]
    patient_ct_test = pd.read_csv(os.path.join(model_dir, "patient_df_test.csv"))["Patient ID"]
    patient_ct_ids = pd.concat([patient_ct_train, patient_ct_val, patient_ct_test], ignore_index=True)

    # Merge/Filter all_patient_id_labels with ct_patient_ids
    all_patient_id_labels = all_patient_id_labels.merge(patient_ct_ids, on='Patient ID')

    # Step 1: Select random images
    num_patients_per_class = 50
    random_ids, random_img_paths = select_random_images_by_class(ct_image_dir, all_patient_id_labels, num_patients_per_class)

    # Step 2: Load pretrained classifier
    model_path = "path/to/pretrained_model.pth"
    model = load_pretrained_model(model_path, config)
    model.to(config["device"])
    model.eval()

    # Step 3: Initialize a randomly initialized model
    random_model = load_pretrained_model(model_path, config)
    randomize_model_weights(random_model)
    random_model.to(config["device"])
    random_model.eval()

    # Step 4: Define saliency methods
    saliency_methods = ["ig", "occlusion", "gradcam", "gradcam++", "scorecam", "rise", "hipe"]

    # Step 5: Perform saliency map comparison
    correlations = saliency_randomization_test(model, random_model, random_img_paths, class_names, saliency_methods, config)

    # Step 6: Plot the results
    plot_correlation_results(correlations, saliency_methods)

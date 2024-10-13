# ruff: noqa: E402
# ruff: noqa: I001

import os

# Set the environment variable for CuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

import sys
import re

import torch

import matplotlib.pyplot as plt
import cv2
import numpy as np
import yaml
import pandas as pd
import torch.nn.functional as F
import json
import random

from captum.attr import (
    GradientShap,
    IntegratedGradients,
    Occlusion,
    LRP
)
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    FullGrad,
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    XGradCAM,
    EigenGradCAM,
    LayerCAM,
)

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

from src.preprocessing.ct_preprocess import load_and_preprocess_ct_image
from src.utils.data_utils import select_even_slices
from src.utils.model_utils import load_pretrained_model
from src.utils.utils import set_seed
from RISE.explanations import RISE
from HiPe.HiPe import hierarchical_perturbation


def apply_grad_cam(
    model,
    original_image,
    input_tensor,
    target_layers="last_conv",
    target_class=0,
    method="gradcam",
    reshape_transform=None,
    use_cuda=False,
):
    grad_cam_methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
    }

    model.eval()

    classes = ["Control", "Type I", "Type II"]
    print(f"Method: {method}")

    # Predict the class of the input image
    with torch.no_grad():
        output = model(input_tensor)
        prediction_score, predicted_idx = torch.max(F.softmax(output, dim=1), 1)
        print(f"Predicted class: {classes[predicted_idx]}")
        print(f"Predicted prob: {prediction_score.squeeze().item()}")

    if target_layers == "last_conv":
        # target_layers = [model.resnet50.layer4[-1].conv3]
        target_layers = [model.resnet50.layer4[-1]]

    cam = grad_cam_methods[method](
        model=model,
        target_layers=target_layers,
        # use_cuda=False,
        reshape_transform=reshape_transform,
    )

    # for c, label in enumerate(classes):
    target_label = classes[target_class]
    print(f"Target class: {target_class} {target_label}")

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    # targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(1), ClassifierOutputTarget(2),
    #             ClassifierOutputTarget(3), ClassifierOutputTarget(4)]
    targets = [ClassifierOutputTarget(target_class)]

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(
        input_tensor=input_tensor, targets=targets, aug_smooth=False, eigen_smooth=True
    )

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    print("------------------------------------")
    return grayscale_cam


def apply_captum_interp(
    model,
    original_image,
    input_tensor,
    method="ig",
    target_class = 0,
    target_layers="last_conv",
    reshape_transform=None,
    use_cuda=False,
):
    # attribution_methods = {
    #     "lrp": LRP,
    #     "ig": IntegratedGradients,
    #     "gs": GradientShap,
    #     "occlusion": Occlusion,
    # }

    model.eval()

    classes = ["Control", "Type I", "Type II"]
    print(f"Method: {method}")

    # Predict the class of the input image
    with torch.no_grad():
        output = model(input_tensor)
        prediction_score, predicted_idx = torch.max(F.softmax(output, dim=1), 1)
        print(f"Predicted class: {classes[predicted_idx]}")
        print(f"Predicted prob: {prediction_score.squeeze().item()}")

    # # Show the original image for comparison
    # _ = viz.visualize_image_attr(
    #     None,
    #     original_image,
    #     method="original_image",
    #     title="Original Image",
    # )

    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#0000ff"), (1, "#0000ff")], N=256
    )

    target_label = classes[target_class]

    # for c, label in enumerate(classes):
    print(f"Target class: {target_class} {target_label}")

    if method == "ig":
        # Integrated Gradients
        # Initialize the attribution algorithm with the model
        integrated_gradients = IntegratedGradients(model)

        # Attribute the output target to
        attributions_ig, delta_ig = integrated_gradients.attribute(
            input_tensor, target=target_class, n_steps=100, return_convergence_delta=True
        )
        attributions_ig = attributions_ig.squeeze(0).cpu().detach().numpy()  # remove batch dimension

        saliency_map_fig, _ = viz.visualize_image_attr(
            np.transpose(attributions_ig, (1, 2, 0)),
            np.transpose(np.expand_dims(original_image, axis=0), (1, 2, 0)),
            method="heat_map",
            cmap=default_cmap,
            show_colorbar=True,
            sign="positive",
            title=f"Integrated Gradients target {target_class} {target_label}",
            use_pyplot=False,
        )

    elif method == "occlusion":
        occlusion = Occlusion(model)
        attributions_occ = occlusion.attribute(
            input_tensor,
            target=target_class,
            strides=(1, 8, 8),
            sliding_window_shapes=(1, 15, 15),
            baselines=0,
        )
        attributions_occ = attributions_occ.squeeze(0).cpu().detach().numpy()  # remove batch dimension

        saliency_map_fig, _ = viz.visualize_image_attr_multiple(
            np.transpose(attributions_occ, (1, 2, 0)),
            np.transpose(np.expand_dims(original_image, axis=0), (1, 2, 0)),
            # ["original_image", "heat_map", "heat_map", "masked_image"],
            # ["all", "positive", "negative", "positive"],
            ["heat_map"],
            ["positive"],  # positive attribution
            show_colorbar=True,
            titles=["Positive Attribution"],
            # fig_size=(18, 6),
            use_pyplot=False,
        )
    elif method == "gs":
        gradient_shap = GradientShap(model)

        # Defining baseline distribution of images
        rand_img_dist = torch.cat([input_tensor * 0, input_tensor * 1])

        attributions_gs = gradient_shap.attribute(
            input_tensor,
            n_samples=50,
            stdevs=0.0001,
            baselines=rand_img_dist,
            target=predicted_idx,
        )
        attributions_gs = attributions_gs.squeeze(0).cpu().detach().numpy()  # remove batch dimension

        saliency_map_fig, _ = viz.visualize_image_attr_multiple(
            np.transpose(attributions_gs, (1, 2, 0)),
            np.transpose(np.expand_dims(original_image, axis=0), (1, 2, 0)),
            ["heat_map"],
            # ["all", "absolute_value"],
            ["positive"],
            cmap=default_cmap,
            # show_colorbar=True,
            use_pyplot=False,
        )
    
    # Extract saliency map from figure
    axes = saliency_map_fig.get_axes()
    saliency_map = axes[0].images[0].get_array()
    print("------------------------------------")

    return saliency_map


def apply_rise(
    model, 
    img, 
    explainer, 
    target_class=None, 
    generate_new_masks=True, 
    masks_path='masks.npy'
):
    """
    Applies RISE saliency method to a given image and model.

    Args:
        - model: The pretrained CNN model.
        - img: The input tensor (single image) for which we want to compute saliency maps.
        - explainer: RISE explainer object.
        - target_class: The target class for which to compute the saliency map.
        - generate_new_masks: Whether to generate new RISE masks.
        - masks_path: Path to the precomputed RISE masks.

    Returns:
        - saliency_map: The saliency map (mask) for the specified target class.
    """
    # Check if masks already exist or if we need to generate them
    if generate_new_masks or not os.path.isfile(masks_path):
        print("Generating new masks for RISE...")
        explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=masks_path)  # N: number of masks, s: resolution, p1: probability of 1
    else:
        print("Loading existing masks for RISE...")
        explainer.load_masks(masks_path)

    # Move image to the correct device
    img = img.cuda() if torch.cuda.is_available() else img

    # Apply RISE to compute saliency for all classes
    with torch.no_grad():
        saliency = explainer(img).cpu().numpy()

    # Predict the class of the input image
    with torch.no_grad():
        output = model(img)
        prediction_score, predicted_idx = torch.max(torch.softmax(output, dim=1), 1)
        print(f"Predicted class: {predicted_idx.item()}, Prediction score: {prediction_score.item()}")

    print(f"Target class: {target_class}")

    # Get the saliency map for the target class
    saliency_map = saliency[target_class]
    print("------------------------------------")

    return saliency_map


def apply_hierarchical_perturbation(
    model, 
    img, 
    target_class=None, 
    perturbation_type='mean', 
    interp_mode='nearest',
    threshold_mode='mid-range',
    resize=None, 
    batch_size=32, 
    vis=False
):
    """
    Applies HiPe saliency method to a given image and model.

    Args:
        - model: The pretrained CNN model.
        - img: The input tensor (single image) for which we want to compute saliency maps.
        - target_class: The target class for which to compute the saliency map.
        - perturbation_type: Type of perturbation ('mean', 'blur', 'fade').
        - interp_mode: Interpolation mode for upscaling masks.
        - threshold_mode: Threshold mode for selecting masks ('mean', 'mid-range').
        - resize: Resize the output saliency map (if provided).
        - batch_size: Batch size for processing masks.
        - vis: Whether to visualize the process at each depth level.

    Returns:
        - saliency_map: The saliency map (mask) for the specified target class.
    """
    # Move image to the correct device (GPU or CPU)
    img = img.cuda() if torch.cuda.is_available() else img

    # Predict the class of the input image
    with torch.no_grad():
        output = model(img)
        prediction_score, predicted_idx = torch.max(torch.softmax(output, dim=1), 1)
        print(f"Predicted class: {predicted_idx.item()}, Prediction score: {prediction_score.item()}")

    print(f"Target class: {target_class}")
    print("Applying HiPe saliency method...")

    # Apply the HiPe method
    saliency_map, total_masks = hierarchical_perturbation(
        model=model,
        input=img,
        target=target_class,
        vis=vis,
        interp_mode=interp_mode,
        resize=resize,
        batch_size=batch_size,
        perturbation_type=perturbation_type,
        threshold_mode=threshold_mode,  # Can be 'mean' or 'mid-range'
        return_info=False  # Set to True if you want additional mask info
    )

    print(f"Total masks used: {total_masks}")
    print("------------------------------------")

    # convert to numpy array for plotting
    saliency_map_np = saliency_map.squeeze().cpu().detach().numpy()
    return saliency_map_np


def extract_number(filepath):
    """Extract the first sequence of digits in the filename."""
    numbers = re.findall(r"\d+", os.path.basename(filepath))
    return int(numbers[-1]) if numbers else 0


def analyse_select_patients_images(patient_df, ct_cnn, ct_image_dir, class_names, config, results_dir):
    """ Process the selected patients' images, perform model inference, and save the concatenated images
    along with the model outputs (logits, softmax) and labels (true and predicted).

    Args:
        - patient_df: DataFrame containing the patient IDs and their respective labels.
        - ct_cnn: Pretrained CNN model for inference.
        - ct_image_dir: Directory containing the CT scan images.
        - class_names: List of class names for the classification task.
        - config: Configuration dict with image processing settings.
        - results_dir: Directory where results will be saved.
    """

    # Prepare figure for saving concatenated patient images
    fig_concat, axes_concat = plt.subplots(len(patient_df), 1, figsize=(22, 22 * len(patient_df) / 6))
    plt.subplots_adjust(wspace=0, hspace=-0.65)
    # Prepare figure for concatenated images with HiPe saliency maps
    fig_concat_hipe, axes_concat_hipe = plt.subplots(len(patient_df), 1, figsize=(22, 22 * len(patient_df) / 6))
    plt.subplots_adjust(wspace=0, hspace=-0.65)

    # # Prepare figure for saving single patient images
    # fig_single, axes_single = plt.subplots(len(patient_df), 1, figsize=(5, 5 * len(patient_df)))
    # plt.subplots_adjust(wspace=0, hspace=0.02)

    # Initialize RISE explainer
    explainer = RISE(ct_cnn, (224, 224), gpu_batch=32)  # adjust batch size if torch.cuda.OutOfMemoryError

    # Saliency methods to apply
    saliency_methods = ["ig", "occlusion", "gradcam", "gradcam++", "scorecam", "rise", "hipe"]
    # saliency_methods = ["hipe"]

    # Figure params
    num_patients = len(patient_df)
    num_methods = len(saliency_methods)
    fig_width = 3 * (num_methods + 1)
    fig_height = 3 * num_patients

    # Prepare figure for saving single patient images with saliency maps (columns)
    fig_single_saliency, axes_single_saliency = plt.subplots(
        num_patients, num_methods + 1, figsize=(fig_width, fig_height)
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.1) 

    single_images = []

    # Loop through each patient in selected_patient_df
    for i, (index, row) in enumerate(patient_df.iterrows()):
        patient_id = row['Patient ID']

        # Concatenate the original images horizontally
        patient_dir = os.path.join(ct_image_dir, "Patient " + str(patient_id))
        ct_dir = os.path.join(patient_dir, "CT")
        img_paths = [os.path.join(ct_dir, img) for img in os.listdir(ct_dir) if img.endswith(".jpg")]
        img_paths = sorted(img_paths, key=extract_number)

        # Select the k even slices (default 10 images from 60% of slices in the middle)
        k_imgs = select_even_slices(img_paths, nb_imgs=config["k_imgs"])

        # Read and preprocess images as input tensor
        images = []
        for img_path in k_imgs:
            image = load_and_preprocess_ct_image(img_path, config, patient_dir, is_train=False)  # no data augmentation
            images.append(image)
        
        # Select image for single image analysis - comparison of saliency maps
        single_image = images[5]  # 6th image
        single_images.append(single_image)

        # Create input tensor for concatenated images (stacked horizontally)
        input_tensor_concat = torch.cat(images, dim=2).unsqueeze(0).to(config["device"])
        img_np_concat = input_tensor_concat.squeeze().cpu().numpy()

        # Display concatenated images
        axes_concat[i].imshow(img_np_concat, cmap="gray")
        axes_concat[i].axis('off')

        # Create input tensor for single image
        input_tensor_single = single_image.unsqueeze(0).to(config["device"])
        img_np_single = input_tensor_single.squeeze().cpu().numpy()

            # # Display single image
            # axes_single[i].imshow(img_np_single, cmap="gray")
            # axes_single[i].axis('off')

        # Display the original single image in the first column
        axes_single_saliency[i, 0].imshow(img_np_single, cmap="gray")
        axes_single_saliency[i, 0].axis('off')
        # axes_single_saliency[i, 0].set_title('Input')

        # Model inference for concatenated images
        with torch.no_grad():
            logits_concat = ct_cnn(input_tensor_concat)
            softmax_output_concat = F.softmax(logits_concat, dim=1)

        # Model inference for single image
        with torch.no_grad():
            logits_single = ct_cnn(input_tensor_single)
            softmax_output_single = F.softmax(logits_single, dim=1)

        # Get predicted labels for concatenated images
        predicted_label_idx_concat = torch.argmax(softmax_output_concat, dim=1).item()
        predicted_label_concat = class_names[predicted_label_idx_concat]

        # Get predicted labels for single image
        predicted_label_idx_single = torch.argmax(softmax_output_single, dim=1).item()
        predicted_label_single = class_names[predicted_label_idx_single]

        # Get the one-hot encoded true label from selected_patient_df row
        true_label_idx = row[['cat__Morbidity outcome_Control',
                              'cat__Morbidity outcome_Type I',
                              'cat__Morbidity outcome_Type II']].values.argmax()
        true_label = class_names[true_label_idx]

        # Apply HiPe saliency method on concatenated image
        # saliency_map_concat = apply_hierarchical_perturbation(
        #     model=ct_cnn,
        #     img=input_tensor_concat,
        #     target_class=predicted_label_idx_concat,  # Replace with appropriate target class
        #     perturbation_type='blur',  # Choose your perturbation type (blur, fade, mean)
        #     interp_mode='nearest',
        #     threshold_mode='mid-range',
        #     vis=False,  # No visualization for the individual depth steps, set to False to avoid intermediate plots
        #     resize=(2240, 224),  # Resize to the shape of the concatenated image
        #     batch_size=32  # Adjust batch size as needed
        # )
        saliency_map_concat = apply_grad_cam(ct_cnn, img_np_concat, input_tensor_concat, method="scorecam", target_class=predicted_label_idx_concat)


        # Plot concatenated original image with HiPe saliency superimposed
        axes_concat_hipe[i].imshow(img_np_concat, cmap="gray", interpolation='none')  # Original concatenated image
        axes_concat_hipe[i].imshow(saliency_map_concat, cmap="jet", alpha=0.5, interpolation='none')  # HiPe saliency map
        axes_concat_hipe[i].axis('off')

        # Loop through and apply saliency methods
        for j, method in enumerate(saliency_methods):
            if method == "rise":
                # Apply RISE to the single image
                # saliency_maps, probs, classes = apply_rise(ct_cnn, input_tensor_single, explainer, top_k=1, generate_new_masks=generate_new_masks)
                saliency_map = apply_rise(
                    model=ct_cnn, 
                    img=input_tensor_single, 
                    explainer=explainer, 
                    target_class=predicted_label_idx_single,  # Specify the target class (e.g., class 0 for "Control")
                    generate_new_masks=True,
                    masks_path="research/case_study/biomed/results/interpretability/saliency_maps/rise_masks.npy"
                )
                # saliency_map = saliency_maps[0]  # Since we are only interested in the top-1 class

            elif method in ["gradcam", "gradcam++", "scorecam"]:
                # GradCAM-related methods
                saliency_map = apply_grad_cam(ct_cnn, img_np_single, input_tensor_single, method=method, target_class=predicted_label_idx_single)
            elif method in ["ig", "occlusion", "gs"]:
                # Captum methods (Integrated Gradients, GradientShap, etc.)
                saliency_map = apply_captum_interp(ct_cnn, img_np_single, input_tensor_single, method=method, target_class=predicted_label_idx_single)
            elif method == "hipe":
                saliency_map = apply_hierarchical_perturbation(
                                    model=ct_cnn,
                                    img=input_tensor_single,
                                    target_class=predicted_label_idx_single,
                                    perturbation_type='blur',  # Could be 'blur' or 'fade' or 'mean'
                                    interp_mode='nearest',  # Interpolation mode for upscaling masks
                                    threshold_mode='mid-range',  # Threshold mode for selecting masks ('mean' or 'mid-range')
                                    vis=True,  # Set to True to visualise each depth
                                    resize=(224, 224),  # Resize the saliency map to this resolution
                                    batch_size=32  # Batch size for mask processing
                                )
            else:
                raise ValueError(f"Invalid saliency method: {method}")
            
            # fig_single_saliency.add_axes(ax)

            # Display saliency map, superimposed on original image, in the next columns
            axes_single_saliency[i, j + 1].imshow(img_np_single, cmap="gray", interpolation='none')  # original image
            axes_single_saliency[i, j + 1].imshow(saliency_map, cmap="jet", alpha=0.5, interpolation='none')
            axes_single_saliency[i, j + 1].axis('off')
            # axes_single_saliency[i, j + 1].set_title(method)

        # Store the results in the DataFrame for concatenated images
        patient_df.at[i, 'True Label'] = true_label
        patient_df.at[i, 'Predicted Label (Concatenated)'] = predicted_label_concat
        patient_df.loc[index, 'Logits (Concatenated)'] = json.dumps(logits_concat.cpu().numpy().flatten().tolist())
        patient_df.loc[index, 'Softmax (Concatenated)'] = json.dumps(softmax_output_concat.cpu().numpy().flatten().tolist())

        # Store the results in the DataFrame for single image analysis
        patient_df.at[i, 'Predicted Label (Single)'] = predicted_label_single
        patient_df.loc[index, 'Logits (Single)'] = json.dumps(logits_single.cpu().numpy().flatten().tolist())
        patient_df.loc[index, 'Softmax (Single)'] = json.dumps(softmax_output_single.cpu().numpy().flatten().tolist())

    # Save figures
    os.makedirs(results_dir, exist_ok=True)
    fig_concat.savefig(os.path.join(results_dir, "selected_patients_concatenated_images.png"), bbox_inches='tight')
    fig_concat_hipe.savefig(os.path.join(results_dir, "selected_patients_concatenated_images_scorecam.png"), bbox_inches='tight')
    # fig_single.savefig(os.path.join(results_dir, "selected_patients_single_images.png"), bbox_inches='tight')
    fig_single_saliency.savefig(os.path.join(results_dir, "selected_patients_single_images_saliency_maps.png"), bbox_inches='tight')

    # Save the DataFrame with model outputs
    patient_df.to_csv(os.path.join(results_dir, "selected_patients_model_outputs.csv"), index=False)
    plt.close(fig_concat)
    # plt.close(fig_single)
    plt.close(fig_single_saliency)
    plt.close(fig_concat_hipe)


def select_random_images_by_class(ct_image_dir, all_patient_id_labels, num_patients_per_class=2):
    """
    Selects `num_patients_per_class` random patients from each class (Control, Type I, Type II),
    and selects one random image from each patient's CT directory.

    Args:
        ct_image_dir: Directory containing the CT scan images.
        all_patient_id_labels: DataFrame with patient IDs and their respective classes.
        num_patients_per_class: Number of patients to select from each class.

    Returns:
        random_ids: List of randomly selected patient IDs.
        random_img_paths: List of paths to the randomly selected images.
    """
    class_columns = {
        'Control': 'cat__Morbidity outcome_Control',
        'Type I': 'cat__Morbidity outcome_Type I',
        'Type II': 'cat__Morbidity outcome_Type II'
    }

    # Dictionary to store random patients and images per class
    random_ids = []
    random_img_paths = []

    def extract_number(filepath):
        """Extract the first sequence of digits in the filename."""
        numbers = re.findall(r"\d+", os.path.basename(filepath))
        return int(numbers[-1]) if numbers else 0

    # Iterate over each class
    for class_name, class_column in class_columns.items():
        # Filter the DataFrame for the current class
        class_patient_df = all_patient_id_labels[all_patient_id_labels[class_column] == 1]

        # Randomly select the specified number of patients from the current class
        selected_patients = class_patient_df.sample(n=num_patients_per_class, random_state=42)

        # Loop through the selected patients and select a random image from each
        for _, row in selected_patients.iterrows():
            patient_id = row['Patient ID']
            patient_dir = os.path.join(ct_image_dir, f"Patient {patient_id}", "CT")

            # Get all images in the patient's CT directory
            img_paths = [os.path.join(patient_dir, img) for img in os.listdir(patient_dir) if img.endswith(".jpg")]
            img_paths = sorted(img_paths, key=extract_number)
            
            # Get the 20% middle image paths (if there are enough images)
            start_index = int(len(img_paths) * 0.4)
            end_index = int(len(img_paths) * 0.6)
            middle_img_paths = img_paths[start_index:end_index]

            # Randomly select one image
            selected_img_path = random.choice(middle_img_paths)
            
            # Store the patient ID and the selected image path
            random_ids.append(patient_id)
            random_img_paths.append(selected_img_path)

    return random_ids, random_img_paths



if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "extract_lung_parenchyma": False,
        "resize": (224, 224),
        "crop_margin": False,
        "normalise_pixels": "standardise",
        "k_imgs": 10,
        "data_augmentation": False
    }

    ct_image_dir = "research/case_study/biomed/datasets/iCTCF/CT"

    # Selected patients (correctly classified by the model)
    patient_ids = [
        1088,  # Control
        # 552,  # Control (community-acquired pneumonia)
        # 113,  # Type I (mild, CT negative)
        862,  # Type I (regular, CT positive)
        # 265,  # Type II (severe, cured, CT positive)
        44, # Type II (severe, deceased, CT positive)
        # 804  # Type II (critically ill, deceased, CT positive)
    ]
    class_names = ['Control', 'Type I', 'Type II']

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

    # Filter the DataFrame for the selected patients
    selected_patient_df = all_patient_id_labels[all_patient_id_labels['Patient ID'].isin(patient_ids)]
    # Reindex the DataFrame to match the order of patient_ids
    selected_patient_df = selected_patient_df.set_index('Patient ID').reindex(patient_ids).reset_index()

    # Load the pretrained classifier model
    run_name = "worldly-sweep-4"
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as file:
        classifier_config = yaml.safe_load(file)
        classifier_config["device"] = config["device"]
    model_path = os.path.join(model_dir, f"f1_{classifier_config['model']}_{run_name}.pth")
    init_random_states_path = os.path.join(model_dir, f"random_states_{run_name}.pth")
    eval_random_states_path = os.path.join(model_dir, f"random_states_{run_name}-best-epoch.pth")

    # Set random seed for reproducibility
    set_seed(classifier_config["seed"])

    ct_cnn = load_pretrained_model(model_path, classifier_config, input_dim=None)
    ct_cnn.to(config["device"])
    ct_cnn.eval()

    results_dir = "research/case_study/biomed/results/interpretability/saliency_maps"

    # Analyse selected patients' images
    analyse_select_patients_images(
        patient_df=selected_patient_df, 
        ct_cnn=ct_cnn, 
        ct_image_dir=ct_image_dir, 
        class_names=class_names, 
        config=config, 
        results_dir=results_dir
    )

    # # Select 2 random patients from each class and get their random images
    # random_ids, random_img_paths = select_random_images_by_class(ct_image_dir, all_patient_id_labels, num_patients_per_class=2)

    # # Filter the DataFrame for the random patients
    # random_patient_df = all_patient_id_labels[all_patient_id_labels['Patient ID'].isin(random_ids)].copy()
    # random_patient_df = random_patient_df.set_index('Patient ID').reindex(random_ids).reset_index()
    # random_patient_df['img_path'] = random_img_paths

    # print("done")

    # # Perform analysis for random patients
    # analyse_select_patients_images(
    #     patient_df=random_patient_df, 
    #     ct_cnn=ct_cnn, 
    #     ct_image_dir=ct_image_dir, 
    #     class_names=class_names, 
    #     config=config, 
    #     results_dir=results_dir,
    #     single_img=True
    # )

    # analyse_patients(random_patient_df, ct_image_dir, config, class_names, analysis_type="random", results_dir=results_dir)

    # Apply Grad-CAM
    # apply_grad_cam(
    #     ct_cnn,
    #     original_images,
    #     input_tensor,
    #     target_layers="last_conv",
    #     method="scorecam",
    #     use_cuda=False,
    # )

# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys
import yaml
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import umap
from sklearn.manifold import TSNE
from lucent.optvis import objectives, param, render, transform
from lucent.modelzoo.util import get_model_layers

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.utils.model_utils import load_pretrained_model
from src.preprocessing.ct_preprocess import load_and_preprocess_ct_image

def plot_activations(activation, title, num_cols=8, max_channels=None):
    """
    Plot feature maps (activations) in a grid.

    Args:
        activation (torch.Tensor): The activation tensor (output of a layer).
        title (str): Title for the plot.
        num_cols (int): Number of columns in the grid. Default is 8.
        max_channels (int): Maximum number of feature maps to display. Default is all.
    """
    # Get the total number of channels (feature maps)
    total_channels = activation.shape[1]

    # Split the activations into batches to fit within max_channels per figure
    for batch_start in range(0, total_channels, max_channels):
        batch_end = min(batch_start + max_channels, total_channels)
        num_channels_in_batch = batch_end - batch_start

        # Calculate the number of rows based on the number of columns and channels
        num_rows = math.ceil(num_channels_in_batch / num_cols)

        # Create a figure
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

        # Reduce space between the subplots
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        for i, channel_idx in enumerate(range(batch_start, batch_end)):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.imshow(activation[0, channel_idx].cpu().detach().numpy(), cmap='gray')
            ax.axis('off')

        # Hide any unused subplots
        for i in range(num_channels_in_batch, num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.axis('off')

        # Set the title for this batch
        batch_title = f"{title}_batch_{batch_start // max_channels + 1}"
        plt.suptitle(batch_title, fontsize=16)

        # Save each batch as a separate figure
        plt.savefig(f"research/case_study/biomed/results/interpretability/feature_vis/{batch_title}.png")
        # plt.show()
        plt.close()


# Adapted from activation-atlas-simple.ipynb - Distill Activation Atlas - https://distill.pub/2019/activation-atlas/
# https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/activation-atlas/activation-atlas-simple.ipynb#scrollTo=Vvm12KVNmpHF

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def whiten(full_activations):
    """ Whiten activations by inverting the correlation matrix of the activations.
    Whitening removes correlations between features and scales them to have unit variance."""
    correl = np.matmul(full_activations.T, full_activations) / len(full_activations)
    correl = correl.astype("float32")
    S = np.linalg.inv(correl)
    S = S.astype("float32")
    return S

# Function for dimensionality reduction (UMAP or TSNE)
def reduce_dimensions(activations):
    return umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.01, metric="cosine", random_state=42).fit_transform(activations)

# Normalize layout to ensure it fits within a grid
def normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1):
    """Removes outliers and scales layout to between [0,1]."""
    mins = np.percentile(layout, min_percentile, axis=0)
    maxs = np.percentile(layout, max_percentile, axis=0)
    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)
    clipped = np.clip(layout, mins, maxs)
    clipped -= clipped.min(axis=0)
    clipped /= clipped.max(axis=0)
    return clipped

# Whitened, euclidean neuron objective
def direction_neuron_S(layer_name, vec, batch=None, x=None, y=None, S=None):
    def inner(T):
        layer = T(layer_name)  # Get the activations from the layer
        # Set spatial coordinates (center of the layer if not provided)
        x_ = layer.shape[2] // 2 if x is None else x
        y_ = layer.shape[3] // 2 if y is None else y
        # Get the activations at the specific (x_, y_) position
        acts = layer[batch, :, x_, y_]
        vec_ = vec
        if S is not None:
            # Apply whitening transformation if S is provided
            vec_ = torch.matmul(vec_.unsqueeze(0), S)[0]

        # Compute dot product between activations and vec_
        dot = torch.mean(acts * vec_)
        return dot
    return inner 


# Cosine similarity objective for neurons
def direction_neuron_cossim_S(model, layer_name, vec, batch=None, x=None, y=None, cossim_pow=1, S=None):
    def inner(optimized_image):
        # Pass the optimized image through the model to get activations
        layer_activations = model.get_features(optimized_image, layer_name)  # Replace with actual function that retrieves layer activations

        # Set the spatial coordinates to the center if not provided
        x_ = layer_activations.shape[2] // 2 if x is None else x
        y_ = layer_activations.shape[3] // 2 if y is None else y

        # Get the activations at the specific (x_, y_) position for the given batch index
        if batch is not None:
            acts = layer_activations[batch, :, x_, y_]
        else:
            acts = layer_activations[:, :, x_, y_]  # Handle no specific batch

        # Apply whitening transformation if S is provided
        vec_ = vec
        if S is not None:
            vec_ = torch.matmul(torch.from_numpy(vec_).float().unsqueeze(0), torch.from_numpy(S).float())[0]

        # Compute the magnitude of activations
        mag = torch.sqrt(torch.sum(acts**2))

        # Compute dot product between activations and vec_
        dot = torch.mean(acts * vec_)

        # Compute cosine similarity and ensure it's above a threshold (0.1)
        cossim = dot / (1e-4 + mag)
        cossim = torch.maximum(torch.tensor(0.1), cossim)

        # Return the scaled cosine similarity as the objective
        return dot * cossim ** cossim_pow

    return inner


def optimize_image(model, obj_list, param_f, transforms, n_steps=512, alpha=True):
    print("Starting image optimization...")

    # Initialize image and optimizer
    params, image_f = param_f()
    optimizer = optim.Adam(params, lr=0.05)

    losses = []

    # Optimization loop
    for step in range(n_steps):
        optimizer.zero_grad()

        # Apply the image transformations
        transformed_image = image_f()
        for transform_fn in transforms:
            transformed_image = transform_fn(transformed_image)

        # Compute the loss from the objective function
        loss = sum([obj(transformed_image) for obj in obj_list])
        loss.backward()  # Backpropagate to get the gradients
        optimizer.step()  # Update the image

        # Store the loss
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item()}")

    # After optimization, retrieve the final optimized image
    optimized_image = image_f().detach().cpu().numpy()
    return optimized_image, loss.item()


# Function to render a batch of activations as icons
def render_icons(directions, model, layer, size=80, n_steps=128, S=None, num_attempts=1, cossim=True, alpha=True):
    image_attempts = []
    loss_attempts = []

    for attempt in range(num_attempts):
        # Render an image for each activation vector
        param_f = lambda: param.image(size, batch=len(directions), fft=True, decorrelate=True, channels=1)

        if S is not None:
            if cossim:
                obj_list = [
                    direction_neuron_cossim_S(model, layer, v, batch=n, S=S, cossim_pow=4)
                    for n, v in enumerate(directions)
                ]
            else:
                obj_list = [
                    direction_neuron_S(layer, v, batch=n, S=S)
                    for n, v in enumerate(directions)
                ]
        else:
            obj_list = [
                objectives.direction_neuron(layer, v, batch=n)
                for n, v in enumerate(directions)
            ]

        # obj = objectives.Objective.sum(obj_list)
        def combined_objective(T):
            return sum([obj(T) for obj in obj_list])
        
        obj = combined_objective

        # Define transformations
        transforms = []
        if alpha:
            transforms.append(transform.collapse_alpha_random())  # Randomly collapse alpha
        transforms.append(transform.pad(2, mode='constant', constant_value=1))  # Pad with constant value
        transforms.append(transform.jitter(4))  # Jitter with a displacement of 4 pixels
        transforms.append(transform.jitter(4))
        transforms.append(transform.jitter(8))  # Larger jitter with 8-pixel displacement
        transforms.append(transform.jitter(8))
        transforms.append(transform.jitter(8))
        transforms.append(transform.random_scale([0.995**n for n in range(-5, 80)] + [0.998**n for n in 2*list(range(20, 40))]))  # Complex scaling
        transforms.append(transform.random_rotate(list(range(-20, 20)) + list(range(-10, 10)) + list(range(-5, 5)) + 5*[0]))  # Multi-range rotation
        transforms.append(transform.jitter(2))  # Final jitter with 2-pixel displacement

        # transforms = [transform.jitter(2), transform.random_scale([1.1, 1.2]), transform.random_rotate(list(range(-20, 20)))]
        
        # Call the optimize_image function to optimize the image and compute the losses
        image, loss = optimize_image(model=model, 
                                            obj_list=obj_list, 
                                            param_f=param_f, 
                                            transforms=transforms, 
                                            n_steps=n_steps, 
                                            alpha=alpha)

        # Store the best image and the corresponding loss
        loss_attempts.append(loss)
        image = np.transpose(image, [0, 2, 3, 1])[:, :, :, 0]  # Keep only the first channel
        image_attempts.append(image)

    # Use the image with the lowest loss
    loss_attempts = np.array(loss_attempts)
    best_attempt_idx = np.argmin(loss_attempts)
    best_image = image_attempts[best_attempt_idx]
    print("Image optimization completed.")
    return best_image, loss_attempts[best_attempt_idx]


# Grid function for binning activations into cells
def grid(xpts=None, ypts=None, grid_size=(8, 8), x_extent=(0., 1.), y_extent=(0., 1.)):
    xpx_length, ypx_length = grid_size
    xpt_length, ypt_length = x_extent[1] - x_extent[0], y_extent[1] - y_extent[0]
    xpxs = ((xpts - x_extent[0]) / xpt_length) * xpx_length
    ypxs = ((ypts - y_extent[0]) / ypt_length) * ypx_length

    xs = []
    for xi in range(grid_size[0]):
        ys = []
        for yi in range(grid_size[1]):
            in_bounds_x = np.logical_and(xi <= xpxs, xpxs <= xi + 1)
            in_bounds_y = np.logical_and(yi <= ypxs, ypxs <= yi + 1)
            in_bounds = np.logical_and(in_bounds_x, in_bounds_y)
            in_bounds_indices = np.where(in_bounds)[0]
            ys.append(in_bounds_indices)
        xs.append(ys)
    # return np.asarray(xs)
    return xs  # list of lists


# Rendering the layout of activations as an atlas
def render_layout(model, layer, S, xs, ys, activations, n_steps=512, min_density=10, grid_size=(10, 10), icon_size=80):
    grid_layout = grid(xpts=xs, ypts=ys, grid_size=grid_size)
    icons = []
    
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            indices = grid_layout[x][y]
            if len(indices) > min_density:  # ensure the cell has enough activations
                avg_activation = np.mean(activations[indices], axis=0)
                icons.append((avg_activation, x, y))

    # Separate avg_activation, x, and y for passing to render_icons
    avg_activations = [icon[0] for icon in icons]
    coords_x = [icon[1] for icon in icons]
    coords_y = [icon[2] for icon in icons]

    # Pass avg_activations (icons[:, 0]) to render_icons
    icon_batch, losses = render_icons(avg_activations, model, layer, alpha=False, size=icon_size, S=S, n_steps=n_steps)

    # Initialize canvas for rendering the activation atlas
    canvas = np.ones((icon_size * grid_size[0], icon_size * grid_size[1]))
    # Iterate through icon_batch and place them on the canvas
    for i, icon in enumerate(icon_batch):
        y = int(coords_y[i])  # Access y from coords_y list
        x = int(coords_x[i])  # Access x from coords_x list
        canvas[(grid_size[0] - x - 1) * icon_size:(grid_size[0] - x) * icon_size, y * icon_size:(y + 1) * icon_size] = icon

    return canvas


# Main script to execute the visualization
def run_visualization(model, image, layer_name):
    """Main function to run the visualization of activations atlas."""
    # Get activations for a specific layer
    raw_activations = model.get_features(image, layer_name)
    raw_activations = raw_activations.detach().cpu().numpy()
    raw_activations = raw_activations.squeeze(0)
    channels, height, width = raw_activations.shape
    # Reshape to (channels, spatial_dim) where spatial_dim = height * width
    raw_activations = raw_activations.reshape(channels, height * width)
    # Transpose to (spatial_dim, channels)
    raw_activations = raw_activations.T

    S = whiten(raw_activations)

    # Dimensionality reduction
    layout = reduce_dimensions(raw_activations)
    layout = normalize_layout(layout)

    xs, ys = layout[:, 0], layout[:, 1]

    plt.figure(figsize=(10, 10))  # Create a figure of size 10x10 inches
    plt.scatter(x=layout[:, 0], y=layout[:, 1], s=2)  # Plot the points with a size of 2
    plt.savefig(f"research/case_study/biomed/results/interpretability/feature_vis/activations_layout_{layer_name}.png")
    plt.show()
    plt.close()
    
    # Render layout and activation atlas
    canvas = render_layout(model, layer_name, S, xs, ys, raw_activations, n_steps=512, grid_size=(20, 20))
    
    # Display the final activation atlas
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap='gray', aspect='auto')
    plt.axis("off")
    plt.savefig(f"research/case_study/biomed/results/interpretability/feature_vis/activations_atlas_{layer_name}.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "extract_lung_parenchyma": False,
        "resize": (224, 224),
        "crop_margin": False,
        "normalise_pixels": "standardise",
    }

    ct_image_dir = "research/case_study/biomed/datasets/iCTCF/CT"
    patient_dir = os.path.join(ct_image_dir, "Patient 6")
    image_path = os.path.join(patient_dir, "CT/IMG-0001-00030.jpg")

    original_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    original_img = cv2.resize(original_img, config["resize"])
    original_img = np.float32(original_img) / 255
    # Read and preprocess images as input tensor
    image = load_and_preprocess_ct_image(image_path, config, patient_dir, is_train=False)
    image = image.unsqueeze(0).to(config["device"])

    # Load the pretrained classifier model
    model_dir = "research/case_study/biomed/models/CT_CNN/worldly-sweep-4"
    run_name = "worldly-sweep-4"
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as file:
        classifier_config = yaml.safe_load(file)
        classifier_config["device"] = config["device"]
    model_path = os.path.join(model_dir, f"f1_{classifier_config['model']}_{run_name}.pth")
    init_random_states_path = os.path.join(model_dir, f"random_states_{run_name}.pth")
    eval_random_states_path = os.path.join(model_dir, f"random_states_{run_name}-best-epoch.pth")

    ct_cnn = load_pretrained_model(model_path, classifier_config, input_dim=None)
    ct_cnn.eval()

    # Get activations for all layers
    activations = ct_cnn.get_features(image, layer='all')

    # # Visualize the activations for each layer
    # for layer_name, activation in activations.items():
    #     if layer_name not in ['avgpool', 'fc']:
    #         plot_activations(activation, f"Activations of {layer_name}", num_cols=8, max_channels=64)
    
    # Visualize the activations of a specific layer: Activations Atlas - Lucent
    # What features is the NN focusing on?
    # --------------------------------------------------------------
    # 1. Extract activations a specific layer of the model
    # 2. Whiten activations: remove correlations between features and scales them to have unit variance.
    # 3. Reduce the dimensionality of activations using UMAP or TSNE
    # 4. For each activation in the reduced space, the method generates a small visual representation ("icon")
    # 5. The icons are arranged in a grid to form an "activation atlas"
    # --------------------------------------------------------------
    for layer_name, activation in activations.items():
        run_visualization(ct_cnn, image, layer_name)






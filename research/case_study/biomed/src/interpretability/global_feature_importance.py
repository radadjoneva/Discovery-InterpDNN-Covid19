# Reference: Seba's original code in packages/interpretability/tabular/feature_importance.py
# Adapted to PyTorch model (skorch) and multi-class classification (vs regression)

# ruff: noqa: E402
# ruff: noqa: I001
# ruff: noqa: E731

import os
import sys
import wandb


from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from alibi.explainers import ALE, plot_ale
from sklearn.inspection import PartialDependenceDisplay, partial_dependence, permutation_importance
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from alepython import ale_plot


# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.utils.model_utils import load_pretrained_model


def compute_pdp_feature_importances(
    model: Pipeline,
    X: pd.DataFrame,
    features: Optional[List[str]] = None,
    grid_resolution: int = 100,
    classes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute PDP-based feature importances for all numerical features.

    This algorithm calculates the importance of each feature by measuring the deviation
    of the Partial Dependence Plot (PDP) values from their mean. The steps are as follows:

    1. For each feature in the list of features:
        a. Compute the partial dependence of the feature using the trained model and dataset.
        b. Extract the average PDP values.
        c. Calculate the mean of these PDP values.
        d. Compute the deviations of the PDP values from their mean.
        e. Calculate the root mean square of these deviations to get the feature importance.
    2. Store the computed importance for each feature in a dictionary.
    3. Return the dictionary containing feature importances.

    Args:
        model (Pipeline): Trained sklearn model.
        X (pd.DataFrame): DataFrame containing the dataset.
        features (List[str]): List of numerical feature names.
        grid_resolution (int): Number of points to evaluate PDP.

    Returns:
        Dict[str, float]: Dictionary of feature importances.
    """
    importances = {}

    if features is None:
        features = X.columns.tolist()

    X_copy = X.copy()
    # NOT used?
    # categorical_features = X_copy.select_dtypes(include=["object"]).columns
    # X_copy[categorical_features] = X_copy[categorical_features].fillna("missing")

    for feature in features:
        pdp_results = partial_dependence(
            model,
            X_copy,
            # scikit-learn's type is incorrect and should accept List[str]
            # https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html
            [feature],  # type: ignore
            grid_resolution=grid_resolution,
        )

        # Dictionary to store feature importances for each class
        feature_importances = {}

        for class_idx in range(pdp_results["average"].shape[0]):
            pdp_values = pdp_results["average"][class_idx]

            # Calculate the deviation from the average pdp_value for the current class
            mean_pdp = np.mean(pdp_values)
            deviations = pdp_values - mean_pdp
            importance = np.sqrt(np.mean(deviations**2))

            # Use class labels as keys in the dictionary, if available
            class_label = classes[class_idx] if classes is not None else class_idx
            feature_importances[class_label] = importance
        
        importances[feature] = feature_importances

    return importances


def compute_permutation_feature_importances(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.DataFrame,
    scoring_func: Callable = f1_score,
    n_repeats: int = 5,
    classes: Optional[List[str]] = None,
    mode: str = "macro",
    custom_class_group: Optional[List[int]] = None
) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute permutation-based feature importances.

    This algorithm works by measuring the increase in the prediction error of the model
    after permuting the values of each feature. The idea is that if a feature is important,
    shuffling its values will increase the model error, because the model relies on that feature
    for making accurate predictions.

    Args:
        model (Pipeline): Trained sklearn model.
        X (pd.DataFrame): DataFrame containing the dataset.
        y (pd.Series): Series containing the target values.
        scoring_func (callable): Function to compute the error metric. Default is f1_score.
        n_repeats (int): Number of times to permute a feature. Default is 5.
        classes (List[str]): List of class labels. Default is None.
        mode (str): "macro" for overall importance, "one-vs-all" for individual classes, "custom" for specific comparisons.
        custom_class_group (List[int]): List of class indices to group together for custom comparisons.

    Returns:
        Union[Dict[str, float], Dict[str, Dict[str, float]]]: Dictionary of feature importances.
        If "macro" returns overall importances.
        If "one-vs-all", returns a nested dictionary with importances for each class or comparison.
        If "custom", returns importances for the custom class group.
    """

    def calculate_importance(model, X, y, scoring_fn, n_repeats):
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            scoring=scoring_fn,
            random_state=0,
        )
        importances_mean = result["importances_mean"]
        return importances_mean
    
    if mode == "macro":
        y_labels = np.argmax(y.values, axis=1)
        scoring_fn = lambda estimator, X, y: scoring_func(y, estimator.predict(X), average="macro")
        importances = calculate_importance(model, X, y_labels, scoring_fn, n_repeats)
        return {feature: importance for feature, importance in zip(X.columns, importances)}

    elif mode == "one-vs-all":
        if classes is None:
            raise ValueError("For 'one-vs-all' mode, 'classes' must be provided.")
        
        importances = {}
        y_labels = np.argmax(y.values, axis=1)

        for class_idx, class_label in enumerate(classes):
            scoring_fn_class = lambda estimator, X, y: scoring_func(y, estimator.predict(X), labels=[class_idx], average=None)
            feature_importances = calculate_importance(model, X, y_labels, scoring_fn_class, n_repeats)
            importances[class_label] = {feature: importance[0] for feature, importance in zip(X.columns, feature_importances)}
        return importances

    elif mode == "custom":
        if custom_class_group is None:
            raise ValueError("For 'custom' mode, 'custom_class_group' must be provided.")

        y_custom = np.where(np.isin(np.argmax(y.values, axis=1), custom_class_group), 1, 0)
        scoring_fn_custom = lambda estimator, X, y: scoring_func(y, np.where(np.isin(estimator.predict(X), custom_class_group), 1, 0), average='binary')
        importances = calculate_importance(model, X, y_custom, scoring_fn_custom, n_repeats)
        return {feature: importance for feature, importance in zip(X.columns, importances)}

    else:
        raise ValueError("Invalid mode. Use 'macro', 'one-vs-all', or 'custom'.")


def compute_ale_feature_importances(
    model: Pipeline,
    X: pd.DataFrame,
    X_real: pd.DataFrame,
    features: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    proba: bool = False,
    plot: bool = True,
    save_path: str = "./ALE"
) -> Dict[str, float]:
    """
    Compute ALE-based feature importances.

    This function calculates feature importances based on Accumulated Local Effects (ALE).
    It handles both numerical and categorical features, and works with sklearn Pipelines
    that may include preprocessing steps.

    Args:
        model (Pipeline): Trained sklearn Pipeline.
        X (pd.DataFrame): DataFrame containing the dataset.
        X_real (pd.DataFrame): DataFrame containing the inverse-transformed dataset.
        features (List[str]): List of feature names to compute importances for.
        classes (List[str]): List of class labels.
        proba (bool): Whether the model outputs probabilities. Default is False.
        plot (bool): Whether to plot the ALE plots. Default is True.
        save_path (str): Path to save the plots. Default is "./ALE".

    Returns:
        Dict[str, float]: Dictionary of feature importances.
    """
    X_preprocessed_df, predict_fn, all_processed_features = (
        prepare_data_and_predict_function_for_ale(model, X, proba=proba)
    )

    if features is None:
        features = X.columns.tolist()

    target_names = classes if classes is not None else None

    ale_explainer = ALE(predict_fn, feature_names=all_processed_features, target_names=target_names)
    ale_explanation = ale_explainer.explain(X_preprocessed_df.values)

    # Plot ALE plots
    if plot:
        ale_explanation_inv = inverse_transform_ale(ale_explanation, X_real)
        plot_ale_grid(ale_explanation_inv, features, X_real, target_names=target_names, proba=proba, save_path=save_path, n_cols=len(classes), sharey='row')

    ale_importances = {}
    for feature in features:
        # Find all processed features that correspond to the original feature
        corresponding_features = [f for f in all_processed_features if feature in f]

        # Sum ALE values for all corresponding processed features
        importance = sum(
            np.sum(np.abs(ale_explanation.ale_values[all_processed_features.index(f)]), axis=0)
            for f in corresponding_features
        )

        if classes is not None and len(classes) == len(importance):
            # Create the dictionary mapping classes to their importance
            importance_dict = {cls: imp for cls, imp in zip(classes, importance)}
            ale_importances[feature] = importance_dict
        else:
            ale_importances[feature] = importance

    return ale_importances


def prepare_data_and_predict_function_for_ale(
    model: Pipeline, X: pd.DataFrame, proba: bool = False
) -> Tuple[pd.DataFrame, Callable, List[str]]:
    predict_fn = model.predict
    X_preprocessed = X
    processed_feature_names = X.columns.tolist()

    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps and model.named_steps["preprocessor"] is not None:
        X_preprocessed = model.named_steps["preprocessor"].transform(X)
        if hasattr(model.named_steps["preprocessor"], "get_feature_names_out"):
            processed_feature_names = (
                model.named_steps["preprocessor"].get_feature_names_out().tolist()
            )

        if hasattr(X_preprocessed, "toarray"):
            X_preprocessed = X_preprocessed.toarray()
        X_preprocessed = pd.DataFrame(X_preprocessed)
        predict_fn = model.named_steps[list(model.named_steps.keys())[-1]].predict
    
    if hasattr(model, "named_steps") and "to_tensor" in model.named_steps:
        def predict_fn(X_ndarray):
            # Convert ndarray to tensor
            X_tensor = torch.tensor(X_ndarray, dtype=torch.float32)
            
            # Pass the tensor through the PyTorch model in the pipeline
            pytorch_model = model.named_steps["model"].module_
            with torch.no_grad():
                predictions_tensor = pytorch_model(X_tensor)

            # Apply softmax to get probabilities if proba=True
            if proba:
                predictions_tensor = torch.nn.functional.softmax(predictions_tensor, dim=1)
            
            # Convert the model's output back to an ndarray
            predictions_ndarray = predictions_tensor.detach().numpy()

            return predictions_ndarray

    return X_preprocessed, predict_fn, processed_feature_names


def inverse_transform_ale(ale_explanation, X_real, feature_prefix="num__"):
    """
    Inverse transform ALE explanation feature values and deciles for numerical features.

    Args:
        ale_explanation: The ALE explanation object containing feature_values and feature_deciles.
        X_real: DataFrame containing the inverse-transformed (original scale) dataset.
        feature_prefix: Prefix indicating numerical features (default: "num__").

    Returns:
        ale_explanation: The ALE explanation object with transformed feature values and deciles.
    """
    # Compute mean and std for each numerical feature from X_real
    mean_std_dict = {
        feature: {'mean': X_real[feature].mean(), 'std': X_real[feature].std()}
        for feature in X_real.columns if feature.startswith(feature_prefix)
    }

    for idx, feature in enumerate(ale_explanation.feature_names):
        if feature.startswith(feature_prefix):  # Only transform numerical features
            mean = mean_std_dict[feature]['mean']
            std = mean_std_dict[feature]['std']
            
            # Inverse transform feature value and deciles
            ale_explanation.feature_values[idx] = ale_explanation.feature_values[idx] * std + mean
            ale_explanation.feature_deciles[idx] = ale_explanation.feature_deciles[idx] * std + mean

    return ale_explanation


def plot_feature_importances(
    feature_importances: Dict[str, Dict[Union[int, str], float]],
    class_label: Union[int, str],
    title: str = "Feature Importance Plot",
    importance_fraction: float = 1.0,
    classes: List[str] = ["Control", "Type I", "Type II"],
    save_path: str = "./feature_importance_plot",
    proba: bool = False
):
    """
    Plot PDP-based feature importances.

    Args:
        feature_importances (Dict[str, Dict[Union[int, str], float]]): Dictionary of feature importances per class.
        class_label (Union[int, str]): Class label for which to plot the feature importances.
        title (str): Title of the plot.
        importance_fraction (float): Fraction of total importance to include in the plot
            Default is 1.0 (100%), which includes all features.
        classes (List[str]): List of class labels. Default is None.
        save_path (str): Path to save the plot. Default is "./feature_importance_plot".
        proba (bool): Whether the model outputs probabilities. Default is False.
    """
    # Determine whether importances are class-specific or overall
    if isinstance(next(iter(feature_importances.values())), dict):
        # Class-specific importances
        if class_label is None:
            raise ValueError("class_label must be specified when plotting class-specific importances.")
        class_importances = {feature: importance[class_label] for feature, importance in feature_importances.items()}
    else:
        # Overall importances
        class_importances = feature_importances

    # Remove the "num__" and "cat__" prefix from feature names
    class_importances = {feature.replace("num__", ""): importance for feature, importance in class_importances.items()}
    class_importances = {feature.replace("cat__", ""): importance for feature, importance in class_importances.items()}

    # Sort features by importance
    sorted_importances = sorted(class_importances.items(), key=lambda x: x[1], reverse=True)

    # Calculate cumulative importance and select features up to the specified fraction
    total_importance = sum(importance for _, importance in sorted_importances)
    cumulative_importance = 0.0
    selected_features = []
    for feature, importance in sorted_importances:
        cumulative_importance += importance
        selected_features.append((feature, importance))
        if cumulative_importance / total_importance >= importance_fraction:
            break
    
    # Define colors for each class
    cmap = plt.get_cmap("tab10")
    class_colors = {cls: cmap(i) for i, cls in enumerate(classes)}
    class_color = class_colors.get(class_label, "plum") if classes is not None else "plum"

    # Plot feature importances
    features_sorted, importances_sorted = zip(*selected_features)
    fig, ax = plt.subplots(figsize=(6, 10))

    # Adjusting the height of the bars to reduce overlap
    ax.barh(features_sorted, importances_sorted, color=class_color, height=0.8)  # Set height to reduce overlap
    
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"{class_label}", fontsize=12)

    ax.set_yticks(np.arange(len(features_sorted)))
    ax.set_yticklabels(features_sorted, fontsize=10, rotation=0)  # Adjust fontsize and rotation for readability
    
    ax.invert_yaxis()
    
    plt.margins(y=0.001)  # This reduces space above and below the bars
    plt.tight_layout()
    # plt.subplots_adjust(left=0.3, right=0.98, top=0.97, bottom=0.07)
    plt.savefig(os.path.join(save_path, f"{class_label}_feature_importance.png"))
    plt.close()


def plot_pdp_grid(
    model: Pipeline, 
    X: pd.DataFrame, 
    X_real: pd.DataFrame,  # Inverse-transformed data for x-axis
    features: List[str], 
    grid_resolution: int = 100, 
    classes: Optional[List[str]] = None,
    save_path: str = "./PDP"
):
    """
    Plot separate figures for each feature, with subplots for each class, showing Partial Dependence Plots (PDP).

    Args:
        model (Pipeline): Trained sklearn model.
        X (pd.DataFrame): DataFrame containing the dataset.
        X_real (pd.DataFrame): DataFrame containing the inverse-transformed dataset.
        features (List[str]): List of numerical feature names.
        grid_resolution (int): Number of points to evaluate PDP. Default is 100.
        classes (List[str]): List of class labels. Default is None.
        save_path (str): Path to save the plots. Default is "./PDP".
    """
    os.makedirs(save_path, exist_ok=True)  # create directory if it doesn't exist
    n_classes = len(classes) if classes is not None else 1

    for i, feature in enumerate(features):
        feature_name = feature.replace("num__", "").replace("/", "_").replace("cat__", "")
        fig, ax = plt.subplots(nrows=1, ncols=n_classes, figsize=(15, 5))
        if n_classes == 1:
            ax = [ax]
        
        for j in range(n_classes):
            class_idx = classes[j] if classes is not None else j
            PartialDependenceDisplay.from_estimator(
                model, X, [feature], ax=ax[j], grid_resolution=grid_resolution, target=class_idx
            )

            ax[j].set_title(f"{feature_name} - {class_idx}", fontsize=14)
            ax[j].set_xlabel(feature_name, fontsize=14)
            ax[j].set_ylabel("Partial dependence", fontsize=14)
            ax[j].tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        # Save the plot
        file_name = f"{feature_name}_pdp.png"
        full_path = os.path.join(save_path, file_name)
        plt.savefig(full_path)
        plt.close()
        # plt.show()


def plot_ale_grid(
    ale_explanation, 
    features: List[str],
    X_real: pd.DataFrame,
    target_names: Optional[List[str]] = None, 
    proba: bool = False,
    save_path: str = "./ALE", 
    n_cols: int = 3,
    sharey: str = 'row',
    tick_font_size: int = 14,  # Font size for tick labels
    label_font_size: int = 14  # Font size for axis labels
):
    """
    Create and save ALE plots for each feature, with subplots for each class.

    Args:
        ale_explanation: The ALE explanation object.
        features (List[str]): List of feature names.
        X_real (pd.DataFrame): DataFrame containing the inverse-transformed (original scale) dataset.
        target_names (List[str]): List of class labels. Defaults to None.
        proba (bool): Whether the model outputs probabilities. Defaults to False.
        save_path (str): Path to save the plots. Defaults to "./ALE".
        n_cols (int): Number of columns in the plot grid. Defaults to 3.
        sharey (str): Whether to share y-axis. Options are 'all', 'row', or None. Defaults to 'all'.
        tick_font_size (int): Font size for tick labels. Defaults to 14.
        label_font_size (int): Font size for axis labels. Defaults to 14.
    """
    os.makedirs(save_path, exist_ok=True)  # create directory if it doesn't exist

    for feature in features:
        feature_name = feature.replace("num__", "").replace("/", "_").replace("cat__", "")

        # Plot ALE
        ax = plot_ale(
            ale_explanation, 
            features=[feature], 
            targets=target_names, 
            n_cols=n_cols, 
            sharey=sharey,
        )

        fig = ax[0, 0].figure

        # Set tick label font size
        ax[0, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)

        # Set axis label font size
        ax[0, 0].set_xlabel(feature_name, fontsize=label_font_size)
        ax[0, 0].set_ylabel("ALE", fontsize=label_font_size)  # Adjust this label accordingly

        # Apply tight layout and save the figure
        fig.tight_layout()
        file_name = f"prob_{feature_name}_ale.png" if proba else f"{feature_name}_ale.png"
        full_path = os.path.join(save_path, file_name)
        fig.savefig(full_path)
        plt.close()


def plot_comparison_feature_importances(
    pdp_importances, ale_importances, permutation_importances,
    class_label: Union[int, str],
    top_n: int = 10,
    save_path: str = "./feature_importance_plot"
):
    """
    Plot comparison of PDP-based, ALE-based, and Permutation-based feature importances.

    Args:
        pdp_importances (Dict[str, Dict[Union[int, str], float]]): PDP-based feature importances.
        ale_importances (Dict[str, Dict[Union[int, str], float]]): ALE-based feature importances.
        permutation_importances (Dict[str, Dict[Union[int, str], float]]): Permutation-based feature importances.
        class_label (Union[int, str]): Class label for which to plot the feature importances.
        title (str): Title of the plot.
        top_n (int): Number of top features to plot.
        save_path (str): Path to save the plot.
    """

    def process_importances(importances):
        """Process feature importances for a given method."""
        if isinstance(next(iter(importances.values())), dict):
            class_importances = {feature: importance[class_label] for feature, importance in importances.items()}
        else:
            class_importances = importances

        class_importances = {feature.replace("num__", "").replace("cat__", ""): importance for feature, importance in class_importances.items()}
        sorted_importances = sorted(class_importances.items(), key=lambda x: x[1], reverse=True)
        return sorted_importances[:top_n]  # Keep only top N features

    # Process the importances for each method
    pdp_top_features = process_importances(pdp_importances)
    ale_top_features = process_importances(ale_importances)
    permutation_top_features = process_importances(permutation_importances)

    # Create subplots to visualise the different feature importances side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))  # 3 columns for 3 plots

    # Plot PDP-based feature importances
    features_pdp, importances_pdp = zip(*pdp_top_features)
    axes[0].barh(features_pdp, importances_pdp, color='sandybrown')
    axes[0].set_title(f"PDP - {class_label}", fontsize=16)
    axes[0].invert_yaxis()

    # Plot ALE-based feature importances
    features_ale, importances_ale = zip(*ale_top_features)
    axes[1].barh(features_ale, importances_ale, color='darkorange')
    axes[1].set_title(f"ALE - {class_label}", fontsize=16)
    axes[1].invert_yaxis()

    # Plot Permutation-based feature importances
    features_permutation, importances_permutation = zip(*permutation_top_features)
    axes[2].barh(features_permutation, importances_permutation, color='gold')
    axes[2].set_title(f"Permutation - {class_label}", fontsize=16)
    axes[2].invert_yaxis()

    # Set labels for x-axis
    for ax in axes:
        ax.set_xlabel("Importance", fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(save_path, f"{class_label}_comparison_feature_importance.png"))
    plt.close()


def plot_ale_second_order_classification(
    model,  # Sklearn pipeline or model
    X: pd.DataFrame,
    X_real: pd.DataFrame,  # The inverse-transformed (real scale) features
    feature_1: str,
    feature_2: str,
    class_index: int = 0,  # Target class index for ALE plot (e.g., 0 for first class)
    proba: bool = False,   # If True, target probabilities; otherwise, logits
    bins: int = 10,        # Number of bins for each feature
    save_path: str = "./ALE_interactions",
    plot_title: str = "Second-Order ALE Plot"
):
    """
    Plot second-order ALE plot for classification by targeting the logit or probability of a single class.

    Args:
        model (Pipeline): Trained sklearn pipeline or model.
        X (pd.DataFrame): DataFrame containing preprocessed features.
        X_real (pd.DataFrame): DataFrame containing inverse-transformed features (original scale).
        feature_1 (str): First feature for interaction.
        feature_2 (str): Second feature for interaction.
        class_index (int): Index of the class to target. Default is 0.
        proba (bool): If True, target the class probabilities; if False, target the logits.
        bins (int): Number of bins to divide each feature range for ALE computation.
        save_path (str): Directory to save the ALE plot.
        plot_title (str): Title of the plot.
    """
    # Ensure the save_path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Custom predictor function to extract either logits or probabilities for a specific class
    def predictor(X):
        net = model.named_steps['model']  # Extract the 'NeuralNetClassifier' from the pipeline
        # Convert X (pandas DataFrame or NumPy array) to a PyTorch tensor
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        logits = net.forward(X_tensor)  # Get the logits using the forward pass
        if proba:
            # If probabilities are needed, apply softmax on the logits
            probabilities = torch.softmax(logits, dim=1)
            return probabilities[:, class_index].detach().numpy()  # Return probabilities for the target class
        else:
            return logits[:, class_index].detach().numpy()  # Return logits for the target class

    # Now call ale_plot with the custom predictor
    ax = ale_plot(
        model=None,               # No need for model since we're passing a custom predictor
        train_set=X,              # Preprocessed data
        features=[feature_1, feature_2],  # Features for 2D ALE plot
        predictor=predictor,      # Custom predictor targeting class_index
        bins=bins,                # Number of bins for each feature
        monte_carlo=False,
    )

    # Customize the colormap to have zero-centered white using TwoSlopeNorm
    cmap = plt.get_cmap("seismic")  # Diverging colormap (e.g., seismic, RdBu) centered at zero

    # Extract the min and max from the plot for normalization
    vmin = ax.collections[0].get_clim()[0]
    vmax = ax.collections[0].get_clim()[1]
    vcenter = 0  # We want zero to be at the center

    # Use TwoSlopeNorm to center zero at white
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # Update the colormap and normalization for all the contours in the plot
    for contour in ax.collections:
        contour.set_cmap(cmap)
        contour.set_norm(norm)

    # Add colorbar with normalization
    fig = ax.get_figure()
    # cbar = fig.colorbar(ax.collections[0], ax=ax)
    # cbar.set_label('ALE Value')

    # Set labels for the features using the real feature names
    ax.set_xlabel(feature_1, fontsize=14)
    ax.set_ylabel(feature_2, fontsize=14)

    # Set the plot title
    ax.set_title(plot_title, fontsize=16)

    # Save the plot
    file_name = f"{feature_1}_{feature_2}_ale_class_{class_index}.png"
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, file_name))
    plt.close()





if __name__ == "__main__":

    model_path = "research/case_study/biomed/models/CF_DNN/apricot-sweep-24/f1_cf_dnn_apricot-sweep-24.pth"
    run_id = "radadjoneva-icl/covid-outcome-classification/2sspqbzv"  # apricot-sweep-24

    api = wandb.Api()
    run = api.run(run_id)
    config = run.config

    X_train = pd.read_csv("research/case_study/biomed/datasets/iCTCF/processed_cf/apricot-sweep-24/input_features_train.csv")
    X_train_real = pd.read_csv("research/case_study/biomed/datasets/iCTCF/processed_cf/apricot-sweep-24/input_features_train_inverse.csv")
    Y_train = pd.read_csv("research/case_study/biomed/datasets/iCTCF/processed_cf/apricot-sweep-24/target_outcomes_train.csv")

    class_list = config["covid_outcome_classes"]
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    num_cols = [col for col in X_train.columns if col.startswith("num")]
    input_dim = len(X_train.columns)

    # Load the model as a skorch model and wrap it in a scikit-learn pipeline
    skorch_model = load_pretrained_model(model_path, config, input_dim=input_dim, skorch=True)
    model_pipeline = Pipeline([
        ("preprocessor", None),
        ("to_tensor", FunctionTransformer(lambda x: torch.tensor(x.values, dtype=torch.float32))),  # convert DataFrame to tensor
        ("model", skorch_model),
    ])

    # # Compute PDP-based feature importances for all numerical features
    # pdp_importances = compute_pdp_feature_importances(model_pipeline, X_train, num_cols, grid_resolution=100, classes=class_list)
    
    # # Path to store results
    # results_path = "research/case_study/biomed/results/interpretability/feature_importance/PDP"

    # for class_label in class_list:
    #     plot_feature_importances(
    #         feature_importances=pdp_importances,
    #         class_label=class_label,
    #         title="PDP-Based Feature Importance for numeric features",
    #         importance_fraction=1.0
    #     )

    # # Plot PDP grid for all numerical features
    # plot_pdp_grid(model_pipeline, X_train, num_cols, grid_resolution=100, classes=class_list, save_path=results_path)

    # Compute permutation-based feature importances
    # permutation_importances = compute_permutation_feature_importances(
    #     model_pipeline, 
    #     X_train, 
    #     Y_train, 
    #     scoring_func=f1_score, 
    #     n_repeats=5, 
    #     classes=class_list, 
    #     mode="one-vs-all", 
    #     custom_class_group=None
    # )

    # # Create path to save results if it doesn't exist
    # results_path = "research/case_study/biomed/results/interpretability/feature_importance/Permutation"
    # os.makedirs(results_path, exist_ok=True)

    # for class_label in class_list:
    #     plot_feature_importances(
    #         feature_importances=permutation_importances[class_label], 
    #         class_label=class_label, 
    #         title="Permutation-Based Feature Importance", 
    #         importance_fraction=1.0,
    #         save_path=results_path
    #     )


    # plot_feature_importances(
    #     feature_importances=permutation_importances, 
    #     class_label="Type I & Type II (vs Control)", 
    #     title="Permutation-Based Feature Importance", 
    #     importance_fraction=1.0
    # )

    results_path = "research/case_study/biomed/results/interpretability/feature_importance/ALE"
    # ale_importances = compute_ale_feature_importances(model_pipeline, X_train, X_train_real, classes=class_list, proba=True, plot=True, save_path=results_path)

    # for class_label in class_list:
    #     plot_feature_importances(
    #         feature_importances=ale_importances,
    #         class_label=class_label,
    #         title="ALE-Based Feature Importance",
    #         importance_fraction=1.0,
    #         proba=True,
    #         save_path=results_path
    #     )


    # Compare top feature importances across methods: PDP, ALE, Permutation for Type I class
    # selected_label = "Type I"
    # plot_comparison_feature_importances(
    #     pdp_importances=pdp_importances,
    #     ale_importances=ale_importances,
    #     permutation_importances=permutation_importances[selected_label],
    #     class_label=selected_label,
    #     top_n=10,  # Limit to top 10 most important features
    #     save_path="research/case_study/biomed/results/interpretability/feature_importance"
    # )

    # ALE second-order
    plot_ale_second_order_classification(
        model=model_pipeline,
        X=X_train,  # The preprocessed or inverse-transformed data
        X_real=X_train_real,  # The original scale data for inverse transformation
        feature_1="num__IFN-γ",  # First feature
        feature_2="num__TNF-α",  # Second feature
        class_index=2,  # Target class (index)
        proba=False,   # If True, target probabilities; otherwise, logits
        bins=10,  # Number of bins for each feature
        save_path=results_path,
        plot_title="num__TNF-α vs num__IFN-γ (Type II)"
    )



    
# Adapted from Arush's notebook: Prototype Generation loop

# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

import torch
import pandas as pd
import wandb
import yaml
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

from src.data.cf_dataset import CFDataset
from src.utils.model_utils import load_pretrained_model
from src.data.dataset_load import load_datasets
from src.utils.interpretability_utils import calculate_diversity
from src.utils.utils import set_seed


# Define input constraint module
class TabularCoerce:
    """Coerce input data: clamp numerical features within a specified range and normalise probabilities for categorical features."""

    def __init__(self, data: CFDataset = None, min=None, max=None, coerce_probs=True, preprocessor=None):
        if preprocessor is not None:
            self.preprocessor = preprocessor
            input_data = data.X.clone()
            num_ix_cols = [ix for ix, col in enumerate(data.input_columns) if "num" in col]
            input_data[:, num_ix_cols] = preprocessor.reverse(input_data[:, num_ix_cols])
        if min is None and max is None:
            # If not provided, calculate min and max from the data
            self.min_clamp = input_data.amin(dim=0)
            # self.min_clamp = torch.clamp(self.min_clamp, min=0)  # Ensure no negative values (imputation with -1) ?
            self.max_clamp = input_data.amax(dim=0)
        else:
            self.min_clamp = min
            self.max_clamp = max

        self.coerce_probs = coerce_probs

        if coerce_probs:
            if data is not None:
                # 'undis' is not taken into account as it's multi-hot encoded
                cat_groups = [
                    col
                    for col in data.preprocessor.cat_cols
                    if col not in data.config["drop_columns"]
                ]
                prob_group_ixs = []
                for c in cat_groups:
                    cur_group_ixs = []
                    for ix, col in enumerate(data.input_columns):
                        if c in col:
                            cur_group_ixs.append(ix)
                    if len(cur_group_ixs) > 0:
                        prob_group_ixs.append(cur_group_ixs)
                self.cat_ix = prob_group_ixs
            else:
                print(
                    "Cannot coerce probabilities without data being provided at instantiation! Skipping..."
                )

    def __call__(self, X):
        if self.coerce_probs:
            X_clone = X.clone()
            for c in self.cat_ix:
                X_clone[:, c] = X[:, c] / torch.sum(X[:, c], dim=-1, keepdim=True)
            X = X_clone
        X = torch.maximum(X, self.min_clamp.to(X.device)) if self.min_clamp is not None else X
        X = torch.minimum(X, self.max_clamp.to(X.device)) if self.max_clamp is not None else X
        return X


class StandardPreprocessor:
    def __init__(self, transformer=None, mean=None, std=None):
        if transformer:
            # Extract the StandardScaler for numerical columns
            self.scaler = transformer.named_transformers_["num"]

            # Ensure the scaler is fitted
            if not hasattr(self.scaler, "mean_"):
                raise ValueError("The provided transformer must be fitted with data.")

            # Store the mean and standard deviation
            self.mean = torch.tensor(self.scaler.mean_, dtype=torch.float32)
            self.std = torch.tensor(self.scaler.scale_, dtype=torch.float32)
        elif mean is not None and std is not None:
            # Use provided mean and std
            self.mean = mean
            self.std = std
        else:
            raise ValueError("Either transformer or both mean and std must be provided.")

    def __call__(self, data):
        std = self.std.to(data.device)
        mean = self.mean.to(data.device)
        return (data - mean) / std

    def reverse(self, data):
        std = self.std.to(data.device)
        mean = self.mean.to(data.device)
        return data * std + mean

    def save_stats(self, filepath):
        """Save the mean and standard deviation to a file."""
        stats = {
            "mean": self.mean.cpu().numpy(),
            "std": self.std.cpu().numpy()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(stats, f)
        print(f"Statistics saved to {filepath}")

    @classmethod
    def load_stats(cls, filepath):
        """Load the mean and standard deviation from a file."""
        with open(filepath, 'rb') as f:
            stats = pickle.load(f)
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)
        return cls(mean=mean, std=std)

# Define prototype generation loop
def generation_loop(
    device,
    epochs,
    model,
    init_input,
    target_ix,
    preprocessing,
    coercion,
    lr=0.01,
    diversity_weight=0,
    grad_mask=None,
    input_columns=None,
    target_columns=None,
    log_interim=False,
    type="classification",
    target_state="max",
):
    """Generate prototypes by optimising an input tensor to maximise class logits and diversity (in the case of multiple prototypes).

    Args:
        - device (str): Device to run the optimisation on.
        - epochs (int): Number of optimisation steps.
        - model (torch.nn.Module): The model to generate predictions.
        - init_input (torch.Tensor): The initial input tensor to be optimised.
        - target_ix (torch.Tensor): Target indices of classes to generate prototypes for.
        - preprocessing (Callable, optional): Function used to preprocess input data to the model (e.g. normalisation).
        - coercion (Callable, optional): Function to coerce input data.
        - lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.
        - diversity_weight (float, optional): Weight of diversity loss. Defaults to 1.
        - grad_mask (torch.Tensor, optional): Mask for gradient application.
        - input_columns (list of str, optional): Column names for logging results.
        - target_columns (list of str, optional): Column names for logging results.
        - log_interim (bool, optional): Flag to log interim results. Defaults to False.
        - type (str, optional): Type of task ('classification' or 'regression').
        - target_state (str, optional): Target state for regression ('max' or 'min').

    Returns:
        - torch.Tensor: The optimized input tensor.
    """
    model.eval()
    model.to(device)
    input = torch.nn.Parameter(init_input.to(device))

    wandb.log({"diversity_weight": diversity_weight})

    # If grad_mask is not provided, allow all gradients
    if grad_mask is None:
        grad_mask = torch.ones_like(input).to(device)

    optimiser = torch.optim.Adam([input], lr=lr)

    prototype_class = target_columns[target_ix[0]]

    # Initialise table to store prototype types for logging
    prototype_table = wandb.Table(columns=["epoch", "prototype_number"] + input_columns + target_columns)

    for e in range(epochs):
        # Preprocessing and coercion
        num_cols_idx = [i for i, c in enumerate(input_columns) if "num" in c]

        # Reverse preprocessing (inverse standardisation) for numerical columns
        t_input = input.clone().to(device)
        t_input[:, num_cols_idx] = preprocessing.reverse(t_input[:, num_cols_idx]) if preprocessing else t_input[:, num_cols_idx]

        # Coercion (Min and Max clamping, Binary categorical sum to 1)
        c_input = coercion(t_input) if coercion else t_input
        c_input = c_input.to(device)

        # Preprocess input data (standardisation for numerical columns)
        p_input = c_input.clone().to(device)
        p_input[:, num_cols_idx] = preprocessing(p_input[:, num_cols_idx]) if preprocessing else p_input[:, num_cols_idx]

        if type == "classification":
            # Compute cross entropy loss and diversity loss
            logits = model(p_input)
            probs = torch.softmax(logits, dim=-1)
            logit_loss = torch.nn.functional.cross_entropy(logits, target_ix.to(device))
            diversity_loss = calculate_diversity(p_input).to(device) * diversity_weight
            loss = (logit_loss + diversity_loss).sum().to(device)
        elif type == "regression":
            logits = model(p_input.to(device))
            if target_state == "max":
                logit_loss = -logits.mean()
            elif target_state == "min":
                logit_loss = logits.mean()

            diversity_loss = calculate_diversity(p_input).to(device)
            loss = (logit_loss + diversity_loss * diversity_weight).sum().to(device)

        # Log metrics to WandB
        logits_dict = {f"{c}_{j}": logits[j, i].item() for j in range(logits.size(0)) for i, c in enumerate(target_columns)}
        probs_dict = {f"{c}_{j}": probs[j, i].item() for j in range(probs.size(0)) for i, c in enumerate(target_columns)}
        mean_logits_dict = {f"{c}_mean": logits[:, i].mean().item() for i, c in enumerate(target_columns)}
        mean_probs_dict = {f"{c}_mean": probs[:, i].mean().item() for i, c in enumerate(target_columns)}

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

        # Backpropagation and optimisation step
        optimiser.zero_grad()
        loss.backward()
        input.grad *= grad_mask
        optimiser.step()

        if (log_interim and e % (epochs // 10) == 0) or e == epochs - 1:
            if type == "classification":
                prototype_numbers = torch.arange(probs.size(0)).unsqueeze(1).to(device)
                epoch = torch.Tensor([e]* probs.size(0)).unsqueeze(1).to(device)
                cat = torch.cat([epoch, prototype_numbers, c_input, probs], dim=-1)
                df = pd.DataFrame(data=cat.detach().cpu().numpy(), columns=["epoch", "prototype_number"] + input_columns + target_columns)
                for row in df.itertuples(index=False, name=None):
                    prototype_table.add_data(*row)
                if e == epochs - 1:
                    wandb.log({f"prototype_probs_{prototype_class}": prototype_table})  # log table at the end of training

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

            elif type == "regression":
                print(
                    f"{e}: Total Loss: {loss.data} Logit Loss: {logit_loss.data} Diversity Loss: {diversity_loss.data}"
                )
                cat = torch.cat([c_input, logits], dim=-1)
                df = pd.DataFrame(data=cat.detach().cpu().numpy(), columns=input_columns + target_columns)
                wandb.log({f"prototype_probs_{prototype_class}": wandb.Table(dataframe=df)})

    return c_input


def run_protogen_tabular():
    protogen_config = {
        "run_name": "apricot-sweep-24",
        "run_id": "radadjoneva-icl/covid-outcome-classification/2sspqbzv",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 100,
        "target_ix": [0],
        "num_prototypes": 5,
        "lr": 0.01,
        "diversity_weight": 0,
        "grad_mask": None,
        "type": "classification",
        "target_state": "max",
        "min_val": None,  # Minimum value for coercion
        "max_val": None,  # Maximum value for coercion
    }
    
    # Initialise WandB
    wandb.init(project="protogen-tabular", config=protogen_config)
    config = wandb.config

    # Load the trained model
    run_name = config["run_name"]
    model_dir = f"research/case_study/biomed/models/CF_DNN/{run_name}"
    model_path = os.path.join(model_dir, f"f1_cf_dnn_{run_name}.pth")
    # eval_random_states_path = os.path.join(model_dir, f"random_states_{run_name}-best-epoch.pth")
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as file:
        model_config = yaml.safe_load(file)
    
    # Update WandB config with classifier config, preserving original WandB config values
    for key, value in model_config.items():
        if key not in wandb.config:  # Only update if key is not already in WandB config
            wandb.config[key] = value

    # Set random seed for reproducibility (based on config)
    set_seed(config["seed"])

    # Load datasets
    print(f"\nLoading datasets for model {config['model']} ...")
    print(f"Device: {config['device']}")
    train_dataset, val_dataset, test_dataset = load_datasets(config)

    # Parameters 
    class_list = config["covid_outcome_classes"]
    input_dim = len(train_dataset.input_columns)
    config["input_dim"] = input_dim

    # Load the model for evaluation
    cf_dnn = load_pretrained_model(model_path, config, input_dim=input_dim, skorch=False)
    # Set model to evaluation mode (just in case)
    cf_dnn.to(config["device"])
    cf_dnn.eval()

    # eval_random_state = torch.load(eval_random_states_path)
    # restore_random_state(eval_random_state)

    # init_input = torch.normal(mean=mu, std=sigma).to(device)

    # target_ix = torch.tensor(config["target_ix"] * num_p).to(device)

    # Generate prototypes for maximum target value
    device = config["device"]
    num_p = config["num_prototypes"]
    init_input = torch.rand(num_p, input_dim).to(device)
    target_ix = torch.tensor(config["target_ix"] * num_p).to(device)
    preprocessor = StandardPreprocessor(train_dataset.preprocessor.transformer)
    preprocessor.save_stats("research/case_study/biomed/datasets/iCTCF/standardise_stats.pkl")  # Save stats for preprocessing (mean, std)
    # If min max None, will be calculated from the data
    coercion = TabularCoerce(data=train_dataset, min=None, max=None, preprocessor=preprocessor)
    prototype = generation_loop(
        config["device"],
        config["epochs"],
        cf_dnn,
        init_input.clone().to(config["device"]),
        target_ix=target_ix,
        preprocessing=preprocessor,
        coercion=coercion,
        lr=config["lr"],
        diversity_weight=config["diversity_weight"],
        grad_mask=config["grad_mask"],
        input_columns=train_dataset.input_columns,
        target_columns=class_list,
        log_interim=True,
        type=config["type"],
        target_state=config["target_state"],
    )

    print(prototype.var(dim=0).mean())

    print("Done!")
    wandb.finish()



def merge_prototypes(prototype_csv=[], filename="all_prototypes.csv"):
    """Merge prototype CSVs into a single DataFrame.

    Args:
        prototype_csv (list of str): List of paths to prototype CSVs.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    if len(prototype_csv) == 0:
        raise ValueError("No prototype CSVs provided for merging.")

    # Load the first CSV
    df = pd.read_csv(prototype_csv[0])

    # Merge the remaining CSVs
    for p in prototype_csv[1:]:
        df = pd.concat([df, pd.read_csv(p)])
    
    # Save the merged DataFrame
    df.to_csv(filename, index=False)

    return df



def plot_prototypes_heatmap(
    prototypes_path: str, 
    train_data_path: str, 
    figsize=(15, 8), 
    cmap="coolwarm", 
    save_path=None
):
    """
    Plot a heatmap to compare the prototypes (rows) across features (columns),
    with values inverse-transformed based on the statistics of the real data.

    Args:
        prototypes_path (str): Path to the CSV file containing the prototypes.
        train_data_path (str): Path to the CSV file containing the real training data.
        figsize (tuple): Size of the heatmap figure.
        cmap (str): Colormap to use for the heatmap.
        save_path (str): If provided, save the heatmap as an image at this path.
    """
    # Load the prototypes from the CSV file
    prototypes_df = pd.read_csv(prototypes_path)
    
    # Load the real training data to calculate mean and std
    train_data = pd.read_csv(train_data_path)

    # Remove class columns
    prototypes_df = prototypes_df.drop(columns=["Control", "Type I", "Type II"])
    
    # Calculate the mean and standard deviation of the training data
    mean = train_data.mean()
    std = train_data.std()
    
    # Normalise the prototype values using the mean and std from the real data
    inverted_prototypes = (prototypes_df - mean) / std
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(inverted_prototypes, annot=False, cmap=cmap, linewidths=0.5)
    
    # Add title and labels
    plt.title("Prototypes Heatmap (Inverted to Real Scale)", fontsize=16)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Classes", fontsize=12)
    
    # Remove the column labels and numbers in the heatmap
    plt.xticks([], [])
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_prototypes_bar_charts(
    prototypes_path: str, 
    features: list = None,  # List of specific features to plot (default: all)
    figsize=(12, 12),  # Adjusted to make subplots more square
    save_path=None
):
    """
    Create bar charts to compare the prototypes (rows) across selected features (columns).

    Args:
        prototypes_path (str): Path to the CSV file containing the prototypes.
        features (list): List of feature names to plot. If None, plot all features.
        figsize (tuple): Size of the figure.
        save_path (str): If provided, save the plot as an image at this path.
    """
    # Load the prototypes from the CSV file
    prototypes_df = pd.read_csv(prototypes_path)

    # Get the class names
    class_names = ["Control", "Type I", "Type II"]
    
    # If no features are specified, use all features
    if features is None:
        features = prototypes_df.drop(columns=class_names).columns.tolist()
    
    # Calculate the number of rows and columns for subplots
    num_features = len(features)
    num_cols = 3  # We can fit 3 subplots per row
    num_rows = (num_features + num_cols - 1) // num_cols  # Round up the number of rows

    # Create the figure and axes for subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case of multiple rows/columns

    # Plot bar charts for each feature
    for i, feature in enumerate(features):
        if feature not in class_names:
            # Get the feature values for each class
            values = prototypes_df.loc[:, feature]

            min_value = values.min()
            max_value = values.max()
            y_margin = 0.2 * (max_value - min_value)  # 10% of the range
            y_min = min_value - y_margin
            y_max = max_value + y_margin

            # Create bar chart
            axes[i].bar(class_names, values, color=['royalblue', 'darkorange', 'forestgreen'])
            axes[i].set_title(feature)

            axes[i].set_ylim(y_min, y_max)

            axes[i].set_xticklabels([])
            plt.setp(axes[i].get_yticklabels(), fontsize=14)
            # axes[i].set_ylabel('Value')
            # axes[i].set_xlabel('Class')

    # Remove any unused subplots
    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout to add more vertical space between plots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.4)  # More space between subplots

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
    plt.close()
    # plt.show()


if __name__ == "__main__":
    # run_protogen_tabular()

    # # Tabular protytype list
    # prototype_csv = [
    #     "research/case_study/biomed/results/interpretability/prototypes/tab_prototypes/Control_prototype_single_neat-sweep-1.csv",
    #     "research/case_study/biomed/results/interpretability/prototypes/tab_prototypes/TypeI_prototype_single_morning-sweep-1.csv",
    #     "research/case_study/biomed/results/interpretability/prototypes/tab_prototypes/TypeII_prototype_single_unique-sweep-1.csv"
    # ]
    filename = "research/case_study/biomed/results/interpretability/prototypes/tab_prototypes/all_prototypes.csv"

    # merge_prototypes(prototype_csv, filename)

    # # Diverse prototype generation
    # num_p = 4
    # init_input = train_dataset.X.mean(dim=0).unsqueeze(0).repeat(num_p, 1)
    # init_input = torch.rand_like(init_input)
    # coercion = TabularCoerce(data=train_dataset)
    # epochs = 1000
    # prototype = generation_loop(
    #     epochs,
    #     model,
    #     init_input.clone(),
    #     target_ix=torch.tensor([1] * num_p),
    #     preprocessing=prep,
    #     coercion=coercion,
    #     lr=0.001,
    #     diversity_weight=1,
    #     grad_mask=None,
    #     columns=train_dataset.columns,
    #     log_interim=False,
    #     type='regression',
    #     target_state='max'
    # )

    # print(prototype.var(dim=0).mean())


    # Compare prototypes between classes
    train_data_path = "research/case_study/biomed/datasets/iCTCF/processed_cf/apricot-sweep-24/input_features_train_inverse.csv"
    # save_path = os.path.join(os.path.dirname(filename),"prototypes_heatmap.png")
    # plot_prototypes_heatmap(filename, train_data_path, save_path=save_path)

    list_features = ["num__Age","num__Body temperature","num__Platelet count",
                     "num__Eosinophil count","num__Lymphocyte count","num__Neutrophil count",
                     "num__Erythrocyte sedimentation rate","num__C-reactive protein","num__Procalcitonin","num__D-Dimer",
                     "num__Albumin/Globulin ratio","num__Albumin",
                     "num__Alkaline phosphatase","num__Alanine aminotransferase",
                     "num__Urea nitrogen","num__Calcium","num__Creatinine",
                     "num__Potassium","num__Magnesium","num__Sodium","num__Phosphorus",
                     "num__CD3+ T cell","num__CD4+ T cell","num__CD8+ T cell","num__B lymphocyte","num__Natural killer cell","num__CD4/CD8 ratio",
                     "num__Interleukin-2","num__Interleukin-4","num__Interleukin-6","num__Interleukin-10","num__TNF-α","num__IFN-γ"]

    save_path = os.path.join(os.path.dirname(filename),"prototypes_barchart.png")
    plot_prototypes_bar_charts(filename, features=list_features, save_path=save_path)
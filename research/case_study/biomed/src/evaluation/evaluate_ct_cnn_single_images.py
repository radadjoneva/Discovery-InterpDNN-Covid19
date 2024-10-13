# ruff: noqa: E402
# ruff: noqa: I001

import os

# Set the environment variable for CuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import torch
import yaml

import wandb


# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.evaluation.performance_metrics import evaluate_performance
from src.utils.evaluation_utils import print_performance
from src.model.model_optim_prep import prepare_model_and_optimizer
from src.utils.utils import restore_random_state, set_seed
from src.data.prep_data_loaders import load_datasets_and_initialize_loaders
from src.data.ct_dataset import CovidCTDataset
from torch.utils.data import DataLoader, ConcatDataset


def get_logged_value(run_id, key):
    # Initialize WandB API and get the run
    api = wandb.Api()
    run = api.run(run_id)

    # Fetch the history of logged values
    history = run.history(keys=[key])

    # Get the last logged value of the specified key
    if not history.empty:
        value = history[key].iloc[-1]
        return value
    else:
        print(f"No data found for key: {key}")
        return None
    

def evaluate_single_images(model, loader, device, classes):
    """Evaluate model on single images (NiCT, pCT, nCT) and log proportions."""
    # Initialize counters
    counters = {
        'NiCT': {'Control': 0, 'Type I': 0, 'Type II': 0},
        'pCT': {'Control': 0, 'Type I': 0, 'Type II': 0},
        'nCT': {'Control': 0, 'Type I': 0, 'Type II': 0}
    }
    total_counts = {'NiCT': 0, 'pCT': 0, 'nCT': 0}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)

            # Map true labels and predictions to their corresponding classes
            true_labels = labels.argmax(1).cpu().numpy()
            predicted_labels = outputs.argmax(1).cpu().numpy()

            for true_label, pred_label in zip(true_labels, predicted_labels):
                true_class = classes[true_label]
                pred_class = ["Control", "Type I", "Type II"][pred_label]

                counters[true_class][pred_class] += 1
                total_counts[true_class] += 1

    # Calculate and log proportions
    proportions = {}
    for true_class, counts in counters.items():
        proportions[true_class] = {pred_class: counts[pred_class] / total_counts[true_class] 
                                   for pred_class in counts}
    
    # Print results
    for true_class, pred_proportions in proportions.items():
        print(f"\nProportions for true class {true_class}:")
        for pred_class, proportion in pred_proportions.items():
            count = counters[true_class][pred_class]
            total = total_counts[true_class]
            print(f"  Predicted as {pred_class}: {proportion:.2%} ({count}/{total})")

    wandb.log(proportions)


def evaluate_per_patient_ct_scans(model, loaders, criterion, target_classes, config):
    """Evaluate model on per-patient CT scans (100 individual images per patient)."""
    for split, loader in loaders.items():
        metrics = evaluate_performance(
            config,
            model,
            loader,
            criterion,
            target_classes,
            split=split,
            plots=True,
            best_model=""
        )
        print_performance(metrics, split.capitalize())


def evaluate_ct_cnn_single_images(run_id, model_filename, init_random_states_path=None, eval_random_states_path=None):
    # Initialise WandB API and get the run
    api = wandb.Api()
    run = api.run(run_id)

    # Print Model performance (evaluated at the end of training run)
    print(f"Model performance for: {os.path.basename(model_filename)}")
    metrics = [
        "_F1_Score_best_val_f1",
        "_Loss_best_val_f1",
        "_Accuracy_best_val_f1",
        "_Overall_ROC_AUC_best_val_f1",
    ]
    splits = ["val", "train"]
    for split in splits:
        print(f"\nPerformance on {split} set:")
        for metric in metrics:
            metric = split + metric
            value = get_logged_value(run_id, metric)
            print(f"{metric}: {value:.4f}")

    # Load the configuration
    config = run.config
    # Save config as yaml
    config_path = os.path.join(os.path.dirname(model_filename), "config.yaml")
    with open(config_path, "w") as file:
        yaml.dump(config, file)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # config["num_workers"] = 0  # Set to 0 for cpu

    config["run_id"] = run_id
    config["model_filename"] = model_filename
    config["init_random_states_path"] = init_random_states_path
    config["eval_random_states_path"] = eval_random_states_path

    # Initialise WandB
    wandb.init(project="covid-outcome-classification-analysis", notes="single_image_performance_ct_cnn", config=config)
    
    random_states = wandb.Artifact(f"random_states_{run.name}", type="random_states")
    random_states.add_file(init_random_states_path)
    random_states.add_file(eval_random_states_path)
    wandb.log_artifact(random_states)

    # Set random seed for reproducibility
    seed = config["seed"]
    set_seed(seed)

    # SINGLE IMAGES NiCT, pCT, nCT
    config["model"] = "ct_images_cnn"
    config["data_augmentation"] = False

    # Load datasets and initialize DataLoaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_datasets_and_initialize_loaders(config, init_random_states_path)

    # Revert model name
    config["model"] = "resnet50"

    # Prepare model and criterion
    model, criterion, _ = prepare_model_and_optimizer(config, input_dim=None)

    if eval_random_states_path:
        eval_random_state = torch.load(eval_random_states_path)
        restore_random_state(eval_random_state)

    # Load the model state dict
    state_dict = torch.load(model_filename, map_location=config["device"])
    model.load_state_dict(state_dict, strict=True)
    model.to(config["device"])
    model.eval()

    # Evaluate model on single images (NiCT, pCT, nCT)
    print("\nEvaluating single images performance...")
    evaluate_single_images(model, train_loader, config["device"], train_dataset.classes)

    # SINGLE IMAGES 100 per patient
    config["data_augmentation"] = False
    config["k_imgs"] = 100  # 100 CT scans for the 60% middle ones or the total 60% of the slices
    config["batch_size"] = 1

    # Load datasets and initialize DataLoaders for per-patient evaluation
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_datasets_and_initialize_loaders(config, init_random_states_path)
    
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    # Evaluate model on per-patient CT scans
    print("\nEvaluating per-patient CT scans...")
    evaluate_per_patient_ct_scans(model, loaders, criterion, train_dataset.classes, config)

    wandb.finish()


if __name__ == "__main__":
    # CT_CNN: wordly-sweep-4
    run_name = "worldly-sweep-4"
    run_id = "radadjoneva-icl/covid-outcome-classification/c3hpiq15"  # wordly-sweep-4
    model_dir = f"research/case_study/biomed/models/CT_CNN/{run_name}"
    model_path = os.path.join(model_dir, f"f1_resnet50_{run_name}.pth")
    init_random_states_path = os.path.join(model_dir, f"random_states_{run_name}.pth")
    eval_random_states_path = os.path.join(model_dir, f"random_states_{run_name}-best-epoch.pth")

    evaluate_ct_cnn_single_images(run_id, model_path, init_random_states_path=init_random_states_path, eval_random_states_path=eval_random_states_path)

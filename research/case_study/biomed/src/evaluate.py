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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.evaluation.performance_metrics import evaluate_performance
from src.utils.evaluation_utils import print_performance
from src.model.model_optim_prep import prepare_model_and_optimizer
from src.utils.utils import restore_random_state, set_seed
from src.data.prep_data_loaders import load_datasets_and_initialize_loaders


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


def evaluate(run_id, model_filename, init_random_states_path=None, eval_random_states_path=None):
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
    wandb.init(project="covid-outcome-classification-analysis", notes=model_filename, config=config)
    
    random_states = wandb.Artifact(f"random_states_{run.name}", type="random_states")
    random_states.add_file(init_random_states_path)
    random_states.add_file(eval_random_states_path)
    wandb.log_artifact(random_states)

    # Set random seed for reproducibility
    seed = config["seed"]
    set_seed(seed)

    config["data_augmentation"] = False

    # Load datasets and initialize DataLoaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_datasets_and_initialize_loaders(config, init_random_states_path)

    # Prepare model and optimizer
    input_dim = None
    if config["model"] == "cf_dnn":
        input_dim = len(train_dataset.input_columns)
    elif config["model"] == "multimodal_fusion":
        input_dim = len(train_dataset.cf_dataset.input_columns)

    # Prepare model and criterion
    model, criterion, _ = prepare_model_and_optimizer(config, input_dim)

    if eval_random_states_path:
        eval_random_state = torch.load(eval_random_states_path)
        restore_random_state(eval_random_state)

    # Load the model state dict
    state_dict = torch.load(model_filename, map_location=config["device"])
    model.load_state_dict(state_dict, strict=True)
    model.to(config["device"])
    model.eval()

    # Evaluate model
    print("\nEvaluating performance...")
    target_classes = train_dataset.classes

    test_metrics = evaluate_performance(
        config,
        model,
        test_loader,
        criterion,
        target_classes,
        split="test",
        plots=True,
        best_model=None,
    )
    val_metrics = evaluate_performance(
        config,
        model,
        val_loader,
        criterion,
        target_classes,
        split="val",
        plots=True,
        best_model=None,
    )
    train_metrics = evaluate_performance(
        config,
        model,
        train_loader,
        criterion,
        target_classes,
        split="train",
        plots=True,
        best_model=None,
    )

    # Print performance metrics
    print_performance(test_metrics, "Test")
    print_performance(val_metrics, "Validation")
    print_performance(train_metrics, "Train")

    print("Model Evaluation complete!")

    wandb.finish()


if __name__ == "__main__":
    # CF_DNN: apricot-sweep-24
    run_name = "apricot-sweep-24"
    run_id = "radadjoneva-icl/covid-outcome-classification/2sspqbzv"  # apricot-sweep-24
    model_dir = f"research/case_study/biomed/models/CF_DNN/{run_name}"
    model_path = os.path.join(model_dir, f"f1_cf_dnn_{run_name}.pth")
    init_random_states_path = os.path.join(model_dir, f"random_states_{run_name}.pth")
    eval_random_states_path = os.path.join(model_dir, f"random_states_{run_name}-best-epoch.pth")

    # CT_CNN: wordly-sweep-4
    # run_name = "worldly-sweep-4"
    # run_id = "radadjoneva-icl/covid-outcome-classification/c3hpiq15"  # wordly-sweep-4
    # model_dir = f"research/case_study/biomed/models/CT_CNN/{run_name}"
    # model_path = os.path.join(model_dir, f"f1_resnet50_{run_name}.pth")
    # init_random_states_path = os.path.join(model_dir, f"random_states_{run_name}.pth")
    # eval_random_states_path = os.path.join(model_dir, f"random_states_{run_name}-best-epoch.pth")

    evaluate(run_id, model_path, init_random_states_path=init_random_states_path, eval_random_states_path=eval_random_states_path)

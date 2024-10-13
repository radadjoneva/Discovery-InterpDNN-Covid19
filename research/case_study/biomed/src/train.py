# ruff: noqa: E402
# ruff: noqa: I001

import os

# Set the environment variable for CuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import argparse
import torch
import copy
import random
import numpy as np

import wandb

from torch.utils.data import DataLoader

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.data.dataset_load import load_datasets
from src.evaluation.performance_metrics import evaluate_performance
from src.model.model_trainer import ModelTrainer
from src.utils.evaluation_utils import print_performance
from src.model.model_optim_prep import prepare_model_and_optimizer
from src.data.eda import print_and_log_class_distribution
from src.utils.utils import log_random_states, log_final_random_state, set_seed

# from config import load_config
api = wandb.Api()


def parse_args():
    parser = argparse.ArgumentParser(description="covid-outcome-classification")

    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--experiment", type=str, default="", help="Experiment name")

    # Device configuration
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computation")

    # Experiment configuration
    # --------------------------------
    # --------------------------------
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="cf_dnn",
        choices=[
            "ct_images_cnn",
            "ct_patient_cnn",
            "cf_dnn",
            "vgg16",
            "vgg19",
            "resnet50",
            "multimodal_fusion",
        ],
        help="Model to use",
    )
    parser.add_argument(
        "--pretrained", type=bool, default=True, help="Use pretrained model"
    )  # for VGG, ResNet, Inception
    parser.add_argument(
        "--cf_dnn_path",
        type=str,
        default=(
            "research/case_study/biomed/models/CF_DNN/auc_cf_dnn_vibrant-sweep-41.pth",
            "radadjoneva-icl/covid-outcome-classification/o5ho73xa",
        ),
    )
    parser.add_argument(
        "--ct_resnet_path",
        type=str,
        default=(
            "research/case_study/biomed/models/CT_CNN/best_loss_resnet50_desert-sweep-14.pth",
            "radadjoneva-icl/covid-outcome-classification/d51yh439",
        ),
    )
    parser.add_argument(
        "--fusion_type", type=str, default="late_post_classifier", help="Fusion type"
    )  # late_pre_classifier, late_post_classifier
    parser.add_argument(
        "--freeze_layers", type=bool, default=True, help="Freeze layers after loading weights"
    )
    parser.add_argument(
        "--pretrained_models", type=bool, default=True, help="Use pretrained models"
    )

    # --------------------------------
    # --------------------------------
    # Dataset configuration
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--split_ratio",
        type=float,
        nargs=3,
        default=[0.7, 0.2, 0.1],
        help="Train, val, test split ratios",
    )
    parser.add_argument(
        "--covid_outcome_classes",
        type=str,
        nargs="+",
        default=["Control", "Type I", "Type II"],
        help="COVID outcome classes",
    )
    parser.add_argument(
        "--outcome",
        type=str,
        default="Morbidity outcome",
        choices=["Morbidity outcome"],
        help="Outcome to predict",
    )  # Mortality outcome?
    # --------------------------------
    # CT scans dataset (patient and single images)
    parser.add_argument(
        "--ct_patient_dir",
        type=str,
        default="research/case_study/biomed/datasets/iCTCF/CT",
        help="Directory for CT images per patient",
    )
    parser.add_argument(
        "--ct_img_dir",
        type=str,
        default="research/case_study/biomed/datasets/iCTCF/single_images",
        help="Root directory for single CT images",
    )
    parser.add_argument(
        "--ct_classes", type=str, nargs="+", default=["NiCT", "pCT", "nCT"], help="CT classes"
    )
    parser.add_argument("--k_imgs", type=int, default=10, help="K images per patient")
    parser.add_argument(
        "--single_channel", type=bool, default=False, help="True if single channel per patient"
    )
    # parser.add_argument("--3D", type=bool, default=False, help="3D CT scans use SIZ algorithm")
    # Image preprocessing and augmentation
    parser.add_argument(
        "--normalise_pixels",
        type=str,
        default=None,
        choices=["standardise", None],
        help="Normalise CT images",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=(224, 224),
        choices=[(200, 200), (512, 512), (224, 224)],
        help="Resize dimensions for the images",
    )
    parser.add_argument("--data_augmentation", type=bool, default=False, help="Data augmentation")
    parser.add_argument(
        "--transform",
        type=str,
        nargs="+",
        default=["random_crop", "rotate", "brightness"],
        help="Transform functions for Data Augmentation (e.g., random_crop, rotate, brightness)",
    )
    parser.add_argument(
        "--crop_prob", type=float, default=0.5, help="Probability of applying random resize crop"
    )
    parser.add_argument(
        "--transform_prob",
        type=float,
        default=0.2,
        help="Probability of applying the transform functions",
    )
    parser.add_argument(
        "--extract_lung_parenchyma", type=bool, default=False, help="Extract lung parenchyma"
    )  # original paper (not use?)
    parser.add_argument(
        "--crop_margin", type=bool, default=False, help="Crop margin"
    )  # original paper (not use?)
    # --------------------------------
    # Clinical features dataset
    # parser.add_argument('--raw_cf_data', type=str, default='research/case_study/biomed/datasets/iCTCF/raw_cf_data', help='Path to raw clinical features data')
    parser.add_argument(
        "--cleaned_cf_data",
        type=str,
        default="research/case_study/biomed/datasets/iCTCF/cleaned_cf_data.csv",
        help="Path to cleaned clinical features data",
    )
    # Preprocessing for model input
    parser.add_argument(
        "--drop_columns",
        type=str,
        nargs="+",
        default=[
            "Patient ID",
            "Hospital",
            "SARS-CoV-2 nucleic acids",
            "Computed tomography (CT)",
            "Morbidity outcome",
            "Mortality outcome",
        ],
        help="Columns to drop for model input",
    )
    parser.add_argument("--ct_required", type=bool, default=True, help="CT data required")
    parser.add_argument(
        "--undis_binary", type=bool, default=True, help="Turn underlying diseases into binary"
    )
    parser.add_argument(
        "--remove_suspected", type=bool, default=True, help="Remove suspected covid cases"
    )
    # Normalisation and encoding
    parser.add_argument(
        "--normalisation",
        type=str,
        default="standardise",
        choices=["standardise", "minmax"],
        help="Normalization strategy",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="onehot",
        choices=["onehot"],
        help="Encoding strategy",
    )
    # NaN handling
    parser.add_argument(
        "--nan_strategy",
        type=str,
        default="mean",
        choices=["mean", "median", "iterative_imputer", "knn_imputer", "miss_forest"],
        help="NaN strategy",
    )
    parser.add_argument(
        "--max_iter", type=int, default=10, help="Max iterations for iterative imputer"
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="bayesian_ridge",
        choices=["bayesian_ridge", "rand_forest", "k_neighbour"],
        help="Estimator for Iterative Imputer",
    )
    parser.add_argument("--init_strategy", type=str, default="mean", help="Initialisation strategy")
    parser.add_argument("--n_neighbours", type=int, default=20, help="Number of nearest neighbours")
    # TODO Check if it works?!
    parser.add_argument(
        "--missing_indicator", type=bool, default=False, help="Missingness indicator"
    )
    parser.add_argument(
        "--missingness_threshold",
        type=float,
        default=0.5,
        help="NaN threshold above which a missingness indicator is added",
    )

    # --------------------------------
    # --------------------------------
    # Model training configuration
    parser.add_argument("--criterion", type=str, default="cross_entropy", help="Loss criterion")
    parser.add_argument(
        "--class_weights", type=float, nargs=3, default=[10 / 3, 2, 5], help="Class proportions"
    )  # None if no weights, inverse of class proportions
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer"
    )
    # TODO label smoothing? e.g. 0.1, encourage model to be less confident in its predictions and prevent overfitting to training data
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default=None,
        choices=["plateau", "step", "exponential"],
        help="Learning scheduler",
    )
    parser.add_argument("--lr_factor", type=float, default=0.1, help="Factor for LR scheduler")
    parser.add_argument("--lr_patience", type=int, default=10, help="Patience for LR scheduler")
    parser.add_argument("--lr_step_size", type=int, default=30, help="Step size for LR scheduler")
    parser.add_argument("--lr_gamma", type=float, default=0.9, help="Gamma for LR scheduler")

    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--batch_norm", type=bool, default=False, help="Batch Normalisation"
    )  # For CF_DNN
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="research/case_study/biomed/results/model_training/",
        help="Path to save the model",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers for DataLoader"
    )

    return parser.parse_args()


def train():
    for run in range(1):
        # Initialise Weights & Biases
        # run_name = config["experiment"] + f"{run}-" + config["model"]
        run = wandb.init(
            project="covid-outcome-classification", notes=f"run-{run}", config=vars(parse_args())
        )
        config = wandb.config

        # Set random seed for reproducibility
        seed = config["seed"]
        set_seed(seed)

        # Log random states
        log_random_states(config)

        print(f"Run {config['runs']}")

        # Load datasets
        print(f"\nLoading datasets for model {config['model']} ...")
        print(f"Device: {config['device']}")
        train_dataset, val_dataset, test_dataset = load_datasets(config)

        # Print class distributions
        print_and_log_class_distribution(train_dataset, "Dataset", "train")
        print_and_log_class_distribution(val_dataset, "Dataset", "val")

        # Prepare model and optimizer
        input_dim = None
        if config["model"] == "cf_dnn":
            input_dim = len(train_dataset.input_columns)
        elif config["model"] == "multimodal_fusion":
            input_dim = len(train_dataset.cf_dataset.input_columns)

        model, criterion, optimizer = prepare_model_and_optimizer(config, input_dim)

        def worker_init_fn(worker_id):
            # Set the seed for each worker
            np.random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)
            random.seed(seed + worker_id)

        # Initialise DataLoaders and ModelTrainer
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        trainer = ModelTrainer(
            config=config,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        # Train model
        print(f"\nTraining model: {config['model']}...")
        trainer.train()

        # Evaluate model
        print("\nEvaluating performance...")
        target_classes = train_dataset.classes

        # Load best models
        auc_model_path = os.path.join(
            config["save_model_path"], f"auc_{config['model']}_{wandb.run.name}.pth"
        )
        loss_model_path = os.path.join(
            config["save_model_path"], f"loss_{config['model']}_{wandb.run.name}.pth"
        )
        f1_model_path = os.path.join(
            config["save_model_path"], f"f1_{config['model']}_{wandb.run.name}.pth"
        )

        # Load the state dictionary for the best AUC model
        if os.path.exists(auc_model_path):
            state_dict = torch.load(auc_model_path)
            model.load_state_dict(state_dict)
            auc_best_model = copy.deepcopy(model)
        else:
            auc_best_model = copy.deepcopy(model)

        # Load the state dictionary for the best loss model
        if os.path.exists(loss_model_path):
            state_dict = torch.load(loss_model_path)
            model.load_state_dict(state_dict)
            loss_best_model = copy.deepcopy(model)
        else:
            loss_best_model = copy.deepcopy(model)

        # Load the state dictionary for the best f1 model
        if os.path.exists(f1_model_path):
            state_dict = torch.load(f1_model_path)
            model.load_state_dict(state_dict)
            f1_best_model = copy.deepcopy(model)
        else:
            f1_best_model = copy.deepcopy(model)

        # Before evaluating, log final random states
        log_final_random_state(config)

        # Evaluate model on test, validation, and train sets
        for best_metric, model in zip(
            ["best_val_auc", "best_val_loss", "best_val_f1"],
            [auc_best_model, loss_best_model, f1_best_model],
        ):
            print(f"\nEvaluating performance for {best_metric} model...")
            val_metrics = evaluate_performance(
                config,
                model,
                val_loader,
                criterion,
                target_classes,
                split="val",
                plots=True,
                best_model=best_metric,
            )
            train_metrics = evaluate_performance(
                config,
                model,
                train_loader,
                criterion,
                target_classes,
                split="train",
                plots=True,
                best_model=best_metric,
            )

            # Print performance metrics
            print_performance(val_metrics, "Validation")
            print_performance(train_metrics, "Train")

        print("Done!")

        wandb.finish()


if __name__ == "__main__":
    train()

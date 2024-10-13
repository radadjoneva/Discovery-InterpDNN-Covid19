# ruff: noqa: E402
# ruff: noqa: I001

import os

# Set the environment variable for CuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import wandb

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.train import train

cf_dnn_sweep_config = {
    "name": "final_cf_dnn_bayes",
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "run": {"values": [1]},
        "device": {"values": ["cuda"]},
        "seed": {"values": [42]},
        "model": {"values": ["cf_dnn"]},
        "optimizer": {"values": ["adam"]},
        "num_epochs": {"values": [100]},
        "batch_size": {"values": [64]},
        "learning_rate": {"values": [0.009908664820605284]},
        "lr_gamma": {"values": [0.9846158827902316]},
        "weight_decay": {"values": [0.0037300754185376137]},
        "dropout": {"values": [0.5]},
        "batch_norm": {"values": [True]},
        "num_workers": {"values": [10]},
        "normalisation": {"values": ["standardise"]},
        "nan_strategy": {"values": [-1]},
        "max_iter": {"values": [20]},
        "estimator": {"values": ["rand_forest"]},
        "init_strategy": {"values": ["mean"]},
        "n_neighbours": {"values": [None]},
        "ct_required": {"values": [False]},
        "undis_binary": {"values": [False]},
        "remove_suspected": {"values": [True]},
    },
}

ct_images_cnn_sweep_config = {
    "name": "exp1_ct_images_cnn",
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "device": {"values": ["cuda"]},
        "model": {"values": ["ct_images_cnn"]},
        "optimizer": {"values": ["adam"]},
        "num_epochs": {"values": [300]},
        "batch_size": {"values": [64]},
        "learning_rate": {"values": [0.001]},
        "weight_decay": {"values": [0.05]},
        "dropout": {"values": [0.5]},
        "normalise_pixels": {"values": [None]},
        "resize": {"values": [(200, 200)]},
        "data_augmentation": {"values": [False, True]},
        "transform": {"values": [["random_crop", "rotate", "brightness"]]},
        "transform_prob": {"values": [0.5]},
        "extract_lung_parenchyma": {"values": [False]},
        "crop_margin": {"values": [False]},
    },
}

ct_patient_cnn_sweep_config = {
    "name": "exp1_ct_patient_cnn",
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "device": {"values": ["cuda"]},
        "model": {"values": ["ct_patient_cnn"]},
        "optimizer": {"values": ["adam"]},
        "num_epochs": {"values": [300]},
        "batch_size": {"values": [64]},
        "learning_rate": {"values": [0.0007]},
        "weight_decay": {"values": [0.05]},
        "dropout": {"values": [0.5]},
        "normalise_pixels": {"values": [None]},
        "resize": {"values": [(200, 200)]},
        "data_augmentation": {"values": [False]},
        "transform": {"values": [["random_crop", "rotate", "brightness"]]},
        "transform_prob": {"values": [0.5]},
        "k_imgs": {"values": [10]},
        "extract_lung_parenchyma": {"values": [False]},
        "crop_margin": {"values": [False]},
        "ct_required": {"values": [True]},
        "undis_binary": {"values": [True]},
        "remove_suspected": {"values": [True]},
    },
}


resnet50_sweep_config = {
    "name": "final_resnet50_bayes",
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "run": {"values": [1]},
        "device": {"values": ["cuda"]},
        "seed": {"values": [42]},
        "model": {"values": ["resnet50"]},
        "pretrained": {"values": [True]},
        "optimizer": {"values": ["adam"]},
        "num_epochs": {"values": [100]},
        "batch_size": {"values": [64]},
        "learning_rate": { "values": [0.00002121836163969912]},
        "lr_scheduler": {"values": ["exponential"]},
        "lr_gamma": {"values": [0.9201643819406394]},
        "weight_decay": {"values": [0.000158927457599188366]},
        "num_workers": {"values": [20]},  # num workers
        "normalise_pixels": {"values": ["standardise"]},
        "resize": {"values": [(224, 224)]},
        "data_augmentation": {"values": [True]},
        "transform": {"values": [["random_crop", "rotate", "brightness", "contrast"]]},
        "crop_prob": {"values": [0.5]},
        "transform_prob": {"values": [0.2]},
        "k_imgs": {"values": [10]},
        "single_channel": {"values": [True]},  # single channel input
        "extract_lung_parenchyma": {"values": [False]},
        "crop_margin": {"values": [False]},
        "ct_required": {"values": [True]},
        "remove_suspected": {"values": [True]},
    },
}


multimodal_fusion_sweep_config = {
    "name": "exp4_multimodal-pre_bayes",
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        # General parameters
        "device": {"values": ["cuda"]},
        "seed": {"values": [42]},
        "model": {"values": ["multimodal_fusion"]},
        "load_model": {"values": [False]},
        "optimizer": {"values": ["adam"]},
        "num_epochs": {"values": [100]},
        "batch_size": {"values": [64]},
        # "learning_rate": {"values": [0.0001]},
        # "weight_decay": {"values": [0.0001]},
        "learning_rate": {
            "distribution": "uniform",
            "min": 0.00001,
            "max": 0.001,
        },
        "weight_decay": {
            "distribution": "uniform",
            "min": 0.0001,
            "max": 0.01,
        },
        "lr_scheduler": {"values": ["exponential"]},
        "lr_gamma": {
            "distribution": "uniform",
            "min": 0.85,
            "max": 0.99,
        },
        "num_workers": {"values": [20]},  # num workers
        # CT Resnet parameters
        "normalise_pixels": {"values": ["standardise"]},
        "resize": {"values": [(224, 224)]},
        "data_augmentation": {"values": [True]},
        "transform": {"values": [["random_crop", "rotate", "brightness", "contrast"]]},
        "crop_prob": {"values": [0.5]},
        "transform_prob": {"values": [0.2]},
        "k_imgs": {"values": [10]},
        "single_channel": {"values": [True]},  # single channel input
        "ct_required": {"values": [True]},
        "pretrained": {"values": [True]},
        # CF DNN parameters
        "dropout": {"values": [0.5]},  # dropout, also for multimodal head (late_pre_classifier)
        "undis_binary": {"values": [False]},
        "remove_suspected": {"values": [True]},
        "batch_norm": {"values": [True]},  # batch norm?
        "normalisation": {"values": ["standardise"]},
        "nan_strategy": {"values": [-1]},
        "max_iter": {"values": [20]},
        "init_strategy": {"values": ["mean"]},
        "n_neighbours": {"values": [None]},
        # Multimodal fusion parameters
        "fusion_type": {"values": ["late_pre_classifier"]},  # fusion type
        "freeze_layers": {"values": [False]},  # freeze layers after loading weights
        "pretrained_models": {"values": [False]},  # load pretrained models
        "cf_dnn_path": {
            "values": [
                (
                    "research/case_study/biomed/models/CF_DNN/f1_cf_dnn_apricot-sweep-24.pth",
                    "radadjoneva-icl/covid-outcome-classification/2sspqbzv",
                )
            ]
        },
        "ct_resnet_path": {
            "values": [
                (
                    "research/case_study/biomed/models/CT_CNN/f1_resnet50_worldly-sweep-4.pth",
                    "radadjoneva-icl/covid-outcome-classification/c3hpiq15",
                )
            ]
        },
    },
}


if __name__ == "__main__":
    # Initialise the sweep
    # sweep_type = cf_dnn_sweep_config
    # sweep_id = wandb.sweep(sweep=sweep_type, project="covid-outcome-classification")

    sweep_type = multimodal_fusion_sweep_config
    sweep_id = wandb.sweep(sweep=sweep_type, project="covid-outcome-multimodal-classification")

    # Run the sweep agent
    wandb.agent(sweep_id, function=train)

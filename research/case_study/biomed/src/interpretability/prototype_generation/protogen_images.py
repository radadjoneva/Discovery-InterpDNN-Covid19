# ruff: noqa: E402
# ruff: noqa: I001

# Leap Interpretability Engine: Prototype Generation for image classifiers
import os

import sys
import yaml

import torch
# from leap_ie.vision import engine

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "UIE"))

from src.utils.model_utils import load_pretrained_model
from src.preprocessing.ct_preprocess import load_and_preprocess_ct_image
from leap_ie.vision.engine import generate


def run_protogen_img():
    leap_api_key = os.getenv("LEAP_API_KEY")
    if not leap_api_key:
        raise ValueError("API Key not found. Please set the environment variable.")
    
    protogen_config = {
        "leap_api_key": leap_api_key,
        "run_name": "worldly-sweep-4",
        "run_id": "radadjoneva-icl/covid-outcome-classification/2sspqbzv",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "target_classes": [0, 1, 2],
        "input_dim": (1, 1, 512, 512),
        "verbose": 3,
        "max_steps": 500,
        "num_prototypes": 1,
        "lr": 0.001,
        "diversity_weight": 0,
        "use_blur": True,
        "hf_weight": 0,
        "blur_max": 1,
        "transform": "shift_scale",
        # "transform": "s",
        "objective": "logit_objective",
        "use_baseline": True,
    }

    # Load the trained model
    run_name = protogen_config["run_name"]
    # run_id = protogen_config["run_id"]
    model_dir = f"research/case_study/biomed/models/CT_CNN/{run_name}"
    model_path = os.path.join(model_dir, f"f1_resnet50_{run_name}.pth")
    # eval_random_states_path = os.path.join(model_dir, f"random_states_{run_name}-best-epoch.pth")
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    config.update(protogen_config)

    # Load the model for evaluation
    ct_resnet = load_pretrained_model(model_path, config, skorch=False)
    # Set model to evaluation mode (just in case)
    ct_resnet.to(config["device"])
    ct_resnet.eval()

    # Process baseline input image (negative Covid-19) if use_baseline is True
    img_dir = "research/case_study/biomed/datasets/iCTCF/single_images"
    img_path = "research/case_study/biomed/datasets/iCTCF/single_images/nCT/nCT21.jpg"  # No Covid-19
    # img_path = "research/case_study/biomed/datasets/iCTCF/single_images/pCT/pCT8.jpg"  # Positive Covid-19
    img_config = {
        "extract_lung_parenchyma": False,
        "crop_margin": False,
        "resize": (512, 512),
        "normalise_pixels": "standardise",
    }
    baseline_image = load_and_preprocess_ct_image(
                img_path, img_config, img_dir, is_train=False
            )
    baseline_image = baseline_image.unsqueeze(0)  # Add batch dimension

    results_df, results_dict = generate(
        project_name="CT_prototypes_baseline",
        model=ct_resnet,
        class_list=config["covid_outcome_classes"],
        config=config,
        target_classes=config["target_classes"],
        base_input = baseline_image
    )

    # results_df, results_dict = generate_img_prototypes(
    #     leap_api_key=leap_api_key,
    #     model=ct_resnet,
    #     config=config,
    # )

    # results_df, results_dict = generate_img_prototypes(
    #     leap_api_key=leap_api_key,
    #     model=ct_resnet,
    #     input_dim=(1, 1, 224, 2240),
    #     class_list=class_list,
    #     target_classes=target_classes,
    #     learning_rate=0.003,
    #     max_steps=250,
    #     use_blur=True,
    #     hf_weight=0,
    #     blur_max=0.1,
    #     transform="s",
    #     objective="logit_objective",
    #     diversity_weight=1,
    # )


if __name__ == "__main__":
    run_protogen_img()

# config = {
#     "leap_api_key": key,
#     "input_dim": (1, 10, 224, 224),
#     "verbose": 3,
#     "lr": 0.01,  # default 0.01
#     "max_steps": 1000,  # default 1000
#     # "find_lr_steps": 1000,  # 0
#     # "find_lr_windows": 25,  # 30
#     # "max_lr": 2.0,  # 2.0
#     # "min_lr": 0.001,
#     # "transform": "s",
#     # "use_blur": False,
#     # "hf_weight": 10,  # 0.0
#     # "diversity_weight": 3
# }

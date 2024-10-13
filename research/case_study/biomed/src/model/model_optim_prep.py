# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys
import torch
import torch.nn as nn
import wandb

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.model.cf_dnn import DNNSimple
from src.model.vgg_simple import VGGSimple
from src.model.vgg import VGG16Adapted, VGG19Adapted
from src.model.resnet import ResNet50Adapted
from src.model.multimodal_fusion import CTCFMultimodalFusion

from src.utils.model_utils import load_pretrained_model


def prepare_model_and_optimizer(config, input_dim=None):
    # Parameters
    num_classes = len(config["covid_outcome_classes"])
    num_channels = config["k_imgs"]
    dropout = config["dropout"]
    resize = config["resize"]
    pretrained = config["pretrained"]
    batch_norm = config["batch_norm"]
    # Pretrained models to use for multimodal fusion
    cf_dnn_path = config["cf_dnn_path"] if "cf_dnn_path" in config else None
    ct_resnet_path = config["ct_resnet_path"] if "ct_resnet_path" in config else None
    freeze_layers = config["freeze_layers"] if "freeze_layers" in config else False
    # use pretrained models for multimodal fusion
    pretrained_models = config["pretrained_models"] if "pretrained_models" in config else False

    if config["model"] in ["ct_images_cnn", "ct_patient_cnn"]:
        if config["model"] == "ct_images_cnn":
            model = VGGSimple(dropout=dropout)
        elif config["model"] == "ct_patient_cnn":
            input_dim = tuple(resize) + (num_channels,)
            model = VGGSimple(input_dim=input_dim, dropout=dropout)
    elif config["model"] == "cf_dnn":
        model = DNNSimple(input_dim, dropout=dropout, batch_norm=batch_norm)
    elif config["model"] == "vgg16":
        model = VGG16Adapted(num_channels, num_classes, dropout, pretrained)
    elif config["model"] == "vgg19":
        model = VGG19Adapted(num_channels, num_classes, dropout, pretrained)
    elif config["model"] == "resnet50":
        if config["single_channel"]:
            num_channels = 1
        model = ResNet50Adapted(num_channels, num_classes, pretrained)
    elif config["model"] == "multimodal_fusion":
        if config["single_channel"]:
            num_channels = 1
        if pretrained_models:
            # Get config files from run_id
            cf_dnn_run_id = cf_dnn_path[1]
            api = wandb.Api()
            cf_dnn_run = api.run(cf_dnn_run_id)
            cf_dnn_config = cf_dnn_run.config
            ct_resnet_run_id = ct_resnet_path[1]
            ct_resnet_run = api.run(ct_resnet_run_id)
            ct_resnet_config = ct_resnet_run.config
            # Load pretrained models for multimodal fusion
            cf_dnn = load_pretrained_model(
                cf_dnn_path[0],
                cf_dnn_config,
                input_dim=input_dim,
            )
            ct_resnet = load_pretrained_model(
                ct_resnet_path[0],
                ct_resnet_config,
            )
        else:
            cf_dnn = DNNSimple(input_dim, dropout=dropout, batch_norm=batch_norm)
            ct_resnet = ResNet50Adapted(num_channels, num_classes, pretrained)
        # Instantiate the multimodal fusion model
        model = CTCFMultimodalFusion(
            cf_dnn,
            ct_resnet,
            fusion_type=config["fusion_type"],
            num_classes=num_classes,
            dropout=dropout,
            freeze_layers=freeze_layers,
        )

    # Weighted CrossEntropyLoss for imbalanced datasets
    if config["class_weights"]:
        # convert to tensor
        class_weights = torch.tensor(config["class_weights"], dtype=torch.float32).to(
            config["device"]
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )

    return model, criterion, optimizer

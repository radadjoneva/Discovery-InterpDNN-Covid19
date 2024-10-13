# ruff: noqa: E402
# ruff: noqa: I001

"""Contains Model class to define, train, fine-tune and evaluate the model.

Model loading reference: RETFound_MAE
"""

import os
import sys

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from itertools import chain
import torch

# from timm.data import create_transform
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.config import load_config
from src.model.vit_retfound import vit_large_patch16
from src.utils.model_utils import (
    get_custom_head,
    initialise_linear_layers,
    validate_input,
)
from src.utils.pos_embed import interpolate_pos_embed


class Model:
    def __init__(self, config):
        self.config = config
        self.model_name = config["model"]
        self.args = config["model_args"]
        self.device = config["device"]

        if config["google_drive"]["use"]:
            self.model_path = os.path.join(config["google_drive"]["models_path"], self.model_name)
        else:
            self.model_path = os.path.join(config["models_local_path"], self.model_name)

        self.model = None

    def define_model(self):
        # Implement model definition logic
        if self.config["model_type"] == "vit_large_patch16":
            self.model = vit_large_patch16(
                img_size=self.args["input_size"],
                num_classes=self.args["nb_classes"],
                drop_path_rate=self.args["drop_path"],
                global_pool=self.args["global_pool"],
            )
            # ADD custom head layers if specified
            if self.config["custom_head"] is not None:
                self.model.head = get_custom_head(self.config["custom_head"])
                print("\nCustom head added to the model!")
            print("\nVit large patch 16 model defined!\n")
        else:
            print("Model type not found!")

    def load_model(self):
        # Define model if not already defined
        if self.model is None:
            self.define_model()

        # Load pre-trained model weights
        checkpoint = torch.load(self.model_path, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        state_dict = self.model.state_dict()

        # Remove incompatible keys if necessary
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(self.model, checkpoint_model)

        # load state dict
        msg = self.model.load_state_dict(checkpoint_model, strict=False)

        print(f"Missing keys: {msg.missing_keys}\n")

        if self.args["finetune"]:
            initialise_linear_layers(self.model.head, std=2e-5)

        self.model.to(self.device)
        print(f"\nModel {self.model_name} loaded!\n")

    # def train_model(self, data):
    #     # Implement model training logic??
    #     pass

    def finetune_model(self, data):
        # Implement model fine-tuning logic
        if not self.args["finetune"]:
            raise ValueError("Model is not set to be fine-tuned")

    def evaluate_model(self, data):
        # Implement model evaluation logic
        pass

    def predict(self, input_data):
        # Validate input
        expected_shape = (
            input_data.size(0),
            self.args["in_chans"],
            self.args["input_size"],
            self.args["input_size"],
        )
        validate_input(input_data, expected_shape)
        input_data = input_data.to(self.device)

        # Perform inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)
            predicted_classes = torch.argmax(output, dim=1)

        return output, predicted_classes


if __name__ == "__main__":
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "config/config_retinal.yaml")
    )
    config = load_config(config_path)
    model = Model(config)

    # Load pre-trained model weights
    model.load_model()
    # print(model.model)

    # Dummy input data
    dummy_input = torch.randn(1, 3, 224, 224)
    output, predicted_classes = model.predict(dummy_input)
    print(f"Output: {output}")
    print(f"Predicted classes: {predicted_classes}")

    # Finetune model (example)
    if config["model_args"]["finetune"]:
        model.finetune_model(data=None)

    # Evaluate model

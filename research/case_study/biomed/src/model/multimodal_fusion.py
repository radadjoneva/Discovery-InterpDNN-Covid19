# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.model.cf_dnn import DNNSimple
from src.model.resnet import ResNet50Adapted


class CTCFMultimodalFusion(nn.Module):
    def __init__(
        self,
        cf_dnn,
        ct_resnet,
        fusion_type="late_post_classifier",
        num_classes=3,
        dropout=0.5,
        freeze_layers=False,
    ):
        super(CTCFMultimodalFusion, self).__init__()
        self.cf_dnn = cf_dnn
        self.ct_resnet = ct_resnet
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        self.dropout = dropout
        self.freeze_layers = freeze_layers

        # Freeze layers if specified
        if self.freeze_layers:
            self.ct_resnet.eval()
            self.cf_dnn.eval()
            for param in self.cf_dnn.parameters():
                param.requires_grad = False
            for param in self.ct_resnet.parameters():
                param.requires_grad

        if fusion_type == "late_post_classifier":
            # Both models output logits with the same number of classes
            self.classifier = nn.Linear(num_classes * 2, num_classes)
        elif fusion_type == "late_pre_classifier":
            # Get feature dimensions from each model (before classifier)
            feature_dim_dnn = cf_dnn.fc4.out_features  # fc4 features (pre-penultimate layer) (48)
            feature_dim_resnet = ct_resnet.resnet50.fc.in_features  # after avgpool (2048)
            combined_feature_dim = feature_dim_dnn + feature_dim_resnet  # (2096)

            self.fc_multi = nn.Linear(combined_feature_dim, 1024)
            self.dropout = nn.Dropout(dropout) if dropout else None
            self.classifier = nn.Linear(1024, num_classes)
        else:
            raise ValueError("Invalid fusion type")

    def forward(self, cf_ct):
        cf, ct = cf_ct
        if self.fusion_type == "late_post_classifier":
            # Concatenate the logits from both models
            out1 = self.cf_dnn(cf)
            out2 = self.ct_resnet(ct)
            combined = torch.cat((out1, out2), dim=1)
            # Pass through the multimodal classifier
            out = self.classifier(combined)
        elif self.fusion_type == "late_pre_classifier":
            # Extract fc4 features from cf_dnn
            features_dnn = self.cf_dnn.get_features(cf, layer=4)
            # Extract penultimate layer features from resnet (after avg pool)
            features_resnet = self.ct_resnet.get_features(ct)
            # Concatenate the features from both models and pass through the classifier
            combined = torch.cat((features_dnn, features_resnet), dim=1)
            x = F.relu(self.fc_multi(combined))
            x = self.dropout(x) if self.dropout else x
            out = self.classifier(x)
        else:
            raise ValueError("Invalid fusion type")
        return out

    def train(self, mode=True):
        # Override train to prevent frozen models from being set to train mode
        super(CTCFMultimodalFusion, self).train(mode)
        if self.freeze_layers:
            self.cf_dnn.eval()
            self.ct_resnet.eval()


if __name__ == "__main__":
    # Number of columns in clinical features input data
    cf_df = pd.read_csv(
        "research/case_study/biomed/datasets/iCTCF/processed_cf_data_train_standardise.csv"
    )
    drop_columns = [
        "Patient ID",
        "Hospital",
        "SARS-CoV-2 nucleic acids",
        "Computed tomography (CT)",
        "Morbidity outcome",
        "Mortality outcome",
    ]
    drop_cols = [
        col for col in list(cf_df.columns) for d_col in drop_columns if d_col in col
    ]  # columns to drop after one hot encoding
    input_dim = len(cf_df.columns) - len(drop_cols)

    # configs
    pretrained = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the individual models
    ct_resnet = ResNet50Adapted(num_channels=1, num_classes=3, pretrained=False)
    cf_dnn = DNNSimple(input_dim=input_dim, dropout=0.5, batch_norm=True)

    # Load models
    cf_dnn_path = "research/case_study/biomed/models/CF_DNN/auc_cf_dnn_vibrant-sweep-41.pth"
    ct_resnet_path = (
        "research/case_study/biomed/models/CT_CNN/best_loss_resnet50_desert-sweep-14.pth"
    )
    if pretrained:
        state_dict_cf = torch.load(cf_dnn_path, map_location="cpu")
        state_dict_ct = torch.load(ct_resnet_path, map_location="cpu")
        cf_dnn.load_state_dict(state_dict_cf)
        ct_resnet.load_state_dict(state_dict_ct)

        ct_resnet.to(device)
        cf_dnn.to(device)
        ct_resnet.eval()
        cf_dnn.eval()

    # Multimodal fusion model with late fusion
    # fusion_model = CTCFMultimodalFusion(cf_dnn, ct_resnet, fusion_type="late_post_classifier", num_classes=3)
    fusion_model = CTCFMultimodalFusion(
        cf_dnn, ct_resnet, fusion_type="late_pre_classifier", num_classes=3
    )

    # Example input data for both models
    ct_scans = torch.randn(8, 1, 224, 2240)  # Example batch of CT scans
    clinical_features = torch.randn(8, input_dim)  # Example batch of clinical features

    # Forward pass for late fusion
    output_late = fusion_model(clinical_features, ct_scans)

    # # Forward pass for intermediate fusion
    # output_intermediate = fusion_model_intermediate(ct_scans, clinical_features)

    print("Done!")

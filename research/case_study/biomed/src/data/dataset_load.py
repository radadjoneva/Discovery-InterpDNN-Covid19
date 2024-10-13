# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys
import pandas as pd

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.data.cf_dataset import CFDataset
from src.data.ct_dataset import CovidCTDataset, CovidOutcomeDataset
from src.data.multimodal_dataset import CTCFMultimodalDataset


def load_datasets(config):
    """Load the appropriate dataset based on the model specified in the configuration."""
    # Determine the dataset based on the model
    single_img_data = config["model"] == "ct_images_cnn"
    patient_ct_data = config["model"] in [
        "ct_patient_cnn",
        "vgg16",
        "vgg19",
        "resnet50",
        "inceptionv3",
    ]
    cf_data = config["model"] == "cf_dnn"
    multimodal_data = config["model"] == "multimodal_fusion"
    split_ratio = config.get("split_ratio", [0.7, 0.2, 0.1])

    if single_img_data:
        # CovidCT dataset (single images)
        train_dataset = CovidCTDataset(config, split="train")
        val_dataset = CovidCTDataset(config, split="val")
        test_dataset = CovidCTDataset(config, split="test")

    elif patient_ct_data:
        # Get COVID outcomes from clinical features dataset
        cf_df = pd.read_csv(config["cleaned_cf_data"])
        patient_id_outcome_df = cf_df[["Patient ID", config["outcome"]]]
        # CovidCT dataset (patient-level images)
        train_dataset = CovidOutcomeDataset(config, patient_id_outcome_df, split="train")
        val_dataset = CovidOutcomeDataset(config, patient_id_outcome_df, split="val")
        test_dataset = CovidOutcomeDataset(config, patient_id_outcome_df, split="test")

    elif cf_data:
        # CF dataset (clinical features)
        cf_df = pd.read_csv(config["cleaned_cf_data"])
        all_patient_ids = cf_df["Patient ID"]

        # Get patient_ids train/val/test split from CT dataset
        patient_id_outcome_df = cf_df[["Patient ID", config["outcome"]]]
        ct_dataset_train = CovidOutcomeDataset(config, patient_id_outcome_df, split="train")
        patient_ids_train = list(ct_dataset_train.patient_df["Patient ID"])
        ct_dataset_val = CovidOutcomeDataset(config, patient_id_outcome_df, split="val")
        patient_ids_val = list(ct_dataset_val.patient_df["Patient ID"])
        ct_dataset_test = CovidOutcomeDataset(config, patient_id_outcome_df, split="test")
        patient_ids_test = list(ct_dataset_test.patient_df["Patient ID"])

        # Patient IDs for train, val, and test sets (add remaining ids to CT dataset patient_ids)
        remaining_ids = [id for id in all_patient_ids if id not in patient_ids_train+patient_ids_val+patient_ids_test]
        n_train = int(split_ratio[0] * len(remaining_ids))
        n_val = int(split_ratio[1] * len(remaining_ids))
        train_ids = patient_ids_train + remaining_ids[:n_train]
        val_ids = patient_ids_val + remaining_ids[n_train : n_train + n_val]
        test_ids = patient_ids_test + remaining_ids[n_train + n_val :]

        # assert no overlap between train, val, and test set ids
        assert len(set(train_ids).intersection(val_ids)) == 0
        assert len(set(train_ids).intersection(test_ids)) == 0
        assert len(set(val_ids).intersection(test_ids)) == 0

        train_dataset = CFDataset(cf_df, config, split="train", patient_ids=train_ids, preprocessor=None)
        val_dataset = CFDataset(
            cf_df, config, split="val", patient_ids=val_ids, preprocessor=train_dataset.get_preprocessor()
        )
        test_dataset = CFDataset(
            cf_df, config, split="test", patient_ids=test_ids, preprocessor=train_dataset.get_preprocessor()
        )
    elif multimodal_data:
        # Load CT and CF datasets and match patient IDs
        # Load clinical features dataset
        cf_df = pd.read_csv(config["cleaned_cf_data"])
        cf_df = cf_df.dropna(subset=["Computed tomography (CT)"])
        patient_id_outcome_df = cf_df[["Patient ID", config["outcome"]]]

        # Train split
        ct_dataset_train = CovidOutcomeDataset(config, patient_id_outcome_df, split="train")
        patient_ids_train = list(ct_dataset_train.patient_df["Patient ID"])
        cf_dataset_train = CFDataset(
            cf_df, config, split="train", patient_ids=patient_ids_train, preprocessor=None
        )
        # Creat multimodla train dataset
        train_dataset = CTCFMultimodalDataset(ct_dataset_train, cf_dataset_train, split="train")

        # Ensure preprocessor is fitted with training data
        preprocessor = cf_dataset_train.get_preprocessor()

        # Validation split
        ct_dataset_val = CovidOutcomeDataset(config, patient_id_outcome_df, split="val")
        patient_ids_val = list(ct_dataset_val.patient_df["Patient ID"])
        cf_dataset_val = CFDataset(
            cf_df, config, split="val", patient_ids=patient_ids_val, preprocessor=preprocessor
        )
        val_dataset = CTCFMultimodalDataset(ct_dataset_val, cf_dataset_val, split="val")

        # Test split
        ct_dataset_test = CovidOutcomeDataset(config, patient_id_outcome_df, split="test")
        patient_ids_test = list(ct_dataset_test.patient_df["Patient ID"])
        cf_dataset_test = CFDataset(
            cf_df, config, split="test", patient_ids=patient_ids_test, preprocessor=preprocessor
        )
        test_dataset = CTCFMultimodalDataset(ct_dataset_test, cf_dataset_test, split="test")

    return train_dataset, val_dataset, test_dataset

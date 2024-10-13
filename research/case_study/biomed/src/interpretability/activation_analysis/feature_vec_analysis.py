# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys
import torch
import pandas as pd
import yaml


# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

from src.model.activations_recorder import ActivationsRecorder
from src.utils.model_utils import load_pretrained_model
from src.data.prep_data_loaders import load_datasets_and_initialize_loaders



if __name__ == "__main__":
    # Load the trained model
    run_name = "apricot-sweep-24"
    run_id = "radadjoneva-icl/covid-outcome-classification/2sspqbzv"  # apricot-sweep-24
    model_dir = f"research/case_study/biomed/models/CF_DNN/{run_name}"
    model_path = os.path.join(model_dir, f"f1_cf_dnn_{run_name}.pth")
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # api = wandb.Api()
    # run = api.run(run_id)
    # config = run.config

    # Load the datasets (preprocessed)
    data_dir = f"research/case_study/biomed/datasets/iCTCF/processed_cf/{run_name}"
    X_train = pd.read_csv(os.path.join(data_dir, "input_features_train.csv"))
    Y_train = pd.read_csv(os.path.join(data_dir, "target_outcomes_train.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "input_features_val.csv"))
    Y_val = pd.read_csv(os.path.join(data_dir, "target_outcomes_val.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "input_features_test.csv"))
    Y_test = pd.read_csv(os.path.join(data_dir, "target_outcomes_test.csv"))
    # Load datasets (preprocessed & inverse standardised - real range values)
    X_train_real = pd.read_csv(os.path.join(data_dir, "input_features_train_inverse.csv"))
    X_val_real = pd.read_csv(os.path.join(data_dir, "input_features_val_inverse.csv"))
    X_test_real = pd.read_csv(os.path.join(data_dir, "input_features_test_inverse.csv"))

    # Combine the datasets
    X_all = pd.concat([X_train, X_val, X_test])
    Y_all = pd.concat([Y_train, Y_val, Y_test])
    X_all_real = pd.concat([X_train_real, X_val_real, X_test_real])

    # Get patient ids
    patient_ids_train = pd.read_csv(os.path.join(data_dir, "patient_ids_train.csv"))
    patient_ids_val = pd.read_csv(os.path.join(data_dir, "patient_ids_val.csv"))
    patient_ids_test = pd.read_csv(os.path.join(data_dir, "patient_ids_test.csv"))
    patient_ids_all = pd.concat([patient_ids_train, patient_ids_val, patient_ids_test])

    # Create the "Split" column
    split_train = ["train"] * len(X_train)
    split_val = ["val"] * len(X_val)
    split_test = ["test"] * len(X_test)
    split_all = split_train + split_val + split_test

    # Combine into a DataFrame
    idx_df = pd.DataFrame({
        "Patient ID": patient_ids_all["Patient ID"].values,
        "Split": split_all,
        "Outcome": Y_all.values.argmax(axis=1)  # encoded: 0 - Control, 1 - Type I, 2 - Type II
    })

    # Parameters 
    class_list = config["covid_outcome_classes"]
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = len(X_train.columns)

    # Load the model for evaluation
    cf_dnn = load_pretrained_model(model_path, config, input_dim=input_dim, skorch=False)
    # Set model to evaluation mode (just in case)
    cf_dnn.eval()

    # Instantiate the recorder
    cf_dnn_recorder = ActivationsRecorder(cf_dnn)
    # Register hooks
    cf_dnn.register_hooks(cf_dnn_recorder)

    # Record activations for the dataset
    cf_dnn_recorder.record_activations(X_data=X_all, idx_df=idx_df)

    # Access activations for a specific patient
    patient_activations = cf_dnn_recorder.activations["some_patient_id"]

    # Saving activations
    # import pickle
    # with open('activations.pkl', 'wb') as f:
    #     pickle.dump(cf_dnn_recorder.activations, f)

    print("Done!")


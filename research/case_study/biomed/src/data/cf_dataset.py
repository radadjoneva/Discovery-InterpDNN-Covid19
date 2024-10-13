# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys

import pandas as pd
import torch
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, MultiLabelBinarizer
from missforest.missforest import MissForest
from torch.utils.data import Dataset
import wandb

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# from data.base_dataset.py import Dataset
from src.utils.data_utils import split_data
from src.preprocessing.cf_preprocess import get_save_path, merge_outcomes_and_filter_rows


class CFDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.drop_columns = config["drop_columns"]
        self.normalisation = config[
            "normalisation"
        ]  # Normalisation strategy for continuous columns. ('standardise' or 'minmax' or 'minmax_normal_range')
        self.nan_strategy = config[
            "nan_strategy"
        ]  # Numerical columns "mean" or "median" or "iterative_imputer" or 0.5 or "normal_range" (?)
        self.max_iter = config["max_iter"]  # for iterative imputer, max number of iterations
        self.missing_indicator = config.get("missing_indicator", False)  # missingness indicator
        self.missingness_threshold = config.get(
            "missingness_threshold", 0.5
        )  # NaN threshold above which a missingness indicator is added
        self.undis_binary = config[
            "undis_binary"
        ]  # whether to encode underlying diseases as binary

        # Select imputer for handling NaN values in numerical columns
        if self.nan_strategy == "mean":
            self.num_imputer = SimpleImputer(strategy="mean")
        elif self.nan_strategy == "median":
            self.num_imputer = SimpleImputer(strategy="median")

        elif self.nan_strategy == "iterative_imputer":
            if config["estimator"] == "bayesian_ridge":
                self.estimator = BayesianRidge()
            elif config["estimator"] == "rand_forest":
                self.estimator = RandomForestRegressor(n_estimators=10, random_state=42)
            elif config["estimator"] == "k_neighbour":
                self.estimator = KNeighborsRegressor()
            self.num_imputer = IterativeImputer(
                estimator=self.estimator,
                max_iter=self.max_iter,
                random_state=42,
                tol=1e-3,
                verbose=2,
                initial_strategy=config["init_strategy"],
                n_nearest_features=config["n_neighbours"],
                min_value=0,
                add_indicator=False,
            )  # random_state for reproducibility (seed)

        elif self.nan_strategy == "knn_imputer":
            self.num_imputer = KNNImputer(n_neighbors=config["n_neighbours"], weights="uniform")
        elif self.nan_strategy == "miss_forest":
            self.num_imputer = MissForest()

        elif isinstance(self.nan_strategy, (int, float)):
            self.num_imputer_cst = SimpleImputer(
                strategy="constant", fill_value=self.nan_strategy
            )  # for columns >= 50% missing
            self.estimator = RandomForestRegressor(
                n_estimators=10, random_state=42
            )  # for columns < 50% missing
            self.num_imputer_iter = IterativeImputer(
                estimator=self.estimator,
                max_iter=self.max_iter,
                random_state=42,
                tol=1e-3,
                verbose=2,
                initial_strategy=config["init_strategy"],
                n_nearest_features=config["n_neighbours"],
                min_value=0,
                add_indicator=False,
            )  # random_state for reproducibility (seed)
        else:
            raise ValueError(
                "NaN strategy not recognised. Please choose 'mean', 'median', or 'iterative_imputer'."
            )

    def fit(self, X, y=None):
        # Separate numerical and categorical columns, excluding 'Patient ID'
        self.num_cols = [
            col
            for col in X.select_dtypes(include=["float64", "int64"]).columns
            if col != "Patient ID"
        ]
        self.cat_cols = [
            col for col in X.columns if col not in self.num_cols and col != "Patient ID"
        ]
        if not self.undis_binary:
            self.cat_undis = "Underlying diseases"
            self.cat_cols = [col for col in self.cat_cols if col != self.cat_undis]

        # Remove drop columns from the consideration for iterative imputer or KNN
        self.iterative_cols = [col for col in self.num_cols if col not in self.drop_columns]
        self.cols_nan_more50 = [col for col in self.iterative_cols if X[col].isna().mean() >= 0.5]
        self.cols_nan_less50 = [col for col in self.iterative_cols if X[col].isna().mean() < 0.5]

        if isinstance(self.nan_strategy, (int, float)):
            self.num_imputer_cst.fit(X[self.cols_nan_more50])
            self.num_imputer_iter.fit(X[self.cols_nan_less50])
        else:
            # Fit the imputers on the training data
            self.num_imputer.fit(X)

        # Normalisation for numerical columns
        if self.normalisation == "standardise":
            self.num_transformer = StandardScaler()
        elif self.normalisation == "minmax":
            self.num_transformer = MinMaxScaler(feature_range=(0, 1))
        else:
            raise ValueError(
                "Normalisation strategy not recognised. Please choose 'standardise' or 'minmax'."
            )

        # One hot encoding for categorical columns
        self.cat_transformer = OneHotEncoder(handle_unknown="ignore")
        if not self.undis_binary:
            self.cat_undis_transformer = MultiLabelBinarizer()

        # Combine transformers for categorical and numerical columns
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", self.num_transformer, self.num_cols),
                ("cat", self.cat_transformer, self.cat_cols),
            ]
        )

        # Fit the transformer/preprocessor
        self.transformer.fit(X)

        # Fit the 'MultiLabelBinarizer' for underlying diseases
        if not self.undis_binary:
            self.cat_undis_transformer.fit(X["Underlying diseases"])

        return self

    def transform(self, X):
        # Apply imputation
        if isinstance(self.nan_strategy, (int, float)):
            X[self.cols_nan_more50] = self.num_imputer_cst.transform(X[self.cols_nan_more50])
            X[self.cols_nan_less50] = self.num_imputer_iter.transform(X[self.cols_nan_less50])
        else:
            X[self.num_cols] = self.num_imputer.transform(X[self.num_cols])

        # Apply the learned transformations to the data
        X_transformed = self.transformer.transform(X)

        if self.undis_binary:
            # Convert transformed data to a DataFrame
            X_transformed_df = pd.DataFrame(X_transformed, columns=self.get_feature_names())
        else:
            diseases_encoded_df = pd.DataFrame(
                self.cat_undis_transformer.transform(list(X[self.cat_undis])),
                columns=self.cat_undis_transformer.classes_,
            ).add_prefix("cat__undis-")
            # Concatenate transformed data with encoded underlying diseases
            X_transformed_df = pd.DataFrame(X_transformed, columns=self.get_feature_names())
            X_transformed_df = pd.concat([X_transformed_df, diseases_encoded_df], axis=1)
        return X_transformed_df

    def get_feature_names(self):
        # Get feature names after transformation
        feature_names = self.transformer.get_feature_names_out()
        return feature_names


class CFDataset(Dataset):
    def __init__(self, df, config, split="train", patient_ids=None, preprocessor=None):
        """
        Initialize the CFDataset with data, configuration, and split type.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset.
            config (dict): Configuration dictionary containing various settings.
            split (str): Type of data split, one of 'train', 'val', or 'test'. Default is 'train'.
            patient_ids (list, optional): List of patient IDs to filter the data. Default is None.
            preprocessor (CFDataPreprocessor, optional): Preprocessor object for data transformation. Default is None.
        """
        self.config = config
        self.split = split
        assert self.split in ["train", "val", "test"]

        # If patient IDs are provided, use them, else use the indices
        if patient_ids is not None:
            # Filter and then sort by the order of patient_ids
            self.original_df = df[df["Patient ID"].isin(patient_ids)]
            self.original_df["Patient ID"] = pd.Categorical(
                self.original_df["Patient ID"], categories=patient_ids, ordered=True
            )
            self.original_df = self.original_df.sort_values("Patient ID").reset_index(drop=True)
        else:
            split_indices = split_data(
                list(range(len(df))), seed=self.config["seed"], ratio=self.config["split_ratio"], split=self.split
            )
            self.original_df = df.iloc[split_indices].reset_index(drop=True)

        # Merge outcomes and filter rows
        self.processed_df = merge_outcomes_and_filter_rows(
            self.original_df.copy(),
            config["remove_suspected"],
            config["undis_binary"],
            config["ct_required"],
        )
        self.processed_df.reset_index(
            drop=True, inplace=True
        )  # reset index to ensure consecutive row numbers

        self.patient_id = self.processed_df["Patient ID"]

        # Fit preprocessors for training split, use preprocessor for validation and test splits
        if preprocessor is None and self.split == "train":
            self.preprocessor = CFDataPreprocessor(config)
            # Fit the preprocessor and transform the training data
            self.data = self.preprocessor.fit_transform(
                self.processed_df.copy()
            )  # calls fit and transform methods (patient ID is not used)
        elif split != "train":
            # Use the provided preprocessor for validation and test splits
            if preprocessor is None:
                raise ValueError("Preprocessor must be provided for validation and test splits.")
            self.preprocessor = preprocessor

            # Preprocess the validation/test data using the already fitted preprocessor
            self.data = self.preprocessor.transform(self.processed_df.copy())
        else:
            raise ValueError("Invalid split type. Expected 'train', 'val', or 'test'.")

        self.data.insert(0, "Patient ID", self.patient_id)  # add patient ID back to the data

        # Save preprocessed data to a CSV file
        save_path = get_save_path(config, self.split)
        self.data.to_csv(save_path, index=False)
        # WandB logging
        data_artifact = wandb.Artifact(f"cf_data_{self.split}_{wandb.run.name}", type="dataset")
        data_artifact.add_file(save_path)
        wandb.run.log_artifact(data_artifact)

        self.feature_names = [col for col in self.data.columns if col != "Patient ID"]
        # self.feature_names = self.preprocessor.get_feature_names()  # df columns after preprocessing
        self.target_columns = [
            f for f in self.feature_names if self.config["outcome"] in f.split("_")
        ]  # target/ outcome columns
        self.input_columns = [
            col for col in self.feature_names if col.split("_")[2] not in config["drop_columns"]
        ]  # input columns for training

        # Sort target_columns based on the order in covid_outcome_classes
        self.classes = [label.split("_")[-1] for label in self.target_columns]
        class_order = {cls: idx for idx, cls in enumerate(self.config["covid_outcome_classes"])}
        self.target_columns.sort(key=lambda x: class_order[x.split("_")[-1]])

        # Split data: X (features), Y (outcome) and convert to tensors
        self.X = torch.tensor(self.data[self.input_columns].values, dtype=torch.float)
        self.Y = torch.tensor(self.data[self.target_columns].values, dtype=torch.float)

        # Save inputs to WandB
        wandb.log({f"input_features_{self.split}": wandb.Table(data=self.X.cpu().numpy(), columns=self.input_columns)})
        wandb.log({f"target_outcomes_{self.split}": wandb.Table(data=self.Y.cpu().numpy(), columns=self.target_columns)})
        wandb.log({f"patient_ids_{self.split}": wandb.Table(data=[[pid] for pid in self.patient_id.values], columns=["Patient ID"])
})      
        # Inverse transform data for analysis
        self.X_df_inverse = self.get_inverse_transform_data()
        wandb.log({f"input_features_{self.split}_inverse": wandb.Table(data=self.X_df_inverse.values, columns=self.input_columns)})

        # Get class names after sorting
        self.classes = [label.split("_")[-1] for label in self.target_columns]
        assert self.classes == self.config["covid_outcome_classes"]  # assert class order

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient_cf = self.X[idx]
        outcome = self.Y[idx]
        patient_id = self.patient_id[idx]

        data = {"input": patient_cf, "label": outcome, "patient_id": patient_id}

        return data

    def get_feature_names(self):
        return self.feature_names

    def get_preprocessor(self):
        return self.preprocessor

    def class_distribution(self):
        class_counts = {cls: 0 for cls in self.target_columns}
        for outcome in self.Y:
            for idx, value in enumerate(outcome):
                if value == 1:
                    class_counts[self.target_columns[idx]] += 1
        total_samples = len(self.Y)
        class_proportions = {cls: count / total_samples for cls, count in class_counts.items()}
        return class_counts, class_proportions, total_samples
    
    def get_inverse_transform_data(self, data=None):
        """Apply inverse transformation to the numerical columns.

        Args:
            data (torch.Tensor or None): Tensor data to apply the inverse transformation to. 
                                        If None, use self.X.

        Returns:
            pd.DataFrame: DataFrame of features after reverting standardization for numerical columns.
        """
        data = data if data is not None else self.X
        # Convert the tensor to a DataFrame
        data_df = pd.DataFrame(data.cpu().numpy(), columns=self.input_columns)

        # Apply inverse transformation to the numerical columns only
        num_transformer = self.preprocessor.transformer.named_transformers_["num"]
        if hasattr(num_transformer, "inverse_transform"):
            data_df_inverse = data_df.copy()
            num_cols = [col for col in data_df.columns if col.startswith('num')]
            # Apply the inverse transform to the numerical columns
            data_df_inverse[num_cols] = num_transformer.inverse_transform(data_df_inverse[num_cols])
        else:
            raise ValueError(f"The numerical transformer {type(num_transformer)} does not support inverse_transform.")

        return data_df_inverse



if __name__ == "__main__":
    # Example usage
    config = {
        "raw_cf_data": "research/case_study/biomed/datasets/iCTCF/cf_data",
        "cleaned_cf_data": "research/case_study/biomed/datasets/iCTCF/cleaned_cf_data.csv",
        "outcome": "Morbidity outcome",
        "drop_columns": [
            "Patient ID",
            "Hospital",
            "SARS-CoV-2 nucleic acids",
            "Computed tomography (CT)",
            "Morbidity outcome",
            "Mortality outcome",
        ],  # drop columns for training
        "ct_required": False,
        "remove_suspected": True,
        "normalisation": "standardise",  # "standardise", "minmax" [0, 1], "minmax_normal_range"
        "undis_binary": True,
        "nan_strategy": "mean",  # numerical columns "mean", "median", "iterative imputer", 0.5, "normal_range" (?)
        "max_iter": 10,  # for iterative imputer, max number of iterations
        "missing_indicator": False,  # missingness indicator works?
        "missingness_threshold": 0.5,  # NaN threshold above which a missingness indicator is added
    }

    cf_df = pd.read_csv(config["cleaned_cf_data"])

    train_dataset = CFDataset(cf_df, config, split="train")
    val_dataset = CFDataset(
        cf_df, config, split="val", preprocessor=train_dataset.get_preprocessor()
    )
    test_dataset = CFDataset(
        cf_df, config, split="test", preprocessor=train_dataset.get_preprocessor()
    )

    # Verify patient IDs don't overlap between train, val, test
    if (
        set(train_dataset.patient_id) & set(val_dataset.patient_id) & set(test_dataset.patient_id)
        == set()
    ):
        print("\nNo overlap between train, val and test patient IDs!")

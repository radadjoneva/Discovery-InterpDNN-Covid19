# ruff: noqa: E402
# ruff: noqa: I001

import os

# Set the environment variable for CuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import wandb

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

from src.interpretability.prototype_generation.protogen_tabular import run_protogen_tabular

protogen_tab_sweep_config = {
    "name": "distribution_prototypes_tabular",
    "method": "grid",
    "parameters": {
        "run_name": {"values": ["apricot-sweep-24"]},
        "run_id": {"values": ["radadjoneva-icl/covid-outcome-classification/2sspqbzv"]},
        "device": {"values": ["cpu"]},
        "epochs": {"values": [500]},
        "target_ix": {"values": [[0], [1], [2]]},
        "num_prototypes": {"values": [1000]},
        "lr": {"values": [0.01, 0.001]},
        "diversity_weight": {"values": [0]},
        "grad_mask": {"values": [None]},
        "type": {"values": ["classification"]},
        "target_state": {"values": ["max"]},
        "min_val": {"values": [None]},
        "max_val": {"values": [None]},
    }
}


if __name__ == "__main__":
    # Initialise the prototype generation sweep
    sweep_type = protogen_tab_sweep_config
    sweep_id = wandb.sweep(sweep=sweep_type, project="protogen-tabular")

    # Run the sweep agent
    wandb.agent(sweep_id, function=run_protogen_tabular)
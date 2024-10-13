# ruff: noqa: E402
# ruff: noqa: I001

import os

# Set the environment variable for CuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import torch
import torch.nn as nn
import yaml
import random
import pandas as pd

import wandb

import torch.optim as optim

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.utils.utils import restore_random_state, set_seed
from src.data.prep_data_loaders import load_datasets_and_initialize_loaders
from src.utils.model_utils import load_pretrained_model
from src.VAE.vae_trainer import VAETrainer
from src.VAE.vae_trainer_images import ImageVAETrainer
from src.VAE.cf_dnn_vae import VariationalEncoder, Decoder
from src.VAE.ct_resnet_vae import ImageVariationalEncoder, ImageDecoder, ResImageDecoder, ResImageDecoder2
from src.VAE.sampling import generate_reconstructed_data, sample_from_decoder


def generate_and_log_synthetic_data(encoder, decoder, classifier, train_dataset, val_dataset, test_dataset, config, column_names, categorical_column_ix, continuous_column_ix):
    # Generate and log synthetic data for train, val, and test splits
    splits = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    for split_name, dataset in splits.items():
        # Deterministic (mu) and non-deterministic reconstructed data (mu + sigma * epsilon)
        for deterministic in [True, False]:
            # Reconstruct real data using the VAE
            reconstructed_data = generate_reconstructed_data(encoder, decoder, classifier, dataset.X.to(config["device"]), deterministic=deterministic)

            # Log reconstructed data to WandB
            wandb.log({
                f"reconstruct_{split_name}_det{deterministic}": wandb.Table(data=reconstructed_data.cpu().numpy(), columns=column_names)
            })
            # Get and log inverse standardised synthetic data
            inv_reconstructed_data = dataset.get_inverse_transform_data(reconstructed_data)
            wandb.log({
                f"inv_reconstruct_{split_name}_det{deterministic}": wandb.Table(data=inv_reconstructed_data.values, columns=column_names)
            })
    
    # Generate and log non-deterministic synthetic data sampled from the decoder
    num_samples = len(train_dataset.X)
    # STD: 1 and 3
    for num_std in [1, 3]:
        sampled_synthetic_data = sample_from_decoder(
            decoder=decoder,
            continuous_column_ix=continuous_column_ix,
            categorical_groups=categorical_column_ix,  # has to be groups! but not used with coerce_probs=False
            latent_dim=config["latent_dim"],
            num_samples=num_samples,
            num_std=num_std,
            device=config["device"],
            act_fn=None,  # Replace with your activation function if needed
            clamp=False
        )

        wandb.log({
            f"sampled_synthetic_std{num_std}": wandb.Table(data=sampled_synthetic_data, columns=column_names)
        })
        sampled_inv_synthetic_data = train_dataset.get_inverse_transform_data(torch.tensor(sampled_synthetic_data))
        wandb.log({
            f"sampled_inv_synthetic_std{num_std}": wandb.Table(data=sampled_inv_synthetic_data.values, columns=column_names)
        })


def inverse_normalize(tensor, mean, std):
    """
    Inverse normalize a tensor using the provided mean and std.
    """
    return tensor * std + mean


def select_random_images_from_classes(dataset, num_images_per_class=10):
    """
    Randomly select num_images_per_class from each class in the dataset.
    Returns a dictionary where keys are class labels and values are lists of selected images.
    """
    class_indices = {label: [] for label in range(len(dataset.classes))}

    # Organize indices by class
    for idx, data in enumerate(dataset):
        class_label = data['label'][0].argmax().item()  # Get the class label from one-hot encoded labels
        class_indices[class_label].append(idx)

    selected_images_per_class = {}

    for label, indices in class_indices.items():
        selected_indices = random.sample(indices, min(len(indices), num_images_per_class))
        
        selected_images = []
        selected_patient_ids = []
        for i in selected_indices:
            # Select a random image for each patient (along the batch dimension)
            img_idx = random.randint(0, dataset[i]['input'].shape[0] - 1)
            selected_images.append(dataset[i]['input'][img_idx])
            selected_patient_ids.append(dataset[i]['patient_id'])
        
        selected_images_per_class[label] = selected_images

        table = wandb.Table(columns=["Patient ID"])
        for patient_id in selected_patient_ids:
            table.add_data(patient_id)
        wandb.log({f"selected_patient_ids_{label}": table})

    return selected_images_per_class



def generate_and_log_synthetic_image_data(encoder, decoder, classifier, datasets, config, mean, std):
    """
    Generate and log synthetic image data for each split.
    """
    splits = {"train": datasets["train"], "val": datasets["val"], "test": datasets["test"]}

    ix_to_class = {0: "Control", 1: "Type I", 2: "Type II"}
    
    for split_name, dataset in splits.items():
        # Select random images from each class
        selected_images_per_class = select_random_images_from_classes(dataset)

        for class_ix, selected_images in selected_images_per_class.items():
            class_label = ix_to_class[class_ix]
            selected_images = torch.stack(selected_images).to(config["device"])

            # Log original images
            inv_images = inverse_normalize(selected_images.to(mean.device), mean, std)
            wandb.log({f"original_{split_name}_{class_label}": [wandb.Image(img, caption=f"Class: {class_label}") for img in inv_images]})

            # Generate and log reconstructed images (deterministic and non-deterministic)
            for deterministic in [True, False]:
                reconstructed_images = generate_reconstructed_data(encoder, decoder, classifier, selected_images, deterministic=deterministic)
                inv_reconstructed_images = inverse_normalize(reconstructed_images.to(mean.device), mean, std)
                wandb.log({
                    f"reconstructed_{split_name}_{class_label}_det{deterministic}": [wandb.Image(img, caption=f"Class: {class_label}") for img in inv_reconstructed_images]
                })
        
        # Generate and log sampled images from the decoder
        num_samples = len(selected_images)
        for num_std in [1, 3]:
            sampled_synthetic_data = sample_from_decoder(
                decoder=decoder,
                continuous_column_ix=None,  # Not used in image generation
                categorical_groups=None,  # Not used in image generation
                latent_dim=config["latent_dim"],
                num_samples=num_samples,
                num_std=num_std,
                device=config["device"],
                act_fn=None,  # Replace with your activation function if needed
                clamp=False
            )
            inv_sampled_synthetic_data = inverse_normalize(torch.tensor(sampled_synthetic_data).to(mean.device), mean, std)
            wandb.log({
                f"sampled_synthetic_{split_name}_std{num_std}": [wandb.Image(img, caption=f"std {num_std}") for img in inv_sampled_synthetic_data]
            })


def train_vae():
    # vae_config = {
    #     "run_name": "apricot-sweep-24",
    #     "run_id": "radadjoneva-icl/covid-outcome-classification/2sspqbzv",
    #     "model_dir": "research/case_study/biomed/models/CF_DNN/apricot-sweep-24",
    #     "device": "cuda" if torch.cuda.is_available() else "cpu",
    #     "save_vae_path": "research/case_study/biomed/results/VAE_training/tabular/",
    #     "latent_dim": 8,
    #     "hidden_dim": 32,
    #     "learning_rate": 1e-3,
    #    "weight_decay": 1e-5,
    #     "num_epochs": 100,
    #     "categorical_coeff": 0.1,
    #     "continuous_coeff": 2,
    #     "kl_coeff": 0.8,
    # }

    vae_config = {
        "run_name": "worldly-sweep-4",
        "run_id": "radadjoneva-icl/covid-outcome-classification/c3hpiq15",
        "model_dir": "research/case_study/biomed/models/CT_CNN/worldly-sweep-4",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_vae_path": "research/case_study/biomed/results/VAE_training/image/",
        "latent_dim": 256,
        "hidden_dim": 1024,
        "output_channels": 1,
        "image_size": 224,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "num_epochs": 200,
        "reconstruction_coeff": 1,
        "recon_loss_type": "l2",  # "l2" or "l1" or "weighted_l2"
        "kl_coeff": 0.01,
        "perceptual_loss": False,
        "percept_weighted": False,
        "perceptual_coeff": 0.1,
        "percept_layers": ["layer1"],
        "k_imgs": 20,  # number of CT scans from the 60% middle ones (single images)
        "class_weights": [10/3, 2, 5],  # if percept_weigths=True or recon_loss_type="weighted_l2"
        "batch_size": 5,  # To prevent OOM errors
        "data_augmentation": False,  # No data augmentation for train dataset
        "fine_tune": False,  # No fine-tuning
        "vae_path": "research/case_study/biomed/models/VAE/image/f1_vae_resnet50_ruby-violet-38.pth",
    }

    # Initialise WandB
    wandb.init(project="covid-vae-image", config=vae_config)
    config = wandb.config

    model_dir = config["model_dir"]
    run_name = config["run_name"]
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as file:
        classifier_config = yaml.safe_load(file)
    model_path = os.path.join(model_dir, f"f1_{classifier_config['model']}_{run_name}.pth")
    init_random_states_path = os.path.join(model_dir, f"random_states_{run_name}.pth")
    eval_random_states_path = os.path.join(model_dir, f"random_states_{run_name}-best-epoch.pth")

    # Update WandB config with classifier config, preserving original WandB config values
    for key, value in classifier_config.items():
        if key not in wandb.config:  # Only update if key is not already in WandB config
            wandb.config[key] = value

    model_filename = os.path.basename(model_path)
    config["model_filename"] = model_filename
    config["init_random_states_path"] = init_random_states_path
    config["eval_random_states_path"] = eval_random_states_path
    target_classes = config["covid_outcome_classes"]

    # Set random seed for reproducibility
    seed = config["seed"]
    set_seed(seed)

    os.makedirs(config["save_vae_path"], exist_ok=True)
    random_states = wandb.Artifact(f"random_states_{model_filename}", type="random_states")
    random_states.add_file(init_random_states_path)
    random_states.add_file(eval_random_states_path)
    wandb.log_artifact(random_states)

    # Load datasets and initialize DataLoaders (also sets random seed)
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_datasets_and_initialize_loaders(config, init_random_states_path)

    # Restore the random state used to evaluate the model/ inference
    if eval_random_states_path:
        eval_random_state = torch.load(eval_random_states_path)
        restore_random_state(eval_random_state)

    # Load the pretrained model
    input_dim = len(train_dataset.input_columns) if config["model"] == "cf_dnn" else None
    classifier = load_pretrained_model(model_path, config, input_dim=input_dim)
    classifier.eval()

    # Initialize VAE components: Encoder, Decoder, and Optimizer
    if config["model"] == "cf_dnn":
        encoder = VariationalEncoder(feature_dim=classifier.fc5.out_features, hidden_dim=config["hidden_dim"], latent_dim=config["latent_dim"])
        decoder = Decoder(num_inputs=input_dim, hidden_dim=config["hidden_dim"], latent_dim=config["latent_dim"])
    elif config["model"] == "resnet50":
        encoder = ImageVariationalEncoder(feature_dim=classifier.resnet50.fc.in_features, hidden_dim=config["hidden_dim"], latent_dim=config["latent_dim"])
        # decoder = ImageDecoder(latent_dim=config["latent_dim"], hidden_dim=config["hidden_dim"], output_channels=config["output_channels"], image_size=config["image_size"])
        # decoder = ResImageDecoder(latent_dim=config["latent_dim"], base_channels=64, output_channels=config["output_channels"])
        decoder = ResImageDecoder2(latent_dim=config["latent_dim"], base_channels=64, output_channels=config["output_channels"])
    else:
        raise ValueError(f"Invalid model: {config['model']}")

    if config["fine_tune"]:
        checkpoint = torch.load(config["vae_path"], map_location=config["device"])
        encoder_state_dict = checkpoint["encoder_state_dict"]
        decoder_state_dict = checkpoint["decoder_state_dict"]
        best_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["val_loss"]
        best_val_f1_diff = checkpoint["val_f1_diff"]
        # Load state dicts into encoder and decoder
        encoder.load_state_dict(encoder_state_dict)
        decoder.load_state_dict(decoder_state_dict)

        print(f"\nFine-tune model: {config['vae_path']}")
        print(f"Epoch: {best_epoch}   |   Validation loss: {best_val_loss}   |  F1 diff: {best_val_f1_diff}")
    
    # Move models to the appropriate device
    classifier.to(config["device"])
    encoder.to(config["device"])
    decoder.to(config["device"])

    # Optimizer
    vae_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Train VAE from classifier representations
    if config["model"] == "cf_dnn":
        # Categorical and continuous feature indices
        column_names = train_dataset.input_columns
        continuous_column_ix = [ix for ix, col in enumerate(column_names) if col.startswith("num")]
        categorical_column_ix = [ix for ix, col in enumerate(column_names) if ix not in continuous_column_ix]

        # Initialise the VAE trainer and train the VAE
        vae_trainer = VAETrainer(
            classifier=classifier,
            encoder=encoder,
            decoder=decoder,
            prep=None,
            train_loader=train_loader,
            val_loader=val_loader,
            optimiser=vae_optimizer,
            categorical_column_ix=categorical_column_ix,
            continuous_column_ix=continuous_column_ix,
            act_fn=None,
            epochs=config["num_epochs"],
            categorical_coeff=config["categorical_coeff"],
            continuous_coeff=config["continuous_coeff"],
            kl_coeff=config["kl_coeff"],
            log_freq=1,
            device=config["device"],
            column_names=column_names,
            target_classes=target_classes,
            save_path=config["save_vae_path"]
        )
        # Train the VAE
        encoder, decoder, mae_df = vae_trainer.train()

    elif config["model"] == "resnet50":
        vae_trainer = ImageVAETrainer(
            classifier=classifier,
            encoder=encoder,
            decoder=decoder,
            prep=None,
            train_loader=train_loader,
            val_loader=val_loader,
            optimiser=vae_optimizer,
            act_fn=None,
            epochs=config["num_epochs"],
            recon_coeff=config["reconstruction_coeff"],
            recon_loss_type=config["recon_loss_type"],
            perceptual_loss=config["perceptual_loss"],
            perceptual_coeff = config["perceptual_coeff"],
            percept_weighted=config["percept_weighted"],
            percept_layers = config["percept_layers"],
            kl_coeff=config["kl_coeff"],
            log_freq=1,
            device=config["device"],
            target_classes=target_classes,
            class_weights=config["class_weights"],
            save_path=config["save_vae_path"]
        )
        # Train the VAE
        encoder, decoder = vae_trainer.train()


    print("VAE training is complete!")

    # Load the best model
    #TODO different for ct cnn and cf dnn
    best_model_path = os.path.join(config["save_vae_path"], f"val_percept_vae_{config['model']}_{wandb.run.name}.pth")
    checkpoint = torch.load(best_model_path)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    best_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["val_loss"]
    # best_val_f1_diff = checkpoint["val_f1_diff"]
    best_val_percept_loss = checkpoint["val_perceptual_loss"]
    print(f"\nBest model: {best_model_path}")
    # print(f"Epoch: {best_epoch}   |   Validation loss: {best_val_loss}   |  F1 diff: {best_val_f1_diff}")
    print(f"Epoch: {best_epoch}   |   Validation loss: {best_val_loss}   |  Perceptual loss: {best_val_percept_loss}")

    if config["model"] == "cf_dnn":
        # Generate and log synthetic data (from real data derived means and sampled from decoder)
        generate_and_log_synthetic_data(
            encoder,
            decoder, 
            classifier, 
            train_dataset, 
            val_dataset, 
            test_dataset, 
            config, 
            column_names, 
            categorical_column_ix, 
            continuous_column_ix
            )
    elif config["model"] == "resnet50":
        # Load your pixel stats
        pixel_stats_path = "research/case_study/biomed/datasets/iCTCF/pixel_stats_224.csv"
        pixel_stats = pd.read_csv(pixel_stats_path)
        # Convert to torch and add to device
        mean = torch.tensor(pixel_stats["mean"].values).view(1, 1, 1).to(config["device"])
        std = torch.tensor(pixel_stats["std"].values).view(1, 1, 1).to(config["device"])

        # Generate and log synthetic image data
        generate_and_log_synthetic_image_data(
            encoder=encoder,
            decoder=decoder, 
            classifier=classifier, 
            datasets={"train": train_dataset, "val": val_dataset, "test": test_dataset}, 
            config=config, 
            mean=mean, 
            std=std
        )
    
    print("Synthetic data generation is complete!")
    wandb.finish()


if __name__ == "__main__":
    # Load the datasets and train VAE from classifier representations
    train_vae()


from typing import Callable

import numpy as np
import torch


def sample_from_decoder(
    decoder: torch.nn.Module,
    continuous_column_ix: list[int],
    categorical_groups: list[list[int]],
    latent_dim: int = 128,
    num_samples: int = 100,
    num_std: float = 1.0,
    coerce_probs: bool = False,
    act_fn: Callable[[torch.Tensor], torch.Tensor] = None,
    clamp: bool = False,
    device: torch.device = None,
):
    """Sample from the decoder of a VAE model."""
    data = []
    mean = torch.zeros(num_samples, latent_dim)
    std = mean + num_std
    if device:
        mean = mean.to(device)
        std = std.to(device)
    noise = torch.normal(mean=mean, std=std)
    gen_samples = decoder(noise)

    # Coerce probs for categorical variables, if necessary
    if coerce_probs:
        for cat_group in categorical_groups:
            max_idx = torch.argmax(gen_samples[:, cat_group], dim=1, keepdims=True)
            gen_samples[:, cat_group] = torch.zeros_like(gen_samples[:, cat_group]).scatter_(
                1, max_idx, 1.0
            )

    # Apply an activation function to continuous variable values, if necessary
    if act_fn:
        gen_samples[:, continuous_column_ix] = act_fn(gen_samples[:, continuous_column_ix])

    # Clamp values of continuous columns to [0,1] if necessary
    if clamp:
        gen_samples[:, continuous_column_ix] = gen_samples[:, continuous_column_ix].clamp(0.0, 1.0)

    data.append(gen_samples.detach().cpu().numpy())

    return np.concatenate(data, axis=0)



def generate_reconstructed_data(encoder, decoder, classifier, real_data, deterministic=True):
    """ Generate reconstructed data by passing the real data through the VAE.
    
    Args:
        encoder (torch.nn.Module): The encoder model that outputs mu and sigma.
        decoder (torch.nn.Module): The decoder model that generates data from z.
        classifier (torch.nn.Module): The classifier model to get latent features.
        real_data (torch.Tensor): The real data points.
        deterministic (bool): Whether to sample z deterministically as mu (default: True).

    Returns:
        torch.Tensor: The generated synthetic data.
    """
    # Set all models to evaluation mode
    encoder.eval()
    decoder.eval()
    classifier.eval()
    with torch.no_grad():
        # Get latent representation of real data
        latent_repr = classifier.get_features(real_data)
        
        # Obtain mean (mu) and standard deviation (sigma) from encoder
        mu, sigma = encoder(latent_repr)
        # Sample z from the encoder (deterministic or stochastic)
        z = encoder.sampling(mu, sigma, deterministic=deterministic)
        
        # Generate synthetic data from z
        generated_data = decoder(z)
    
    return generated_data

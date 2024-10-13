# from discovery/research/automl/predictive_feature_reconstruction/losses.py

import torch


def l2_dist(x: torch.Tensor , y: torch.Tensor) -> torch.Tensor:
  """Compute the L2 distance between two tensors."""
  return (x - y)**2

def l1_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the L1 distance between two tensors."""
    return torch.abs(x - y)


def weighted_l2_dist(x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
  """Compute the weighted L2 distance between two tensors."""
  l2_distance = (x - y)**2

  # Get the weights corresponding to the class/label of each sample
  class_indices = torch.argmax(labels, dim=1)
  weights = class_weights[class_indices]
  weights = weights.view(-1, 1, 1, 1)

  # Apply the weights to l2_distance
  weighted_l2 = l2_distance * weights
  return weighted_l2

def kldiv(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
  """Compute the KL divergence between two tensors."""
  return -0.5 * (1 + torch.log(sigma**2) - mu**2 - sigma**2)
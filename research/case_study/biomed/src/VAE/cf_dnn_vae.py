import torch
import torch.nn.functional as F
from torch import nn


class VariationalEncoder(nn.Module):
    """Variational Encoder class.

    Args:
    ----
        feature_dim: int, number of features in the latent space
        hidden_dim: int, number of hidden units in the encoder
        latent_dim: int, number of latent dimensions

    Returns:
    -------
          mu: tensor, mean of the latent space
          sigma: tensor, standard deviation of the latent space

    """

    def __init__(self, feature_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """Initialise the variational encoder."""
        super(VariationalEncoder, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.fc4 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the variational encoder."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        sigma = torch.exp(0.5 * self.fc4(x))
        return mu, sigma
    
    def sampling(
        self, mu: torch.Tensor, sigma: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample z using the reparametrisation trick."""
        epsilon = torch.randn_like(sigma) if not deterministic else torch.zeros_like(sigma)
        return mu + epsilon * sigma

    def forward_and_sample(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the variational encoder followed by z sampling."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        logvar = self.fc4(x)
        sigma = torch.exp(0.5 * logvar)
        return self.sampling(mu, sigma), mu, sigma, logvar


class Decoder(torch.nn.Module):
    def __init__(self, num_inputs: int, hidden_dim: int, latent_dim: int = 2, act_fn: callable = None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_inputs)
        self.act_fn = act_fn

    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        return self.act_fn(self.fc2(x)) if self.act_fn else self.fc2(x)

# ruff: noqa: E402
# ruff: noqa: I001


import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)


class ImageVariationalEncoder(nn.Module):
    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 1024, latent_dim: int = 128):
        super(ImageVariationalEncoder, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        mu = self.fc_mu(x)
        sigma = torch.exp(0.5 * self.fc_logvar(x))
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
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        sigma = torch.exp(0.5 * logvar)
        return self.sampling(mu, sigma), mu, sigma, logvar


# Adapted Decoder Res Up block from: https://github.com/julschoen/Latent-Space-Exploration-CT/blob/main/Models/VAE.py
#Residual up sampling block for the decoder
#Nearest neighbour is used to perform the upsampling - Not used
class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(ResUpBlock, self).__init__()
        self.scale_factor = scale
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        # self.dropout1 = nn.Dropout2d(0.2)

        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.dropout2 = nn.Dropout2d(0.2)
        
        # Skip connection - Identity
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # Upsample input
        x_up = self.upsample(x)

        # Skip connection
        skip = self.conv_skip(x_up)

        # Main branch
        x = F.rrelu(self.bn1(self.conv1(x_up)))
        # x = self.dropout1(x)
        x = self.bn2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        # x = self.dropout2(x)
        return x

class ResImageDecoder2(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, output_channels=1):
        super(ResImageDecoder2, self).__init__()
        self.dim_z = latent_dim
        
        self.conv1 = ResUpBlock(latent_dim, base_channels * 32, scale=1)  # 1x1 -> 1x1 (latent_dim -> 2048)
        self.conv2 = ResUpBlock(base_channels * 32, base_channels * 16, scale=2)  # 1x1 -> 2x2  (2048 -> 1024)
        self.conv3 = ResUpBlock(base_channels * 16, base_channels * 16, scale=2)  # 2x2 -> 4x4  (1024 -> 1024)
        self.conv4 = ResUpBlock(base_channels * 16, base_channels * 8, scale=1.75)  # 4x4 -> 7x7  (1024 -> 512)
        self.conv5 = ResUpBlock(base_channels * 8, base_channels * 8, scale=2)  # 7x7 -> 14x14  (512 -> 512)
        self.conv6 = ResUpBlock(base_channels * 8, base_channels * 4, scale=2)  # 14x14 -> 28x18  (512 -> 256)
        self.conv7 = ResUpBlock(base_channels * 4, base_channels * 2, scale=2)  # 28x28 -> 56x56  (256 -> 128)
        self.conv8 = ResUpBlock(base_channels * 2, base_channels, scale=2)  # 56x56 -> 112x112  (128 -> 64)
        self.conv9 = ResUpBlock(base_channels, base_channels // 2, scale=2)  # 112x112 -> 224x224  (64 -> 32)
        self.conv10 = nn.Conv2d(base_channels // 2, output_channels, kernel_size=3, stride=1, padding=1)  # 224x224 -> 224x224  (32 -> 1)
        # self.act = nn.Tanh()

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)  # Reshape to 4D tensor
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        
        # return self.act(x)
        return x


# OLD VERSION
class ResImageDecoder(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, output_channels=1):
        super(ResImageDecoder, self).__init__()
        
        self.conv1 = ResUpBlock(latent_dim, base_channels * 16, scale=7)  # 1x1 -> 7x7
        self.conv2 = ResUpBlock(base_channels * 16, base_channels * 8)  # 7x7 -> 14x14
        self.conv3 = ResUpBlock(base_channels * 8, base_channels * 4)  # 14x14 -> 28x28
        self.conv4 = ResUpBlock(base_channels * 4, base_channels * 2)  # 28x28 -> 56x56
        self.conv5 = ResUpBlock(base_channels * 2, base_channels)  # 56x56 -> 112x112
        self.conv6 = ResUpBlock(base_channels, base_channels // 2)  # 112x112 -> 224x224
        self.conv7 = nn.Conv2d(base_channels // 2, output_channels, kernel_size=3, stride=1, padding=1)
        
        # self.act = nn.Tanh()  # ??
    
    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3) # Reshape to 4D tensor

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        # x = self.act(x)

        return x



# OLD VERSION
class ImageDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 1024, output_channels: int = 1, image_size: int = 224):
        super(ImageDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2048)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.bn_fc2 = nn.BatchNorm1d(2048)

        # ConvTranspose2d layers to go from 1x1 -> 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        self.conv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=7, stride=1, padding=0)  # 1x1 -> 7x7
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)  # 7x7 -> 14x14
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 14x14 -> 28x28
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 28x28 -> 56x56
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 56x56 -> 112x112
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)  # 112x112 -> 224x224
        self.act = nn.Tanh()  # ??

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        
        x = x.unsqueeze(2).unsqueeze(3) # Reshape to 4D tensor

        x = F.relu(self.bn1(self.conv1(x)))  # 1x1 -> 7x7
        x = F.relu(self.bn2(self.conv2(x)))  # 7x7 -> 14x14
        x = F.relu(self.bn3(self.conv3(x)))  # 14x14 -> 28x28
        x = F.relu(self.bn4(self.conv4(x)))  # 28x28 -> 56x56
        x = F.relu(self.bn5(self.conv5(x)))  # 56x56 -> 112x112
        x = self.conv6(x)  # 112x112 -> 224x224

        # x = self.act(x)
        
        return x  # Output size is 224x224x1
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(
            out_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class InceptionV3Adapted(nn.Module):
    def __init__(self, num_channels=10, num_classes=3, dropout=0.5, pretrained=True):
        super(InceptionV3Adapted, self).__init__()
        # Load the Inception V3 model with specified weights (default: no pretrained weights)
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        self.inceptionV3 = models.inception_v3(weights=weights)

        # Modify the first convolutional layer to accept num_channels input channels
        self.inceptionV3.Conv2d_1a_3x3 = BasicConv2d(
            num_channels, 32, kernel_size=3, stride=2, bias=False
        )
        # Initialise the new first convolutional layer
        # nn.init.kaiming_normal_(
        #     self.inceptionV3.Conv2d_1a_3x3.conv.weight, mode="fan_out", nonlinearity="relu"
        # )

        # Modify the fully connected layer to match the number of classes
        self.inceptionV3.fc = nn.Linear(2048, num_classes)  # 3 classes: Control, Type I, Type II

    # def _transform_input(self, x):
    #     if self.transform_input:
    #         x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    #         x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    #         x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    #         x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
    #     return x

    def _transform_input(self, x):
        if self.transform_input:
            channels = []
            for i in range(10):
                x_ch = torch.unsqueeze(x[:, i], 1)
                channels.append(x_ch)

            x = torch.cat(channels, 1)
        return x

    def forward(self, x):
        x = self.inceptionV3(x)
        return x[0]


if __name__ == "__main__":
    inception_v3 = InceptionV3Adapted(num_channels=10, num_classes=3, dropout=0.5, pretrained=True)
    print(inception_v3)

    # # Ensure all layers are trainable
    # for param in model.parameters():
    #     param.requires_grad = True

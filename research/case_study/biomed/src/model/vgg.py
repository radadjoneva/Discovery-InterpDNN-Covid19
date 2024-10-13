import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights, VGG19_Weights


class VGG16Adapted(nn.Module):
    def __init__(self, num_channels=10, num_classes=3, dropout=0.5, pretrained=True):
        super(VGG16Adapted, self).__init__()
        # Load the VGG-16 model wih specified weights (default: no pretrained weights)
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vgg16 = models.vgg16(weights=weights)

        # Modify the first convolutional layer to accept num_channels input channels
        self.vgg16.features[0] = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        # Initialise the new first convolutional layer ??
        # nn.init.kaiming_normal_(self.vgg16.features[0].weight, mode="fan_out", nonlinearity="relu")

        # Modify the classifier to match the number of classes
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),  # 3 classes: Type I, Type II, Control
        )

    def forward(self, x):
        x = self.vgg16.features(x)
        x = self.vgg16.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg16.classifier(x)
        return x


class VGG19Adapted(nn.Module):
    def __init__(self, num_channels=10, num_classes=3, dropout=0.5, pretrained=True):
        super(VGG19Adapted, self).__init__()
        # Load the VGG-19 model wih specified weights (default: no pretrained weights)
        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        self.vgg19 = models.vgg19(weights=weights)

        # Modify the first convolutional layer to accept num_channels input channels
        self.vgg19.features[0] = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        # Initialise the new first convolutional layer ??
        # nn.init.kaiming_normal_(self.vgg16.features[0].weight, mode="fan_out", nonlinearity="relu")

        # Modify the classifier to match the number of classes
        self.vgg19.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),  # 3 classes: Type I, Type II, Control
        )

    def forward(self, x):
        x = self.vgg19.features(x)
        x = self.vgg19.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg19.classifier(x)
        return x


if __name__ == "__main__":
    pretrained = True
    # VGG-16
    vgg16 = VGG16Adapted(num_channels=10, num_classes=3, dropout=0.5, pretrained=pretrained)
    print(vgg16)

    # VGG-19
    vgg19 = VGG19Adapted(num_channels=10, num_classes=3, dropout=0.5, pretrained=pretrained)
    print(vgg19)

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNet50Adapted(nn.Module):
    def __init__(self, num_channels=1, num_classes=3, pretrained=True):
        super(ResNet50Adapted, self).__init__()
        # Load the ResNet-50 model with specified weights (default: no pretrained weights)
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet50 = models.resnet50(weights=weights)

        # Modify the first convolutional layer to accept num_channels input channels
        self.resnet50.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify the fully connected layer to match the number of classes
        self.resnet50.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        return x

    def get_features(self, x, layer='avgpool'):
        # Extract features from the avgpool layer or from all layers
        activations = {}

        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        activations['conv1'] = x.clone().detach()
        if layer == 'conv1':
            return x

        x = self.resnet50.layer1(x)
        activations['layer1'] = x.clone().detach()
        if layer == 'layer1':
            return x
        x = self.resnet50.layer2(x)
        activations['layer2'] = x.clone().detach()
        if layer == 'layer2':
            return x
        x = self.resnet50.layer3(x)
        activations['layer3'] = x.clone().detach()
        if layer == 'layer3':
            return x
        x = self.resnet50.layer4(x)
        activations['layer4'] = x.clone().detach()
        if layer == 'layer4':
            return x

        x = self.resnet50.avgpool(x)  # Extract features after avgpool
        x = torch.flatten(x, 1)  # Flatten to prepare for the fully connected layer
        activations['avgpool'] = x.clone().detach()

        if layer == 'avgpool':
            return x
        
        x = self.resnet50.fc(x)
        activations['fc'] = x.clone().detach()
        
        if layer == 'all':
            return activations
        else:
            raise ValueError(f"Invalid layer name: {layer}")

    # def register_hooks(self, recorder):
    #     # Register hooks on the model's layers
    #     self.resnet50.conv1.register_forward_hook(recorder.get_activation('conv1'))
    #     self.resnet50.bn1.register_forward_hook(recorder.get_activation('bn1'))
    #     self.resnet50.layer1.register_forward_hook(recorder.get_activation('layer1'))
    #     self.resnet50.layer2.register_forward_hook(recorder.get_activation('layer2'))
    #     self.resnet50.layer3.register_forward_hook(recorder.get_activation('layer3'))
    #     self.resnet50.layer4.register_forward_hook(recorder.get_activation('layer4'))
    #     self.resnet50.avgpool.register_forward_hook(recorder.get_activation('avgpool'))
    #     self.resnet50.fc.register_forward_hook(recorder.get_activation('fc'))


if __name__ == "__main__":
    # resnet50 = ResNet50Adapted(num_channels=10, num_classes=3, pretrained=True)
    # print(resnet50)

    resnet50 = ResNet50Adapted(num_channels=1, num_classes=3, pretrained=True)
    print(resnet50)

    # Create a single-channel (grayscale) image of size 1120x1120
    input_image = torch.randn(1, 1, 1120, 1120)
    output = resnet50(input_image)

    print("Output shape:", output.shape)
    print("Output:", output)

    # # Ensure all layers are trainable
    # for param in model.parameters():
    #     param.requires

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VGGSimple(nn.Module):
    def __init__(self, input_dim=(200, 200, 1), dropout=0.5):
        super(VGGSimple, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(input_dim[2], 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 25 * 25, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(
            -1, 16 * int((self.input_dim[0] / 8) * (self.input_dim[0] / 8))
        )  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def ct_images_cnn():
    model = VGGSimple(input_dim=(200, 200, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion
    # return model


def ct_patient_cnn():
    model = VGGSimple(input_dim=(200, 200, 10))
    optimizer = optim.Adam(model.parameters(), lr=0.0007, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

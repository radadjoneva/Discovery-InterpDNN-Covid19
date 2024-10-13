# ruff: noqa: E402
# ruff: noqa: I001

import os
import sys
import torch.nn as nn
import torch.nn.functional as F

# Add biomed directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

class DNNSimple(nn.Module):
    def __init__(self, input_dim, dropout=0.5, batch_norm=False):
        super(DNNSimple, self).__init__()

        self.batch_norm = batch_norm

        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128) if self.batch_norm else None
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64) if self.batch_norm else None
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64) if self.batch_norm else None
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(64, 48)
        self.bn4 = nn.BatchNorm1d(48) if self.batch_norm else None
        self.dropout4 = nn.Dropout(dropout)

        self.fc5 = nn.Linear(48, 16)
        self.bn5 = nn.BatchNorm1d(16) if self.batch_norm else None
        self.dropout5 = nn.Dropout(dropout)

        self.fc6 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.batch_norm:
            x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        if self.batch_norm:
            x = self.bn2(x)
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        if self.batch_norm:
            x = self.bn3(x)
        x = self.dropout3(x)

        x = F.relu(self.fc4(x))
        if self.batch_norm:
            x = self.bn4(x)
        x = self.dropout4(x)

        x = F.relu(self.fc5(x))
        if self.batch_norm:
            x = self.bn5(x)
        x = self.dropout5(x)

        x = self.fc6(x)
        return x

    def get_features(self, x, layer=5):
        # Extract features from the 4th or 5th layer (before layer 6 - classifier)
        x = F.relu(self.fc1(x))
        if self.batch_norm:
            x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        if self.batch_norm:
            x = self.bn2(x)
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        if self.batch_norm:
            x = self.bn3(x)
        x = self.dropout3(x)

        x = F.relu(self.fc4(x))
        if self.batch_norm:
            x = self.bn4(x)

        if layer == 4:
            # Return features from the 4th layer
            features = x
            return features
        elif layer == 5:
            # Return features from the 5th (penultimate) layer
            x = self.dropout4(x)
            x = F.relu(self.fc5(x))
            if self.batch_norm:
                x = self.bn5(x)
            features = x
            return features
        else:
            raise ValueError("Invalid layer number")
    
    def register_hooks(self, recorder):
        # Register hooks on the model's layers for a given patient_id
        self.fc1.register_forward_hook(recorder.get_activation('fc1'))
        self.fc2.register_forward_hook(recorder.get_activation('fc2'))
        self.fc3.register_forward_hook(recorder.get_activation('fc3'))
        self.fc4.register_forward_hook(recorder.get_activation('fc4'))
        self.fc5.register_forward_hook(recorder.get_activation('fc5'))
        self.fc6.register_forward_hook(recorder.get_activation('fc6'))



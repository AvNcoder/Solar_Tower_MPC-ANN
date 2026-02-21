import torch
import torch.nn as nn

class SolarMPCNet(nn.Module):
    def __init__(self, input_dim):
        super(SolarMPCNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Outputs the Flow Rate
        )

    def forward(self, x):
        return self.network(x)
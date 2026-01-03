"""
Docstring for models
In this we will implement various models that we will use.
"""

import torch
import torch.nn as nn


class ThreeLayerMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim):
        super(ThreeLayerMLP, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim

        # as per the paper, authors are using 3 layers
        # with tanh activations
        self.layers = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.out_features)
        )

    def forward(self, x):
        return self.layers(x)

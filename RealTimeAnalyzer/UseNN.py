# The code in this file is adapted from a pytorch tutorial by codemy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    # Input layer moves to Hidden Layer 1, H2, etc, output
    def __init__(self, in_features=6, h1=8, h2=9, out_features=2):
        # For my case the in_features will be all the angles and out_features will be 2 (good or bad)
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # Pushes data to the next layer
    def forward(self, x):
        # relu stands for rectified linear unit
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Allows for live inputs and outputs
class User():
    def __init__(self):
        # Load saved model
        self.model = Model()
        self.model.load_state_dict(torch.load('/Users/jasper/Desktop/ATFinal/AT-Final-Project/PytorchML/model.pt'))

    def use(self, x_values):
        # torch.no_grad basically ignores back propagation
        with torch.no_grad():
            y_val = self.model.forward(torch.FloatTensor(x_values))
            result = y_val.argmax().item()

        print(result)
        return result


if __name__ == "__main__":
    user = User()

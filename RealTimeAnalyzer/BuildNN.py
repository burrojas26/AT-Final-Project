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


class Trainer():
    # Train the model based on the dataset
    def train(self):
        # Pick a random seed
        torch.manual_seed(38)
        # Create an instance of model
        model = Model()
        # Read the dataset and drop columns that are not needed
        df = pd.read_csv("/Users/jasper/Desktop/ATFinal/AT-Final-Project/Dataset/squatData.csv")
        df.drop('VideoNum', axis=1, inplace=True)
        df.drop('Frame', axis=1, inplace=True)
        df.drop('ElbowL', axis=1, inplace=True)
        df.drop('ElbowR', axis=1, inplace=True)
        df.drop('ShoulderL', axis=1, inplace=True)
        df.drop('ShoulderR', axis=1, inplace=True)

        # Set the X and Y values (X is the inputs and Y is the output)
        X = df.drop('Outcome', axis=1).values
        Y = df['Outcome'].values

        # Train Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=38)

        # Convert x and ys to float and long tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        Y_train = torch.LongTensor(Y_train)
        Y_test = torch.LongTensor(Y_test)

        # Set the criterion of model to measure the error
        criterion = nn.CrossEntropyLoss()
        # Choose Adam Optimizer (a common optimizer)
        # lr = learning rate (can adjust learning rate if the model is learning correctly)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train model
        epochs = 400  # Epochs are iterations of the data through the model
        losses = []
        for i in range(epochs):
            # Go forward and get a prediction
            y_pred = model.forward(X_train)

            # Measure the loss/error, will be high at first
            loss = criterion(y_pred, Y_train)  # Predicted vs the train value

            # Track losses
            losses.append(loss.detach().numpy())

            # Print every 10 epochs with the loss (the loss should go down)
            if i % 10 == 0:
                print(f'Epoch: {i} and loss: {loss}')

            # Do some back propagation: take the error rate of forward propagation
            # feed it back through to fine tune the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save nn model to be used later
        torch.save(model.state_dict(), 'model.pt')

        # Return the x and y test values, so they can be used for testing in the next func
        return [X_test, Y_test]

    # Test the model based on the pre-determined test values from the dataset
    def test(self, testValues):
        # Load saved model
        model = Model()
        model.load_state_dict(torch.load('model.pt'))
        correct = 0
        X_test = testValues[0]
        Y_test = testValues[1]
        # torch.no_grad basically ignores back propagation
        with torch.no_grad():
            for i, data in enumerate(X_test):
                y_val = model.forward(data)
                # Number, tensor object prediction, the correct value, the estimation
                print(f'{i + 1}. {str(y_val)} {Y_test[i]} {y_val.argmax().item()}')

                # Correct or not
                if y_val.argmax().item() == Y_test[i]:
                    correct += 1

        # Calculate percent correct
        print(f'\n{correct} out of {len(Y_test)} = {100 * correct / len(Y_test)}%')

# Allows for live inputs and outputs
class User():
    def use(self, x_values):
        # Load saved model
        model = Model()
        model.load_state_dict(torch.load('/Users/jasper/Desktop/ATFinal/AT-Final-Project/PytorchML/model.pt'))

        # torch.no_grad basically ignores back propagation
        with torch.no_grad():
            y_val = model.forward(torch.FloatTensor(x_values))
            result = y_val.argmax().item()

        print(result)
        return result


if __name__ == "__main__":
    myTrainer = Trainer()
    testValues = myTrainer.train()
    myTrainer.test(testValues)

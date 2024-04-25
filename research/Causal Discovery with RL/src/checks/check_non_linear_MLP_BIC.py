import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fcl = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        # x = torch.nn.functional.leaky_relu(self.fc2(x))
        # x = torch.nn.functional.leaky_relu(self.fc3(x))

        x = self.fcl(x)
        return x



path = '/Users/saraz/PycharmProjects/tf2/trustworthyAI/datasets/synthetic_datasets/exp3_10_nodes_gp/1'
# path = '/Users/shahrukhqasim/Downloads/temp/synth/gauss_same_noise/1'


with open(path + '/data.npy', 'rb') as f:
    data = np.load(f)
with open(path + '/DAG.npy', 'rb') as f:
    DAG = np.load(f)

print(data)
DAG = DAG != 0
DAG = DAG.astype(np.float32)
print(DAG)




DAG = DAG.transpose()  ## we should transpose (to be double checked)


print("transposed", DAG)
RSS_ls = []

for i in range(10):
    col = DAG[i]  # take the i-th row of the graph and store it in col

    # no parents, then simply use mean
    if np.sum(col) < 0.1:
        y_err = data[:, i]
        y_err = y_err - np.mean(y_err)

    else:
        cols_TrueFalse = col > 0.5  #set to True the elements of col that are greater than 0.5
        #print(cols_TrueFalse)

        X_train = data[:, cols_TrueFalse]  #take the columns of the input data that are trure
        # print("X_train.shape", X_train.shape)

        y_train = data[:, i]  # take the i-th column of the input data



        X_train = torch.tensor(X_train, dtype=torch.float32)


        #print("X_train", X_train)
        #print("X_trainshape ", X_train.shape)

        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_train = y_train.unsqueeze(1)
        #print("y_train", y_train)

        # y_train = torch.from_numpy(y_train).float()
        #input_size = 10
        hidden_size = 100  # Number of neurons in the hidden layer
        output_size = 1  # Regression output

        num_epochs = 100
        model = SimpleMLP(X_train.shape[1], hidden_size, output_size)

        criterion = nn.MSELoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            # print(outputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if (epoch + 1) % 100 == 0:
                # print(outputs)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        model.eval()
        with torch.no_grad():
            predictions = model(X_train)

        # Convert back to numpy
        y_pred = predictions.numpy()
        # print ("Predicted values: ", y_pred)

        y_train = y_train.numpy()
        # Compute the error
        y_err = y_pred - y_train

    RSSi = np.sum(np.square(y_err))
    print("RSSi", RSSi)
    # if the regresors include the true parents, GPR would result in very samll values, e.g., 10^-13
    # so we add 1.0, which does not affect the monotoniticy of the score
    RSSi += 1.0

    RSS_ls.append(RSSi)

print("RSS", RSS_ls)
num_samples = data.shape[0]
num_features = 10
bic_penalty = np.log(data.shape[0]) / data.shape[0]
BIC = np.log(np.sum(RSS_ls) / num_samples + 1e-8) + np.sum(DAG) * bic_penalty / num_features   #1000 num of samples, 10 num of features
print("BIC", BIC)

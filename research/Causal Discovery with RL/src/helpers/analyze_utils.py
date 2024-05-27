import pickle

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np



# Global dictionary to hold numpy arrays with keys
global_data = {}
calls = 0


class GraphMLP(nn.Module):
    def __init__(self, d):
        super(GraphMLP, self).__init__()
        self.d = d
        self.layers = nn.Sequential(
            nn.Linear(d, d * 10),  # Scaling up the feature space
            nn.ReLU(),
            nn.Linear(d * 10, d * d)  # Output layer to reshape into d x d matrix
        )

    def forward(self, x):
        x = self.layers(x)
        return x.view(-1, self.d, self.d)  # Reshape output to d x d matrix


def cache_write(key, data):
    global global_data, calls
    calls += 1
    print(calls)

    # Convert input data to numpy array and flatten
    data = np.asarray(data).flatten()

    # Check if the key already exists in the global dictionary
    if key in global_data:
        # Append new data to the existing array associated with the key
        global_data[key] = np.append(global_data[key], data)
    else:
        # Create a new entry in the dictionary with the key and data
        global_data[key] = data

    # Write the dictionary to a pickled file
    with open('data.pkl', 'wb') as file:
        pickle.dump(global_data, file)

    if calls == 5:
        0/0


def get_training_time(log_path):
    with open(log_path) as f:
        content = f.readlines()
        if content[0][:4] != '2019':
            return -1
        for i in reversed(range(len(content) - 30, len(content))):
            if content[i][:4] == '2019':
                final_idx = i
                break
        else:
            return -1

        start = datetime.strptime(content[0][:19], '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(content[final_idx][:19], '%Y-%m-%d %H:%M:%S')
        diff = end - start
        hours = diff.days * 24 + diff.seconds / (60.0 * 60.0)
        return hours

def mlp_model(input_dim):  #prova
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),  # Hidden layer
        Dense(1, activation='linear')  # Output layer for regression
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

def graph_prunned_by_mlp(graph_batch, X, th=0.3):  #prova
    d = len(graph_batch)
    W = []

    for i in range(d):
        col = np.abs(graph_batch[i]) > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]
        y = X[:, i]

        if X_train.size > 0:
            model = mlp_model(X_train.shape[1])
            model.fit(X_train, y, epochs=100, verbose=0)
            weights = model.layers[0].get_weights()[0]  #Weights[0] contains the weight matrix, Weights[1] contains biases
            new_reg_coeff = np.zeros(d, )
            cj = 0
            for ci in range(d):
                if col[ci]:
                    new_reg_coeff[ci] = weights[cj, 0]
                    cj += 1
            W.append(new_reg_coeff)
        else:
            print(f"Skipping training for feature {i} due to no data.")
            W.append(np.zeros(d))

    return np.float32(np.abs(W) > th)

def graph_prunned_by_coef(graph_batch, X, th=0.3):
    """
    for a given graph, pruning the edge according to edge weights;
    linear regression for each causal regresison for edge weights and then thresholding
    :param graph_batch: graph
    :param X: dataset
    :return:
    """
    d = len(graph_batch)
    reg = LinearRegression()
    W = []

    for i in range(d):
        col = np.abs(graph_batch[i]) > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]

        y = X[:, i]
        reg.fit(X_train, y)
        reg_coeff = reg.coef_

        cj = 0
        new_reg_coeff = np.zeros(d, )
        for ci in range(d):
            if col[ci]:
                new_reg_coeff[ci] = reg_coeff[cj]
                cj += 1

        W.append(new_reg_coeff)

    return np.float32(np.abs(W) > th)


def graph_prunned_by_coef_2nd(graph_batch, X, th=0.3):
    """
    for a given graph, pruning the edge according to edge weights;
    quadratic regression for each causal regresison for edge weights and then thresholding
    :param graph_batch: graph
    :param X: dataset
    :return:
    """
    d = len(graph_batch)
    reg = LinearRegression()
    poly = PolynomialFeatures()
    W = []

    for i in range(d):
        col = graph_batch[i] > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]
        X_train_expand = poly.fit_transform(X_train)[:, 1:]
        X_train_expand_names =  poly.get_feature_names_out()[1:]
        
        y = X[:, i]
        reg.fit(X_train_expand, y)
        reg_coeff = reg.coef_

        cj = 0
        new_reg_coeff = np.zeros(d, )

        for ci in range(d):
            if col[ci]:
                xxi = 'x{}'.format(cj)
                for iii, xxx in enumerate(X_train_expand_names):                
                    if xxi in xxx:
                        if np.abs(reg_coeff[iii]) > th:
                            new_reg_coeff[ci] = 1.0
                            break              
                cj += 1
        W.append(new_reg_coeff)

    return W


def visualize_result(result_dict):
    plot_recovered_graph(result_dict['best_graph_np'],
                         result_dict['true_graph_np'])
    print('bic: {}'.format(result_dict['bic']))
    print('tpr: {}'.format(result_dict['tpr']))
    print('fdr: {}'.format(result_dict['fdr']))
    print('fpr: {}'.format(result_dict['fpr']))
    print('shd: {}'.format(result_dict['shd']))


def get_config(log_path):
    # TODO: A lot of hardcoding, might want to improve this to regex for readability and efficiency
    # Decode training log to get the config parameters
    with open(log_path) as f:
        content = f.readlines()
        for line in content:
            if 'Configuration parameters' in line:
                config = eval('{' + line.split('{')[1].split('}')[0] + '}')
                return config


def plot_recovered_graph(recovered_graph_np, true_graph_np, save_name=None):
    fig = plt.figure(2)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('recovered_graph')
    ax.imshow(recovered_graph_np, cmap=plt.cm.gray)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('ground truth')
    ax.imshow(true_graph_np, cmap=plt.cm.gray)
    if save_name is not None:
        plt.savefig(save_name)


def get_true_graph_int(log_path):
    # TODO: A lot of hardcoding, might want to improve this to regex for readability and efficiency
    # Decode training log to get true_graph_int
    with open(log_path) as f:
        content = f.readlines()
        for line in content:
            if 'training_set.true_graph_int' in line:
                true_graph_int = eval(line.split('-')[-1].split(':')[-1][1:-1])
                return true_graph_int


def convert_graph_int_to_adj_mat(graph_int):
    # Convert graph int to binary adjacency matrix
    # TODO: Make this more readable
    return np.array([list(map(int, ((len(graph_int) - len(np.base_repr(curr_int))) * '0' + np.base_repr(curr_int))))
                     for curr_int in graph_int], dtype=int)


def count_accuracy(B_true, B, B_und=None) -> tuple:

    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        B_true: ground truth graph
        B: predicted graph
        B_und: predicted undirected edges in CPDAG, asymmetric

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive

    Codes are from NOTEARS authors.
    """

    d = B.shape[0]
    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)

    if B_und is not None:
        B_lower = np.add(B_lower, np.tril(B_und + B_und.T), out=B_lower, casting="unsafe")
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    acc_res = {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'pred_size': pred_size}
    return acc_res
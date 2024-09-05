import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import pdist, squareform

from rewards.Reward_BIC import PyTorchMLP2
import torch
from torch import nn
from torch import optim
import pandas as pd
from scipy.special import softmax


def fit_and_err(X_train, y_train):  # Working for linear case : calculate_LR ()

    device = 'cuda'

    #TODO: add the crossentropy
    input_size = X_train.shape[1]
    output_size = y_train.shape[1] if len(y_train.shape)> 1 else 1
    model = PyTorchMLP2(input_size=input_size,hidden_size=64,output_size= output_size).to(device)

    # Convert X_train, y_train from NumPy arrays to PyTorch tensors
    X_train_torch = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_train_torch = torch.from_numpy(y_train.astype(np.float32)).to(device)

    criterion = nn.CrossEntropyLoss().to(device) if output_size>1 else nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    #model.train()
    num_epochs = 100
    for epoch in range(2):
        # print("ccc")
        optimizer.zero_grad()
        outputs = model(X_train_torch)
        #print("OUTPUTTT", outputs)
        #print("y_train", y_train.shape)
        #print ("Y_:TRAIN 1", y_train)
        loss = criterion(outputs, y_train_torch)  # _torch.view(-1, 1))

        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 100 == 0:
            # print(outputs)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # print("qr")

    # Making predictions (inference)
    model.eval()
    with torch.no_grad():
        predictions = model(X_train_torch)

    # Convert back to numpy
    y_pred = predictions.cpu().numpy()
    # Convert logits to probabilities
    probabilities = softmax(y_pred, axis=1)

    # Compute the error
    # y_err = y_pred.flatten() - y_train
    y_train = y_train_torch.cpu().numpy()
    y_err = y_train - probabilities
    return y_err


def BIC_input_graph(X, g, reg_type='LR', score_type='BIC',mappings=None):
    """cal BIC score for given graph"""

    RSS_ls = []
    RSS_ls_2 = []

    n, d = X.shape

    if reg_type in ('LR', 'QR'):
        reg = LinearRegression()
    else:
        reg = GaussianProcessRegressor()

    poly = PolynomialFeatures()

    for i in range(d):
        y_ = X[:, [i]]  # it separates the feature corresponding to the current index i from the rest of the data.
        inds_x = list(np.abs(g[i]) > 0.1)

        #TODO: get dummies 
        df_y = pd.DataFrame(y_)
        # Convert columns that can be converted to numeric
        for col in df_y.columns:
            try:
                df_y[col] = pd.to_numeric(df_y[col])
            except ValueError:
                pass
        y_dummy = pd.get_dummies(df_y).to_numpy()

        mlp_success = False
        if np.sum(inds_x) < 0.1:
            y_pred = np.mean(y_dummy)
        else:
            X_ = X[:, inds_x]
            #TODO: get dummies 
            df_X = pd.DataFrame(X_)
            # Convert columns that can be converted to numeric
            for col in df_X.columns:
                try:
                    df_X[col] = pd.to_numeric(df_X[col])
                except ValueError:
                    pass
            X_dummy = pd.get_dummies(df_X).to_numpy()
            
            if reg_type == 'QR':
                X_dummy = poly.fit_transform(X_dummy)[:, 1:]
            elif reg_type == 'GPR':
                med_w = np.median(pdist(X_dummy, 'euclidean'))
                X_dummy = X_dummy / med_w
            reg.fit(X_dummy, y_dummy)
            y_pred = reg.predict(X_dummy)
            mlp_success = True
            err_mlp = fit_and_err(X_dummy, y_dummy)
        RSSi = np.sum((y_dummy - y_pred)**2)
        if mlp_success:
            RSSi_2 = np.sum(err_mlp**2)
        else:
            RSSi_2 = RSSi

        if reg_type == 'GPR':
            RSS_ls.append(RSSi + 1.0)
        else:
            RSS_ls.append(RSSi)
        RSS_ls_2.append(RSSi_2)

    if score_type == 'BIC':
        return np.log(np.sum(RSS_ls) / n + 1e-8), np.log(np.sum(RSS_ls_2) / n + 1e-8)
    elif score_type == 'BIC_different_var':
        return np.sum(np.log(np.array(RSS_ls) / n) + 1e-8), np.sum(np.log(np.array(RSS_ls_2) / n) + 1e-8)


def BIC_lambdas(X, mappings = None, gl=None, gu=None, gtrue=None, reg_type='LR', score_type='BIC'):
    """
    :param X: dataset
    :param gl: input graph to get score lower bound
    :param gu: input graph to get score upper bound
    :param gtrue: input true graph
    :param reg_type:
    :param score_type:
    :return: score lower bound, score upper bound, true score (only for monitoring)
    """
    

    # Function to reverse the mapping #TODO:add it general and import in dataset_read_data?
    def reverse_mapping(mappings):
        reversed_mappings = {}
        for key, mapping in mappings.items():
            reversed_mappings[key] = {v: k for k, v in mapping.items()}
        return reversed_mappings

    # Function to apply the reversed mapping to the matrix
    def apply_reversed_mapping(matrix, reversed_mappings):
        matrix_reversed = matrix.astype(object)  # Convert to object type to hold strings
        for i, key in enumerate(reversed_mappings.keys()):
            col_idx = int(key)  # Adjust column index to zero-based (if keys are 1-based)
            if col_idx < matrix.shape[1]:  # Ensure we're within the bounds of the matrix columns
                matrix_reversed[:, col_idx] = [reversed_mappings[key][float(value)] for value in matrix[:, col_idx]]
        return matrix_reversed

    if mappings!= None:
        # Reverse the mappings
        reversed_mappings = reverse_mapping(mappings)

        # Apply the reversed mapping to the matrix
        X = apply_reversed_mapping(X, reversed_mappings)

    n, d = X.shape

    if score_type == 'BIC':
        bic_penalty = np.log(n) / (n * d)
    elif score_type == 'BIC_different_var':
        bic_penalty = np.log(n) / n

    # default gl for BIC score: complete graph (except digonals)
    if gl is None:
        g_ones = np.ones((d, d))
        for i in range(d):
            g_ones[i, i] = 0
        gl = g_ones

    # default gu for BIC score: empty graph
    if gu is None:
        gu = np.zeros((d, d))

    sl, sl_mlp = BIC_input_graph(X, gl, reg_type, score_type,mappings)  # Bic score for lower bound graph
    print("sl_mlp",sl_mlp)
    su, su_mlp = BIC_input_graph(X, gu, reg_type, score_type,mappings)  # Bic score for upper bound graph

    if gtrue is None:
        strue = sl - 10
        strue_mlp = sl_mlp - 10
    else:
        print(BIC_input_graph(X, gtrue, reg_type, score_type,mappings))  # print initial BIC score for true graph
        print(gtrue)
        print(bic_penalty)
        strue, strue_mlp = BIC_input_graph(X, gtrue, reg_type, score_type,mappings) + np.sum(
            gtrue) * bic_penalty  # final score for true graph with penalty
        # notice that penalty discourages the use of more variables in the model (more edges in the graph)
        # so the score is penalized with the number of edges in the graph

    # print("slX", sl, sl_mlp)
    # print("suX", su, su_mlp)
    # print("strueX", strue, strue_mlp)
    # 0/0
    # return sl, su, strue  #return lower bound, upper bound and true score
    return sl_mlp, su_mlp, strue_mlp  # TODO: This is new one, with MLP. | return lower bound, upper bound and true score

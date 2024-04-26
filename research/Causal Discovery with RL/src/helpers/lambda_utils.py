import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import pdist, squareform

from rewards.Reward_BIC import PyTorchMLP2
import torch
from torch import nn
from torch import optim

def fit_and_err(X_train, y_train):  # Working for linear case : calculate_LR ()
    device = 'mps'

    input_size = X_train.shape[1]
    model = PyTorchMLP2(input_size=input_size, hidden_size=64, output_size=1).to(device)

    # Convert to pytorch
    X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)
    y_train = y_train.unsqueeze(1)  # going ftom 1-D to 2-D
    # print("X_train", X_train)

    # print (y_train)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # , weight_decay=1e-5)

    # Training loop
    # model.train()
    num_epochs = 100
    for epoch in range(num_epochs):
        # print("ccc")

        outputs = model(X_train)
        loss = criterion(outputs, y_train)  # _torch.view(-1, 1))

        optimizer.zero_grad()
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
        predictions = model(X_train)

    # Convert back to numpy
    y_pred = predictions.cpu().numpy()

    # Compute the error
    # y_err = y_pred.flatten() - y_train
    y_train = y_train.cpu().numpy()
    y_err = y_pred - y_train

    # print(y_err)

    return y_err


def BIC_input_graph(X, g, reg_type='LR', score_type='BIC'):
    """cal BIC score for given graph"""

    RSS_ls = []
    RSS_ls_2 = []

    n, d = X.shape 

    if reg_type in ('LR', 'QR'):
        reg = LinearRegression()
    else:
        reg =GaussianProcessRegressor()

    poly = PolynomialFeatures()

    for i in range(d):
        y_ = X[:, [i]]   #it separates the feature corresponding to the current index i from the rest of the data.
        inds_x = list(np.abs(g[i])>0.1)

        mlp_success = False
        if np.sum(inds_x) < 0.1:   
            y_pred = np.mean(y_)
        else:
            X_ = X[:, inds_x]
            if reg_type == 'QR':              
                X_ = poly.fit_transform(X_)[:, 1:] 
            elif reg_type == 'GPR':                
                med_w = np.median(pdist(X_, 'euclidean'))
                X_ = X_ / med_w
            reg.fit(X_, y_)
            y_pred = reg.predict(X_)
            mlp_success = True
            err_mlp = fit_and_err(X_, y_)
        RSSi = np.sum(np.square(y_ - y_pred))
        if mlp_success:
            RSSi_2 = np.sum(err_mlp)
        else:
            RSSi_2 = RSSi

        if reg_type == 'GPR':
            RSS_ls.append(RSSi+1.0)
        else:
            RSS_ls.append(RSSi)
        RSS_ls_2.append(RSSi_2)

    if score_type == 'BIC':
        return np.log(np.sum(RSS_ls)/n+1e-8), np.log(np.sum(RSS_ls_2)/n+1e-8)
    elif score_type == 'BIC_different_var':
        return np.sum(np.log(np.array(RSS_ls)/n)+1e-8), np.sum(np.log(np.array(RSS_ls_2)/n)+1e-8)
    
    
def BIC_lambdas(X, gl=None, gu=None, gtrue=None, reg_type='LR', score_type='BIC'):
    """
    :param X: dataset
    :param gl: input graph to get score lower bound
    :param gu: input graph to get score upper bound
    :param gtrue: input true graph
    :param reg_type:
    :param score_type:
    :return: score lower bound, score upper bound, true score (only for monitoring)
    """
        
    n, d = X.shape

    if score_type == 'BIC':
        bic_penalty = np.log(n) / (n*d)
    elif score_type == 'BIC_different_var':
        bic_penalty = np.log(n) / n
    
    # default gl for BIC score: complete graph (except digonals)
    if gl is None:
        g_ones= np.ones((d,d))
        for i in range(d):
            g_ones[i, i] = 0
        gl = g_ones

    # default gu for BIC score: empty graph
    if gu is None:
        gu = np.zeros((d, d))

    sl, sl_mlp = BIC_input_graph(X, gl, reg_type, score_type) # Bic score for lower bound graph
    su, su_mlp = BIC_input_graph(X, gu, reg_type, score_type) # Bic score for upper bound graph

    if gtrue is None:
        strue = sl - 10
        strue_mlp = sl_mlp - 10
    else:
        print(BIC_input_graph(X, gtrue, reg_type, score_type))  #print initial BIC score for true graph
        print(gtrue)
        print(bic_penalty)
        strue, strue_mlp = BIC_input_graph(X, gtrue, reg_type, score_type) + np.sum(gtrue) * bic_penalty  #final score for true graph with penalty
        #notice that penalty discourages the use of more variables in the model (more edges in the graph)
        #so the score is penalized with the number of edges in the graph

    # print("slX", sl, sl_mlp)
    # print("suX", su, su_mlp)
    # print("strueX", strue, strue_mlp)
    # 0/0
    # return sl, su, strue  #return lower bound, upper bound and true score
    return sl_mlp, su_mlp, strue_mlp  # TODO: This is new one, with MLP. | return lower bound, upper bound and true score


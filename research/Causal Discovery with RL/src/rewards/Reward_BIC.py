import time

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.linalg import expm as matrix_exponential
from scipy.spatial.distance import pdist, squareform
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import logging
import tensorflow as tf

from helpers.debugger import print_mine, print_mine_np
import torch
import torch.nn as nn
import torch.optim as optim
#import os

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np


import matplotlib.pyplot as plt


class CustomModule(torch.nn.Module):
    def __init__(self, total_cols, features):
        super(CustomModule, self).__init__()
        #self.W = torch.nn.Parameter(torch.randn((768, 8, 17)))
        self.W = torch.nn.Parameter(torch.randn((total_cols, features, 17)))
        #self.W2 = torch.nn.Parameter(torch.randn((768, 17, 1)))
        self.W2 = torch.nn.Parameter(torch.randn((total_cols, 17, 1)))

    def forward(self, D):
        # Reshape D and W for broadcasting
        W = self.W.to(D.device)
        W2 = self.W2.to(D.device)


        D = D.unsqueeze(2)  # Shape: (5000, 768, 1, 8)
        W = W.unsqueeze(0)  # Shape: (1, 768, 8, 17)
        W2 = W2.unsqueeze(0)  # Shape: (1, 768, 17, 1)

        x = torch.matmul(D, W)  # Perform matrix multiplication
        x = torch.relu(x)
        x = torch.matmul(x, W2)

        return x.squeeze(2)



class get_Reward(object):

    _logger = logging.getLogger(__name__)

    def __init__(self, batch_num, maxlen, dim, inputdata, sl, su, lambda1_upper, 
                 score_type='BIC', reg_type='LR', l1_graph_reg=0.0, verbose_flag=True):
        self.y_train_torch = None
        self.X_train_torch = None
        self.features = None
        self.total_cols = None
        self.batch_num = batch_num
        self.maxlen = maxlen # =d: number of vars
        self.dim = dim
        self.baseint = 2**maxlen
        self.d = {} # store results
        self.d_RSS = {} # store RSS for reuse
        self.inputdata = inputdata
        self.n_samples = inputdata.shape[0]
        self.l1_graph_reg = l1_graph_reg 
        self.verbose = verbose_flag
        self.sl = sl
        self.su = su
        self.lambda1_upper = lambda1_upper
        self.bic_penalty = np.log(inputdata.shape[0])/inputdata.shape[0]

        if score_type not in ('BIC', 'BIC_different_var'):
            raise ValueError('Reward type not supported.')
        if reg_type not in ('LR', 'QR', 'GPR'):
            raise ValueError('Reg type not supported')
        self.score_type = score_type
        self.reg_type = reg_type

        self.ones = np.ones((inputdata.shape[0], 1), dtype=np.float32)
        self.poly = PolynomialFeatures()


    def cal_rewards(self, graphs, lambda1, lambda2):
        rewards_batches = []

        print ("graphs shape", graphs.shape) #graph shape (64,12,12)
        print ("max length", self.maxlen) #graphi shape )

        all_X_train = []
        all_y_train = []
        all_indices = []

        max_features = 0  #to track the max num of features across all samples

        for graphi in graphs:
            for i in range(self.maxlen):  # A12
                col = graphi[i]
                if np.sum(col) < 0.1:
                    continue


                else:
                    cols_TrueFalse = tf.greater(tf.cast(col, tf.float32), tf.constant(0.5))
                    X_train = self.inputdata[:, cols_TrueFalse]
                    y_train = self.inputdata[:, i]
                    all_X_train.append(X_train)
                    all_y_train.append(y_train)
                    all_indices.append(i)
                    if X_train.shape[1] > max_features:
                        max_features = X_train.shape[1]  #updating max feautures

        # Padding arrays in all_X_train to have the same second dimension
        all_X_train_padded = [np.pad(x, ((0, 0), (0, max_features - x.shape[1])), 'constant', constant_values=0)
                              for x in all_X_train]

        # Convert lists to tensorrs
        self.X_train_torch = torch.from_numpy(np.stack(all_X_train_padded, axis=1).astype(np.float32))
        y_train_torch = torch.from_numpy(np.stack(all_y_train, axis=1).astype(np.float32))
        self.y_train_torch = y_train_torch.unsqueeze(-1)

        print("X_train shape: ", self.X_train_torch.shape)
        print("y_train shape: ", self.y_train_torch.shape)

        self.total_cols = self.X_train_torch.shape[1]
        self.features = self.X_train_torch.shape[2]


        ###### Trying =======
        # Instantiate the model
        self.model = CustomModule(self.total_cols, self.features).cuda()

        # Train the model
        self.train_model(self.model, self.X_train_torch.cuda(), self.y_train_torch.cuda())

        for graphi in graphs:
            reward_ = self.calculate_reward_single_graph(graphi, lambda1, lambda2) # reward_ is (reward, score, cycness)
            rewards_batches.append(reward_)

        return np.array(rewards_batches) # contains array of (reward, score, cycness) for each graph


    ####### regression 

    def calculate_yerr(self, X_train, y_train):
        if self.reg_type == 'LR':
            return self.calculate_LR(X_train, y_train)
        elif self.reg_type == 'QR':
            return self.calculate_QR(X_train, y_train)
        elif self.reg_type == 'GPR':
            return self.calculate_GPR(X_train, y_train)
        else:
            # raise value error
            assert False, 'Regressor not supported'

    # faster than LinearRegression() from sklearn

    '''def calculate_LR(self, X_train, y_train):
        X = np.hstack((X_train, self.ones))
        XtX = X.T.dot(X)
        Xty = X.T.dot(y_train)
        theta = np.linalg.solve(XtX, Xty)
        y_err = X.dot(theta) - y_train
        return y_err'''




    # Custom training loop
    def train_model(self, model,  X_train, y_train, num_epochs=10, learning_rate=0.001):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        X_train = X_train.cuda()
        y_train = y_train.cuda()

        for epoch in range(num_epochs):
            total_loss = 0.0
            start_time = time.time()

            # Generate random data
            #D = torch.randn((5000, 768, 8)).cuda()
            #D = torch.from_numpy(X_train.astype(np.float32)) #X_train
            # Forward pass
            output = model(X_train)
            print ("output shape: ", output.shape)

            #y_train_torch = torch.from_numpy(y_train.astype(np.float32))

            print ("y_train shape: ", y_train.shape)

            # Dummy loss, you should define your loss function here
            #loss = torch.mean(output)  # Example loss, you should replace it with your loss
            loss = nn.MSELoss()
            loss = loss(output, y_train)



            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            end_time = time.time()
            epoch_time = end_time - start_time

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Time: {epoch_time:.2f}s")



    def calculate_LR(self, X_train, y_train): #FUNZIONA
        #input_size = X_train.shape[1]
        #model = PyTorchMLP(input_size=input_size)

        # Convert to pytorch
        #X_train_torch = torch.from_numpy(X_train.astype(np.float32))
        #y_train_torch = torch.from_numpy(y_train.astype(np.float32))

        #criterion = nn.MSELoss()
        #optimizer = optim.Adam(model.parameters(), lr=0.01)
        ''' #already trained in train _model
        # Training loop
        model.train()
        for epoch in range(100):  # Assuming a fixed number of epochs
            #print("ccc")
            optimizer.zero_grad()
            outputs = model(X_train_torch)
            loss = criterion(outputs, y_train_torch.view(-1, 1))
            loss.backward()
            optimizer.step()
        #print("qr")
        '''
        # Making predictions (inference)
        self.model.eval()

        device = next(self.model.parameters()).device
        X_train = X_train.to(device)
        y_train = y_train.to(device)


        with torch.no_grad():
            predictions = self.model(X_train)

        # Convert back to numpy
        #y_pred = predictions.numpy()
        y_pred = predictions
        #y_train = y_train.numpy()

        # Compute the error
        y_err = y_pred - y_train
        y_err = y_err.cpu().numpy()

        return y_err   #returning the err for all the graphs in one ACTOR_CRITIC Iteration


    '''
    def calculate_QR(self, X_train, y_train):
        X_train = self.poly.fit_transform(X_train)[:,1:]
        return self.calculate_LR(X_train, y_train)
    '''
    def calculate_GPR(self, X_train, y_train):
        med_w = np.median(pdist(X_train, 'euclidean'))
        gpr = GPR().fit(X_train/med_w, y_train)
        return y_train.reshape(-1,1) - gpr.predict(X_train/med_w).reshape(-1,1)

    ####### score calculations

    def calculate_reward_single_graph(self, graph_batch, lambda1, lambda2):
        graph_to_int = []
        graph_to_int2 = []
        #print("graph_batch====")
        #print(graph_batch)

        for i in range(self.maxlen):
            #graph_batch[i,i].assign(0)

            #a = tf.Variable([0], shape=tf.TensorShape(None))
            #tf.compat.v1.assign(a, graph_batch[i, i])
            #graph_batch = tf.tensor_scatter_nd_update(graph_batch, [[i,i]], [0])
            updates = tf.constant([0], dtype=tf.int32)
            graph_batch = tf.tensor_scatter_nd_update(graph_batch, [[i, i]], updates)

            #graph_batch[i][i] = 0  #correction
            tt = np.int32(graph_batch[i])
            #print("tt", tt)
            graph_to_int.append(self.baseint * i + int(''.join([str(ad) for ad in tt]), 2)) # convert binary (taken from the row as a string) to int plus (2**d) *i
            graph_to_int2.append(int(''.join([str(ad) for ad in tt]), 2)) # only the binary transformation
            # print(graph_to_int, graph_to_int2)
            #print("graph_to_int: ", graph_to_int)
            #print("graph_to_int2: ", graph_to_int2)
            #break


        graph_batch_to_tuple = tuple(graph_to_int2) # contains the rows of the graph after the transformation from binary to int

        if graph_batch_to_tuple in self.d: # if the graph has already been calculated, return the stored value
            score_cyc = self.d[graph_batch_to_tuple]
            return self.penalized_score(score_cyc, lambda1, lambda2), score_cyc[0], score_cyc[1]

        RSS_ls = []

        for i in range(self.maxlen):
            col = graph_batch[i]  # take the i-th row of the graph and store it in col
            if graph_to_int[i] in self.d_RSS: # if the RSS for the i-th row has already been calculated, store it in RSS_ls and continue
                RSS_ls.append(self.d_RSS[graph_to_int[i]])
                continue

            # no parents, then simply use mean
            if np.sum(col) < 0.1: 
                y_err = self.inputdata[:, i]
                y_err = y_err - np.mean(y_err)

            else:
                cols_TrueFalse = tf.greater(tf.cast(col, tf.float32), tf.constant(0.5)) # set to True the elements of col that are greater than 0.5
                X_train = self.inputdata[:, cols_TrueFalse] # take the columns of the input data that are True in cols_TrueFalse
                y_train = self.inputdata[:, i] # take the i-th column of the input data
                #y_err = self.calculate_yerr(X_train, y_train)
                y_err = self.calculate_yerr(self.X_train_torch, self.y_train_torch)

            RSSi = np.sum(np.square(y_err))

            # if the regresors include the true parents, GPR would result in very samll values, e.g., 10^-13
            # so we add 1.0, which does not affect the monotoniticy of the score
            if self.reg_type == 'GPR':
                RSSi += 1.0

            RSS_ls.append(RSSi)
            self.d_RSS[graph_to_int[i]] = RSSi


        if self.score_type == 'BIC':
            BIC = np.log(np.sum(RSS_ls)/self.n_samples+1e-8) \
                  + np.sum(graph_batch)*self.bic_penalty/self.maxlen 
        elif self.score_type == 'BIC_different_var':
            BIC = np.sum(np.log(np.array(RSS_ls)/self.n_samples+1e-8)) \
                 + np.sum(graph_batch)*self.bic_penalty

        score = self.score_transform(BIC)
        cycness = np.trace(matrix_exponential(np.array(graph_batch)))- self.maxlen # trace is the sum of the diagonal elements of matrix_exponential(np.array(graph_batch))
        reward = score + lambda1*np.float32(cycness>1e-5) + lambda2*cycness
        '''
        This line calculates the final reward for the graph. The reward is a combination of the transformed BIC score (score), a penalty or reward for the presence of cycles (cycness), and regularization terms controlled by lambda1 and lambda2.
        lambda1*np.float(cycness>1e-5): This term adds a penalty or reward based on the presence of significant cycles in the graph. If cycness is greater than a threshold (1e-5), indicating the presence of cycles, lambda1 is added to the reward (this term enables or disables a fixed impact on the reward based on cycle presence)
        + lambda2*cycness: Adds a term to the reward that is linearly proportional to cycness, scaled by lambda2. This allows the amount of cyclicality in the graph to have a direct, variable impact on the reward, with lambda2 adjusting the sensitivity of the reward to the presence and number of cycles.
        '''



        if self.l1_graph_reg > 0:
            reward = reward + self.l1_grapha_reg * np.sum(graph_batch)
            score = score + self.l1_grapha_reg * np.sum(graph_batch)

        self.d[graph_batch_to_tuple] = (score, cycness)

        if self.verbose:
            self._logger.info('BIC: {}, cycness: {}, returned reward: {}'.format(BIC, cycness, final_score))

        # print_mine(graph_batch, "YYY", only_summary=True)


        # print("ENCODER embedded input ============")
        # print("XXX", reward, score, cycness)
        return reward, score, cycness

    #### helper
    
    def score_transform(self, s):
        return (s-self.sl)/(self.su-self.sl)*self.lambda1_upper

    def penalized_score(self, score_cyc, lambda1, lambda2):
        score, cyc = score_cyc
        return score + lambda1*np.float32(cyc>1e-5) + lambda2*cyc
    
    def update_scores(self, score_cycs, lambda1, lambda2):
        ls = []
        for score_cyc in score_cycs:
            ls.append(self.penalized_score(score_cyc, lambda1, lambda2))
        return ls
    
    def update_all_scores(self, lambda1, lambda2):
        score_cycs = list(self.d.items())
        ls = []
        for graph_int, score_cyc in score_cycs:
            ls.append((graph_int, (self.penalized_score(score_cyc, lambda1, lambda2), score_cyc[0], score_cyc[1])))
        return sorted(ls, key=lambda x: x[1][0])

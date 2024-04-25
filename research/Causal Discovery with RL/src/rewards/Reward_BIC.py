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

class PyTorchMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PyTorchMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fcl = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        # x = torch.nn.functional.leaky_relu(self.fc2(x))
        # x = torch.nn.functional.leaky_relu(self.fc3(x))

        x = self.fcl(x)
        return x
'''
class PyTorchMLP(nn.Module):
    def __init__(self, input_size):
        super(PyTorchMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 1)  

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
'''
'''
class PyTorchMLP(nn.Module):
    def __init__(self, input_size):
        super(PyTorchMLP, self).__init__()

        mid = int(input_size*1.5)
        self.layer1 = nn.Linear(input_size, mid)
        self.layer3 = nn.Linear(mid, 1)


        # self.relu = nn.ReLU()
    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        # x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
'''


'''class PyTorchMLP(nn.Module):
    def __init__(self, input_size):
        super(PyTorchMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)
        self.layer3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.relu3(self.layer3(x))
        x = self.layer4(x)
        return x

'''



class get_Reward(object):

    _logger = logging.getLogger(__name__)

    def __init__(self, batch_num, maxlen, dim, inputdata, sl, su, lambda1_upper, 
                 score_type='BIC', reg_type='LR', l1_graph_reg=0.0, verbose_flag=True):
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
        #print (graphs.shape)

        for graphi in graphs:
            #print (graphi.shape)
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


    def plot_weights_histogram(self, model, epoch, output_dir):
        for name, param in model.named_parameters():
            if 'weight' in name:
                plt.figure(figsize=(9, 6))
                weights = param.detach().cpu().numpy()
                plt.hist(weights.flatten(), bins=100)
                plt.title(f'Weights Histogram - {name} - Epoch {epoch}')
                plt.xlabel('Weight Values')
                plt.ylabel('Frequency')
                plt.grid(True)


                os.makedirs(output_dir, exist_ok=True)
                filename = f'weights_histogram_{name}_epoch_{epoch}.png'
                file_path = os.path.join(output_dir, filename)

                plt.savefig(file_path)
                plt.close()


    '''
    def calculate_QR(self, X_train, y_train):  #should be called NLR
        input_size = X_train.shape[1]

        the_device = lambda x: x.cuda()
        model = the_device(PyTorchMLP(input_size=input_size))

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)


        # Convert X_train and y_train from NumPy arrays to PyTorch tensors
        X_train_torch = torch.from_numpy(X_train.astype(np.float32))
        y_train_torch = torch.from_numpy(y_train.astype(np.float32))

        X_train_torch = the_device(X_train_torch)
        y_train_torch = the_device(y_train_torch)

        X_train_train, X_test, y_train_train, y_test = train_test_split(X_train_torch, y_train_torch, test_size=0.2, random_state=42)


        loss_function = nn.MSELoss()  # MSE error
        #optimizer = optim.Adam(model.parameters(), lr=0.01)
        optimizer = optim.Adam(model.parameters(), lr=0.01,
                               weight_decay=1e-2)  # with L2 regularizzation

        weight_history = []  # Store weights
        epochs = 100
        # Training loop
        #t1 = time.time()
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_train)
            loss = loss_function(outputs, y_train_train.view(-1, 1))
            loss.backward()
            optimizer.step()

            #if epoch % 100 == 0:
                #output_dir = '/Users/saraz/PycharmProjects/tf2/trustworthyAI/research/Causal Discovery with RL/src/output/weights'
                #self.plot_weights_histogram(model, epoch, output_dir)

        #t2 = time.time()
        #print("PyTorch learning took", t2-t1, "seconds")
        # Making predictions (inference)
        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            #test_loss = loss_function(predictions, y_train_torch.view(-1, 1))

        #print(f"Final Test Loss: {test_loss.item()}")
        #return model, predictions, test_loss



        # Convert predictions back to a NumPy array
        y_pred = predictions.cpu().numpy()
        y_test = y_test.cpu().numpy()

        # Compute the error
        y_err = y_pred.flatten() - y_test

        return y_err
    '''


    '''
    def calculate_QR(self, X_train, y_train):  ##SVM (SVR)
        
        svr_model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.2))

        # Fit the SVR model to the scaled training data 
        svr_model.fit(X_train, y_train)

        # Making predictions on the training data 
        predictions = svr_model.predict(X_train)

        # Compute the error 
        y_err = predictions - y_train

        return y_err
    '''
    '''
    def calculate_QR(self, X_train, y_train):   #FUNZIONA!!!!
        
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_train_poly = poly.fit_transform(X_train)

        
        input_size = X_train_poly.shape[1]
        model = PyTorchMLP(input_size=input_size)

        #convet to torch 
        X_train_poly_torch = torch.from_numpy(X_train_poly.astype(np.float32))
        y_train_torch = torch.from_numpy(y_train.astype(np.float32))

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        model.train()
        for epoch in range(100):  
            optimizer.zero_grad()
            outputs = model(X_train_poly_torch)
            loss = criterion(outputs, y_train_torch.view(-1, 1))
            loss.backward()
            optimizer.step()

        # Making predictions (inference)
        model.eval()
        with torch.no_grad():
            predictions = model(X_train_poly_torch)

        # Convert predictions back to numpy
        y_pred = predictions.numpy()

        # Compute error
        y_err = y_pred.flatten() - y_train  # Ensure shapes are compatible

        return y_err
    '''


    def calculate_GPR(self, X_train, y_train): #Working for linear case : calculate_LR ()
        input_size = X_train.shape[1]
        model = PyTorchMLP(input_size=input_size, hidden_size=100, output_size=1)

        # Convert to pytorch
        X_train = torch.from_numpy(X_train.astype(np.float32))
        y_train = torch.from_numpy(y_train.astype(np.float32))
        y_train = y_train.unsqueeze(1)  #going ftom 1-D to 2-D
        #print("X_train", X_train)

        #print (y_train)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)#, weight_decay=1e-5)

        # Training loop
        #model.train()
        num_epochs = 1000
        for epoch in range(num_epochs):
            #print("ccc")

            outputs = model(X_train)
            loss = criterion(outputs, y_train)#_torch.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if (epoch + 1) % 100 == 0:
                # print(outputs)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        #print("qr")

        # Making predictions (inference)
        model.eval()
        with torch.no_grad():
            predictions = model(X_train)

        # Convert back to numpy
        y_pred = predictions.numpy()

        # Compute the error
        #y_err = y_pred.flatten() - y_train
        y_train = y_train.numpy()
        y_err = y_pred - y_train

        return y_err

    '''
    def calculate_LR(self, X_train, y_train):
        # Define the MLP model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(X_train.shape[1],)),  # Hidden layer without activation function
            tf.keras.layers.Dense(1)  # Output layer without activation function for linear regression
        ])

        
        model.compile(optimizer='adam', loss='mse')

        X_train, y_train = tf.convert_to_tensor(X_train, dtype=tf.float32), tf.convert_to_tensor(y_train,
                                                                                                 dtype=tf.float32)
        model.fit(X_train, y_train, epochs=10, verbose=0)

        
        #y_pred = model(X_train, training=False)  #inference mode
        y_pred = model(tf.stop_gradient(X_train), training=False )
        # Flatten the output to match y_train's shape
        #y_pred = tf.reshape(y_pred, [-1])
        y_pred = model.predict(X_train)

        # Compute the error
        y_err = y_pred - y_train


        return y_err



    '''

    '''
    def calculate_LR(self, X_train, y_train):
        # Define the MLP model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(1)
        ])

        # Instead of model.fit(), manually compute forward pass and loss
        with tf.GradientTape() as tape:
            y_pred = model(X_train, training=True)
            loss = tf.reduce_mean(tf.square(y_pred - y_train))

        # Compute gradients and apply them
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.optimizers.Adam()
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # After training, compute error
        y_pred = model.predict(X_train).flatten()
        y_err = y_pred - y_train
        return y_err

    '''

    '''
    def calculate_QR(self, X_train, y_train):
        X_train = self.poly.fit_transform(X_train)[:,1:]
        return self.calculate_LR(X_train, y_train)
    '''
    '''
    def calculate_GPR(self, X_train, y_train):
        med_w = np.median(pdist(X_train, 'euclidean'))
        gpr = GPR().fit(X_train/med_w, y_train)

        return y_train.reshape(-1, 1) - gpr.predict(X_train/med_w).reshape(-1,1)
    '''
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

        ##for the pgr


        #graph_batch = tf.transpose(graph_batch)   #TRANSPOSE IT?


        for i in range(self.maxlen):
            col = graph_batch[i]  # take the i-th row of the graph and store it in col

            #print ("COLONNA", col)

            if graph_to_int[i] in self.d_RSS: # if the RSS for the i-th row has already been calculated, store it in RSS_ls and continue
                RSS_ls.append(self.d_RSS[graph_to_int[i]])
                continue

            # no parents, then simply use mean
            if np.sum(col) < 0.1: 
                y_err = self.inputdata[:, i]
                y_err = y_err - np.mean(y_err)

            else:
                #print ("i", i)
                cols_TrueFalse = tf.greater(tf.cast(col, tf.float32), tf.constant(0.5)) # set to True the elements of col that are greater than 0.5

                #print(cols_TrueFalse)#cols_TrueFalse = col > 0.5

                X_train = self.inputdata[:, cols_TrueFalse] # take the columns of the input data that are True in cols_TrueFalse
                #print ("X_train.shape", X_train.shape)
                #print ("X_train", X_train)
                y_train = self.inputdata[:, i] # take the i-th column of the input data
                y_err = self.calculate_yerr(X_train, y_train)



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

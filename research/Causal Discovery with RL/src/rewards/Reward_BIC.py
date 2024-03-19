import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.linalg import expm as matrix_exponential
from scipy.spatial.distance import pdist, squareform
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.preprocessing import PolynomialFeatures
import logging
import tensorflow as tf

from helpers.debugger import print_mine, print_mine_np


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
    def calculate_LR(self, X_train, y_train):
        X = np.hstack((X_train, self.ones))
        XtX = X.T.dot(X)
        Xty = X.T.dot(y_train)
        theta = np.linalg.solve(XtX, Xty)
        y_err = X.dot(theta) - y_train
        return y_err

    def calculate_QR(self, X_train, y_train):
        X_train = self.poly.fit_transform(X_train)[:,1:]
        return self.calculate_LR(X_train, y_train)
    
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
            graph_to_int.append(self.baseint * i + np.int(''.join([str(ad) for ad in tt]), 2)) # convert binary (taken from the row as a string) to int plus (2**d) *i
            graph_to_int2.append(np.int(''.join([str(ad) for ad in tt]), 2)) # only the binary transformation
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
        reward = score + lambda1*np.float(cycness>1e-5) + lambda2*cycness 
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
        return score + lambda1*np.float(cyc>1e-5) + lambda2*cyc
    
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

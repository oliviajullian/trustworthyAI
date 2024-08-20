import keras.layers
import tensorflow as tf
import numpy as np
#from tqdm import tqdm


class Critic(keras.layers.Layer):
 
 
    def __init__(self, config, is_train):
        super().__init__()

        self.config=config

        # Data config
        self.batch_size = config.batch_size 
        self.max_length = config.max_length 
        self.input_dimension = config.input_dimension 

        # Network config
        self.input_embed = config.hidden_dim 
        self.num_neurons = config.hidden_dim 
        # REMOVED FOR MIGRATION TO TF2: self.initializer = tf.contrib.layers.xavier_initializer()
        self.initializer = tf.keras.initializers.GlorotUniform(seed=42)


        # Baseline setup
        self.init_baseline = 0.

        self.h0 = tf.keras.layers.Dense(self.num_neurons, activation=tf.nn.relu, kernel_initializer=self.initializer)

        self.w1  = self.add_weight(name = "w1", shape= [self.num_neurons, 1], initializer=self.initializer, dtype=tf.float32)
        self.b1  = self.add_weight(name = "b1",initializer=self.initializer, dtype=tf.float32)   # shuld be self.init_baseline?

        self.lr2_start = config.lr1_start  # initial learning rate
        self.lr2_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr2_decay_step = config.lr1_decay_step  # learning rate decay step

        self.lr2 = tf.keras.optimizers.schedules.ExponentialDecay(self.lr2_start, self.lr2_decay_step,
                                              self.lr2_decay_rate, staircase=False, name="learning_rate1")
        # Optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr2, beta_1=0.9, beta_2=0.99, epsilon=0.0000001)



 
    def predict_rewards(self, encoder_output):
        # [Batch size, Sequence Length, Num_neurons] to [Batch size, Num_neurons]
        frame = tf.reduce_mean(encoder_output, 1)  

        # REMOVED FOR MIGRATION TO TF2
        # with tf.variable_scope("ffn"):
        # ffn 1
        x = frame

        h0 = x = self.h0(frame)


        # ffn 2
        # w1 =tf.get_variable("w1", [self.num_neurons, 1], initializer=self.initializer)
        # b1 = tf.Variable(self.init_baseline, name="b1")

        self.predictions = tf.squeeze(tf.matmul(h0, self.w1)+self.b1) # remove dimensions of shape 1

        return self.predictions

    def compute_losses(self, input_, reward_, graphs_, step, actor):

        # self.reward_baseline = tf.stop_gradient(
        #     reward_ - actor.avg_baseline - self.predictions)  # [Batch size, 1]

        weights_ = 1.0  # weights_ = tf.exp(self.log_softmax-tf.reduce_max(self.log_softmax)) # probs / max_prob
        # self.loss2 = tf.losses.mean_squared_error(reward_ - self.avg_baseline, self.critic.predictions,
        #                                           weights=weights_)
        self.loss2 = tf.compat.v1.losses.mean_squared_error(reward_ - actor.avg_baseline, self.predictions) # measure difference between actual returns (adjusted for baseline) and predictions of the critic
        tf.summary.scalar('loss2', self.loss2)

        # print("XYPS LOSS 2", self.loss2)
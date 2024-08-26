# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Adapted from kyubyong park, June 2017.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import keras.layers
import tensorflow as tf

from helpers.debugger import print_mine


# Apply multihead attention to a 3d tensor with shape [batch_size, seq_length, n_hidden].
# Attention size = n_hidden should be a multiple of num_head
# Returns a 3d tensor with shape of [batch_size, seq_length, n_hidden]
 
def multihead_attention(layers, inputs, num_heads=16, dropout_rate=0.1, is_training=True):
    # Linear projections
    Q, K, V, BN = layers['Q'], layers['K'], layers['V'], layers['BN']

    # Split and concat
    Q_ = tf.concat(tf.split(Q(inputs), num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
    K_ = tf.concat(tf.split(K(inputs), num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
    V_ = tf.concat(tf.split(V(inputs), num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]

    # Multiplication
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # num_heads*[batch_size, seq_length, seq_length]

    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs) # num_heads*[batch_size, seq_length, seq_length]

    # Dropouts
    # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

    # Weighted sum
    outputs = tf.matmul(outputs, V_) # num_heads*[batch_size, seq_length, n_hidden/num_heads]

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # [batch_size, seq_length, n_hidden]

    # Residual connection
    outputs += inputs # [batch_size, seq_length, n_hidden]

    # Normalize
    # outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
    outputs = BN(outputs, training=True)

    return outputs
 
 
# Apply point-wise feed forward net to a 3d tensor with shape [batch_size, seq_length, n_hidden]
# Returns: a 3d tensor with the same shape and dtype as inputs
 
def feedforward(layers, inputs):
    conv1, conv2, bn2 = layers['conv1'], layers['conv2'], layers['BN2']

    # Inner layer
    outputs = conv1(inputs)
    outputs = conv1(outputs)
    # Residual connection
    outputs += inputs
    outputs = bn2(outputs, training=True)

    return outputs
 
 
class TransformerEncoder(keras.layers.Layer):
 
    def __init__(self, config, is_train):
        super().__init__()

        self.batch_size = config.batch_size # batch size
        self.max_length = config.max_length # input sequence length (number of cities)
        self.input_dimension = config.input_dimension # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token
 
        self.input_embed = config.hidden_dim # dimension of embedding space (actor)
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks

        # self.i1nitializer =tf.initializers.GlorotNormal()
        self.initializer = tf.initializers.GlorotUniform()
 
        self.is_training = is_train #not config.inference_mode

        self.W_embed  = self.add_weight(name="weights", shape= [1, self.input_dimension, self.input_embed], initializer=self.initializer, dtype=tf.float64)
        # tf.get_variable("weights", [1, self.input_dimension, self.input_embed], initializer=self.initializer)

        self.bn1 = tf.keras.layers.BatchNormalization(axis=2)

        self.the_layers_stack = []
        for i in range(self.num_stacks):  # num blocks
            Q = tf.keras.layers.Dense(self.input_embed, activation=tf.nn.relu)  # [batch_size, seq_length, n_hidden]
            K = tf.keras.layers.Dense(self.input_embed, activation=tf.nn.relu)  # [batch_size, seq_length, n_hidden]
            V = tf.keras.layers.Dense(self.input_embed, activation=tf.nn.relu)  # [batch_size, seq_length, n_hidden]

            BN = tf.keras.layers.BatchNormalization(axis=2)

            print([4*self.input_embed, self.input_embed])

            conv1 = tf.keras.layers.Conv1D(filters=self.input_embed, kernel_size=1, activation=tf.nn.relu, use_bias=True)
            conv2 = tf.keras.layers.Conv1D(filters=self.input_embed, kernel_size=1, activation=tf.nn.relu, use_bias=True)

            # Normalize
            BN2 = tf.keras.layers.BatchNormalization(axis=2)

            self.the_layers_stack += [{'Q': Q, 'K': K, 'V': V, 'BN': BN, 'conv1': conv1, 'conv2': conv2, 'BN2': BN2}]

    def encode(self, inputs):
      # REMOVED FOR MIGRATION TO TF2 with tf.variable_scope("embedding"):
      # Embed input sequence
      self.embedded_input = tf.nn.conv1d(inputs, self.W_embed, 1, "VALID", name="embedded_input")



      # Batch Normalization
      # self.enc = tf.layers.batch_normalization(self.embedded_input, axis=2, training=self.is_training, name='layer_norm', reuse=None)
      self.enc = self.bn1(self.embedded_input, training=True)

      # REMOVED FOR MIGRATION TO TF2 with tf.variable_scope("embedding"):
      # Blocks
      for i in range(self.num_stacks): # num blocks
          # REMOVED FOR MIGRATION TO TF2: with tf.variable_scope("block_{}".format(i)):
          # Multihead Attention
          self.enc = multihead_attention(self.the_layers_stack[i], self.enc, num_heads=self.num_heads, dropout_rate=0.0, is_training=self.is_training)

          # Feed Forward
          self.enc = feedforward(self.the_layers_stack[i], self.enc)

      # Return the output activations [Batch size, Sequence Length, Num_neurons] as tensors.
      self.encoder_output = self.enc ### NOTE: encoder_output is the ref for attention ###
      return self.encoder_output

 
 
 
 

import numpy as np
import tensorflow as tf


print(tf.__version__)

a = tf.constant([8,9,10])
b = tf.tensor_scatter_nd_update(a, [[0]], [1])

print(b)
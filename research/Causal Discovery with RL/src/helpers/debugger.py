import numpy as np
import tensorflow as tf


def print_mine(variable, message, only_summary=False):
    mean = float(tf.reduce_mean(tf.cast(variable, tf.float32)).numpy())
    min = float(tf.reduce_min(tf.cast(variable, tf.float32)).numpy())
    max = float(tf.reduce_max(variable).numpy())

    print(message, min, mean, max, variable.numpy() if not only_summary else '')



def print_mine_np(variable, message, only_summary=False):
    mean = np.mean(variable)
    min = np.min(variable)
    max = np.max(variable)

    print(message, min, mean, max, variable.numpy() if not only_summary else '')
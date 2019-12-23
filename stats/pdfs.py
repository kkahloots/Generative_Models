import tensorflow as tf
import numpy as np

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

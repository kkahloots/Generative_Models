import tensorflow as tf
import tensorflow_probability as tfp

def log_normal_pdf(sample, mean, logvar):
    dist = tfp.distributions.Normal(loc=mean, scale=tf.sqrt(tf.exp(logvar)))
    return tf.reduce_sum(dist.log_prob(value=sample))

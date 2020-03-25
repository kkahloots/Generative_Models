import tensorflow as tf
import tensorflow_probability as tfp

def log_normal_pdf(sample, mean, logvariance):
    dist = tfp.distributions.Normal(loc=mean, scale=tf.sqrt(tf.exp(logvariance)))
    return tf.add(x=dist.log_prob(value=sample), y=0.0, name='logpdf')

import tensorflow as tf
import tensorflow_probability as tfp

def log_normal_pdf(sample, mean, logvariance):
    dist = tfp.distributions.Normal(loc=mean, scale=tf.sqrt(tf.exp(logvariance)))
    return tf.identity(dist.log_prob(value=sample), name='logpdf')

def log_Student_df1_pdf(sample, mean, logvariance):
    dist = tfp.distributions.StudentT(df=1, loc=mean, scale=tf.sqrt(tf.exp(logvariance)))
    return tf.identity(dist.log_prob(value=sample), name='logpdf')

def log_Student_df05_pdf(sample, mean, logvariance):
    dist = tfp.distributions.StudentT(df=0.5, loc=mean, scale=tf.sqrt(tf.exp(logvariance)))
    return tf.identity(dist.log_prob(value=sample), name='logpdf')
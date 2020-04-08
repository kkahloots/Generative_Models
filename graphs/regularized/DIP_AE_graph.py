
from graphs.basics.AE_graph import cross_entropy
import tensorflow as tf

def create_DIP_losses():
    return {'x_logits': cross_entropy,
            'covariance_regularized': covariance_regularized
            }

def create_DIP_Bayesian_losses():
    return {'x_logits': cross_entropy,
            'covariance_regularized': covariance_regularized,
            'bayesian_divergent': bayesian_divergence
            }

def covariance_regularized(inputs, covariance_regularizer):
    return covariance_regularizer

def bayesian_divergence(inputs, bayesian_divergent):
    return -tf.reduce_sum(bayesian_divergent)

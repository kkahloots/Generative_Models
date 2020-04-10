
from graphs.basics.AE_graph import cross_entropy
import tensorflow as tf

def create_regularized_losses():
    return {'x_logits': cross_entropy,
            'covariance_regularized': covariance_regularized
            }

def create_Bayesian_losses():
    return {'x_logits': cross_entropy,
            'bayesian_divergent': bayesian_divergence
            }

def create_regularized_Bayesian_losses():
    return {'x_logits': cross_entropy,
            'covariance_regularized': covariance_regularized,
            'bayesian_divergent': bayesian_divergence
            }

def covariance_regularized(inputs, covariance_regularizer):
    return covariance_regularizer

def bayesian_divergence(inputs, bayesian_divergent):
    return bayesian_divergent

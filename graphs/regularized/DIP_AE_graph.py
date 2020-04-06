
import tensorflow as tf
from graphs.basics.AE_graph import cross_entropy

def create_losses():
    return {'x_logits': cross_entropy,
            'covariance_regularized': covariance_regularized
            }

def covariance_regularized(inputs, covariance_regularizer):
    return covariance_regularizer

'''
------------------------------------------------------------------------------
                                     DIP_Covarance OPERATIONS
------------------------------------------------------------------------------
'''

import tensorflow as tf

def regularize(latent_mean, regularize=True, lambda_d=100, lambda_od=50):
    cov_latent_mean = compute_covariance(latent_mean)

    # Eq 6 page 4
    # mu = z_mean is [batch_size, num_latent]
    # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
    if regularize:
        cov_dip_regularizer = regularize_diag_off_diag_dip(cov_latent_mean, lambda_d, lambda_od)
        cov_dip_regularizer = tf.identity(cov_dip_regularizer, name='covariance_regularized')
        return cov_latent_mean, cov_dip_regularizer
    else:
        return cov_latent_mean


def gaussian_regularize(latent_mean, latent_logvariance, regularize=True, lambda_d=100, lambda_od=50):
    cov_enc = tf.linalg.diag(tf.exp(latent_logvariance))
    cov_latent_sigma = tf.reduce_mean(cov_enc, axis=0)
    cov_latent_mean = compute_covariance(latent_mean)

    cov_latent = cov_latent_sigma + cov_latent_mean

    # Eq 6 page 4
    # mu = z_mean is [batch_size, num_latent]
    # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
    if regularize:
        cov_dip_regularizer = regularize_diag_off_diag_dip(cov_latent, lambda_d, lambda_od)
        cov_dip_regularizer = tf.identity(cov_dip_regularizer, name='covariance_regularized')
        return cov_latent_mean, cov_dip_regularizer
    else:
        return cov_latent_mean


def compute_covariance(latent_mean):
    """
    :param latent_mean:
    :return:
    Computes the covariance_regularizer of latent_mean.
    Uses cov(latent_mean) = E[latent_mean*latent_mean^T] - E[latent_mean]E[latent_mean]^T.
    Args:
      latent_mean: Encoder mean, tensor of size [batch_size, num_latent].
    Returns:
      cov_latent_mean: Covariance of encoder mean, tensor of size [latent_dim, latent_dim].
    """
    exp_latent_mean_latent_mean_t = tf.reduce_mean(
        tf.expand_dims(latent_mean, 2) * tf.expand_dims(latent_mean, 1), axis=0)
    expectation_latent_mean = tf.reduce_mean(latent_mean, axis=0)

    cov_latent_mean = tf.subtract(exp_latent_mean_latent_mean_t,
                                  tf.expand_dims(expectation_latent_mean, 1) * tf.expand_dims(expectation_latent_mean,
                                                                                              0),
                                  name='covariance_mean')
    return cov_latent_mean


def regularize_diag_off_diag_dip(covariance_matrix, lambda_od, lambda_d):
    """
    Compute on and off diagonal covariance_regularizer for DIP_Covarance-VAE models.
    Penalize deviations of covariance_matrix from the identity matrix. Uses
    different weights for the deviations of the diagonal and off diagonal entries.
    Args:
        covariance_matrix: Tensor of size [num_latent, num_latent] to covar_reg.
        lambda_od: Weight of penalty for off diagonal elements.
        lambda_d: Weight of penalty for diagonal elements.
    Returns:
        dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
    """
    # matrix_diag_part
    covariance_matrix_diagonal = tf.linalg.diag_part(covariance_matrix)
    covariance_matrix_off_diagonal = covariance_matrix - tf.linalg.diag(covariance_matrix_diagonal)
    dip_regularizer = tf.add(
        lambda_od * covariance_matrix_off_diagonal ** 2,
        lambda_d * (covariance_matrix_diagonal - 1) ** 2)
    return dip_regularizer

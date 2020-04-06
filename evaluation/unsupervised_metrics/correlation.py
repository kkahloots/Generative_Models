import scipy
import numpy as np

def gaussian_total_correlation(cov):
    """
    Computes the total correlation of a Gaussian with covariance_regularizer matrix cov.
    We use that the total correlation is the KL divergence between the Gaussian
    and the product of its marginals. By design, the means of these two Gaussians
    are zero and the covariance_regularizer matrix of the second Gaussian is equal to the
    covariance_regularizer matrix of the first Gaussian with off-diagonal entries set to zero.
    Args:
        cov: Numpy array with covariance_regularizer matrix.
    Returns:
          Scalar with total correlation.
    """
    return 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])

def gaussian_wasserstein_correlation(cov):
    """Wasserstein L2 distance between Gaussian and the product of its marginals.
    Args:
      cov: Numpy array with covariance_regularizer matrix.
    Returns:
      Scalar with score.
    """
    sqrtm = scipy.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
    return 2 * np.trace(cov) - 2 * np.trace(sqrtm)
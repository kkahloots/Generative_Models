import numpy as np
import logging
from evaluation.unsupervised_metrics.correlation import gaussian_total_correlation, gaussian_wasserstein_correlation
from evaluation.unsupervised_metrics.mutual_info import discrete_mutual_info, discrete_entropy
from evaluation.shared import generate_batch_factor_code

def unsupervised_metrics(
        ground_truth_data,
        representation_fn,
        random_state,
        num_train,
        batch_size=16):
    """Computes unsupervised scores based on covariance and mutual information.
       Args:
            ground_truth_data: GroundTruthData to be sampled from.
            representation_fn: Function that takes observations as input and
              outputs a dim_representation sized representation for each observation.
            random_state: Numpy random state used for randomness.
            artifact_dir: Optional path to directory where artifacts can be saved.
            num_train: Number of points used for training.
            batch_size: Batch size for sampling.
       Returns:
          Dictionary with scores.
    """
    scores = {}
    mus_train, _ = generate_batch_factor_code(
            ground_truth_data,
            representation_fn,
            num_train,
            random_state,
            batch_size
    )
    num_codes = mus_train.shape[0]
    cov_mus = np.cov(mus_train)
    assert num_codes == cov_mus.shape[0]

    # Gaussian total correlation.
    scores["gaussian_total_correlation"] = gaussian_total_correlation(cov_mus)

    # Gaussian Wasserstein correlation.
    scores["gaussian_wasserstein_correlation"] = gaussian_wasserstein_correlation(cov_mus)
    scores["gaussian_wasserstein_correlation_norm"] = (scores["gaussian_wasserstein_correlation"] / np.sum(np.diag(cov_mus)))

    # Compute average mutual information between different factors.
    mus_discrete = discrete_entropy(mus_train).reshape(-1,1)
    mutual_info_matrix = discrete_mutual_info(mus_discrete, mus_discrete)
    np.fill_diagonal(mutual_info_matrix, 0)
    mutual_info_score = np.sum(mutual_info_matrix) / (num_codes**2 - num_codes)
    scores["mutual_info_score"] = mutual_info_score
    return scores

import numpy as np

def generate_batch_factor_code(
        ground_truth_data,
        representation_function,
        num_points,
        random_state,
        batch_size
):
    """
    Sample a single training sample based on a mini-batch of ground-truth data.
        Args:
            ground_truth_data: GroundTruthData to be sampled from.
            representation_function: Function that takes observation as input and
            outputs a representation.
            num_points: Number of points to sample.
            random_state: Numpy random state used for randomness.
            batch_size: Batchsize to sample points.
            Returns:
            representations: Codes (num_codes, num_points)-np array.
            factors: Factors generating the codes (num_factors, num_points)-np array.
    """
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = \
            ground_truth_data.sample(num_points_iter, random_state)
        if i == 0:
            factors = current_factors
            representations = representation_function(current_observations)
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations,
                                       representation_function(
                                           current_observations)))
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)
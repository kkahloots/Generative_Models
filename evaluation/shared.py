import numpy as np
import tensorflow as tf

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
        #num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = \
            ground_truth_data.sample(batch_size, random_state)
        if i == 0:
            factors = current_factors
            representations = representation_function(current_observations)
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations,
                                       representation_function(
                                           current_observations)))
        i += batch_size
    representations = representations[:num_points]
    factors = factors[:num_points]
    return np.transpose(representations), np.transpose(factors)


def log10(t):
    """
    Calculates the base-10 log of each element in t.
    @param t: The tensor from which to calculate the base-10 log.
    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = tf.math.log(t)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
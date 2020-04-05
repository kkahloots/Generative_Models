
import logging
from utils.reporting.logging import log_message
from evaluation.supervised_metrics.completeness import completeness
from evaluation.supervised_metrics.disentanglement import disentanglement
from evaluation.supervised_metrics.informativeness import compute_importance_gbt
from evaluation.supervised_metrics.separated_attribute_predictability import compute_sap
from evaluation.shared import generate_batch_factor_code

def supervised_metrics(
        ground_truth_data,
        representation_fn,
        random_state,
        num_train,
        num_test,
        continuous_factors,
        batch_size=16
):
    """
    Computes the DCI scores according to Sec 2.
          Args:
                ground_truth_data: GroundTruthData to be sampled from.
                representation_fn: Function that takes observations as input and
                  outputs a dim_representation sized representation for each observation.
                random_state: Numpy random state used for randomness.
                artifact_dir: Optional path to directory where artifacts can be saved.
                num_train: Number of points used for training.
                num_test: Number of points used for testing.
                batch_size: Batch size for sampling.
          Returns:
                Dictionary with average disentanglement score, supervised_metrics and
                  informativeness (train and test).
    """

    # mus_train are of shape [num_codes, num_train], while ys_train are of shape
    # [num_factors, num_train].
    mus_train, ys_train = generate_batch_factor_code(
        ground_truth_data,
        representation_fn,
        num_train,
        random_state,
        batch_size
    )
    assert mus_train.shape[1] == num_train
    assert ys_train.shape[1] == num_train
    mus_test, ys_test = generate_batch_factor_code(
        ground_truth_data,
        representation_fn,
        num_test,
        random_state,
        batch_size
    )

    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(
        mus_train,
        ys_train,
        mus_test,
        ys_test
    )
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    scores["sap"] = compute_sap(mus_train, ys_train, mus_test, ys_test, continuous_factors)
    return scores


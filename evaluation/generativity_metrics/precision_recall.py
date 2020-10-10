from evaluation.generativity_metrics.shared_api import mean_fn, sigma_fn, bootstrapping_additive, slerp
import numpy as np
import time
epsilon=1e-4
from tensorflow.keras.applications import VGG16
import dask.delayed as delayed

def batch_pairwise_distances(U, V):
    """ Compute pairwise distances between two batches of feature vectors."""
    # Squared norms of each row in U and V.
    norm_u = np.sum(np.square(U), 1)
    norm_v = np.sum(np.square(V), 1)

    # norm_u as a row and norm_v as a column vectors.
    norm_u = np.reshape(norm_u, [-1, 1])
    norm_v = np.reshape(norm_v, [1, -1])

    # Pairwise squared Euclidean distances.
    D = np.maximum(norm_u - 2*np.matmul(U, V.numpy().T) + norm_v, 0.0)

    return D

class ManifoldEstimator():
    """Finds an estimate for the manifold of given feature vectors."""
    def __init__(self, features, row_batch_size, col_batch_size, nhood_sizes, clamp_to_percentile=None):
        """Find an estimate of the manifold of given feature vectors."""
        num_images = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features

        # Estimate manifold of features by calculating distances to kth nearest neighbor of each sample.
        self.D = np.zeros([num_images, self.num_nhoods], dtype=np.float16)
        distance_batch = np.zeros([row_batch_size, num_images], dtype=np.float16)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0:end1-begin1, begin2:end2] = batch_pairwise_distances(row_batch, col_batch)

            # Find the kth nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(distance_batch[0:end1-begin1, :], seq, axis=1)[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0  #max_distances  # 0

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        """Evaluate if new feature vectors are in the estimated manifold."""
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_images], dtype=np.float16)
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods], dtype=np.int32)

        realism_score = np.zeros([num_eval_images,], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_images,], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0:end1-begin1, begin2:end2] = batch_pairwise_distances(feature_batch, ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then the new sample lies on the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0:end1-begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

            nearest_indices[begin1:end1] = np.argmin(distance_batch[0:end1-begin1, :], axis=1)
            realism_score[begin1:end1] = self.D[nearest_indices[begin1:end1], 0] / np.min(distance_batch[0:end1-begin1, :], axis=1)

        if return_realism and return_neighbors:
            return batch_predictions, realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions



def knn_precision_recall_features(ref_features, eval_features, feature_net, nhood_sizes, row_batch_size, col_batch_size):
    """Calculates k-NN precision and recall for two sets of feature vectors."""
    state = dict()
    num_images = ref_features.shape[0]
    num_features = feature_net.output_shape[1]
    state['ref_features'] = ref_features
    state['eval_features'] = eval_features

    # Initialize DistanceBlock and ManifoldEstimators.
    state['ref_manifold'] = ManifoldEstimator(state['ref_features'], row_batch_size, col_batch_size, nhood_sizes)
    state['eval_manifold'] = ManifoldEstimator(state['eval_features'], row_batch_size, col_batch_size, nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    print('Evaluating k-NN precision and recall with %i samples...' % num_images)
    start = time.time()

    # Precision: How many points from eval_features are in ref_features manifold.
    state['precision'], state['realism_scores'], state['nearest_neighbors'] = state['ref_manifold'].evaluate(state['eval_features'], return_realism=True, return_neighbors=True)
    state['knn_precision'] = state['precision'].mean(axis=0)

    # Recall: How many points from ref_features are in eval_features manifold.
    state['recall'] = state['eval_manifold'].evaluate(state['ref_features'])
    state['knn_recall'] = state['recall'].mean(axis=0)

    elapsed_time = time.time() - start
    print('Done evaluation in: %gs' % elapsed_time)

    return state

def precision_score(model, data_generator, nhood_size=3, row_batch_size=10000, col_batch_size=10000,
                    tolerance_threshold=1e-6, max_iteration=200):
    # prepare the inception v3 model
    VGG16_model = VGG16(include_top=False, pooling='avg', input_shape=model.get_inputs_shape())
    VGG16_featues_fn = lambda x: VGG16_model(x)

    # prepare the ae model random_images_generator
    def model_random_images_generator():
        while True:
            data = next(data_generator)['images']
            ref_features = VGG16_featues_fn(data)

            # Generate latents from the data
            latents_real = model.encode(data)

            # Generate random latents and interpolation t-values.
            latents_t = np.random.normal(size=latents_real.shape)
            lerp_t = np.random.uniform(size=1)[0]

            latents_e = slerp(lerp_t+epsilon, latents_real, latents_t)
            images = model.decode(latents_e).numpy()
            images = (images * 255).astype(np.float32)

            eval_features = VGG16_featues_fn(images)

            # Calculate precision and recall.
            state = knn_precision_recall_features(ref_features=ref_features, eval_features=eval_features,
                                                  feature_net=VGG16_model,
                                                  nhood_sizes=[nhood_size], row_batch_size=row_batch_size,
                                                  col_batch_size=col_batch_size)

            knn_precision = state['knn_precision'][0]
            knn_recall = state['knn_recall'][0]
            yield knn_precision

    knn_precision_mean = bootstrapping_additive(
        data_generator=model_random_images_generator(), func=delayed(lambda x: x), \
        stopping_func=mean_fn, tolerance_threshold=tolerance_threshold, max_iteration=max_iteration
    )

    return knn_precision_mean.compute()

def recall_score(model, data_generator, nhood_size=3, row_batch_size=10000, col_batch_size=10000,
                    tolerance_threshold=1e-6, max_iteration=200):
    # prepare the inception v3 model
    VGG16_model = VGG16(include_top=False, pooling='avg', input_shape=model.get_inputs_shape())
    VGG16_featues_fn = lambda x: VGG16_model(x)

    # prepare the ae model random_images_generator
    def model_random_images_generator():
        while True:
            data = next(data_generator)['images']
            ref_features = VGG16_featues_fn(data)

            # Generate latents from the data
            latents_real = model.encode(data)

            # Generate random latents and interpolation t-values.
            latents_t = np.random.normal(size=latents_real.shape)
            lerp_t = np.random.uniform(size=1)[0]

            latents_e = slerp(lerp_t+epsilon, latents_real, latents_t)
            images = model.decode(latents_e).numpy()
            images = (images * 255).astype(np.float32)

            eval_features = VGG16_featues_fn(images)

            # Calculate precision and recall.
            state = knn_precision_recall_features(ref_features=ref_features, eval_features=eval_features,
                                                  feature_net=VGG16_model,
                                                  nhood_sizes=[nhood_size], row_batch_size=row_batch_size,
                                                  col_batch_size=col_batch_size)

            knn_precision = state['knn_precision'][0]
            knn_recall = state['knn_recall'][0]
            yield knn_recall

    knn_recall_mean = bootstrapping_additive(
        data_generator=model_random_images_generator(), func=delayed(lambda x: x), \
        stopping_func=mean_fn, tolerance_threshold=tolerance_threshold, max_iteration=max_iteration
    )

    return knn_recall_mean.compute()
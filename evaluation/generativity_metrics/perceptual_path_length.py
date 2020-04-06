import tensorflow as tf
from tensorflow.keras.applications import VGG16
from evaluation.generativity_metrics.shared_api import mean_fn, sigma_fn, bootstrapping_additive, slerp
import numpy as np

epsilon=1e-4
def perceptual_path_length_score(model, data_generator, tolerance_threshold=1e-6, max_iteration=100, batch_size=10):
    # prepare the inception v3 model
    VGG16_model = VGG16(include_top=False, pooling='avg', input_shape=model.get_inputs_shape())
    VGG16_featues_fn = lambda x: VGG16_model(x)
    epsilon = 1e-2

    def learned_perceptual_image_patch_similarity(images_a, images_b):
        """LPIPS metric using VGG-16 and Zhang weighting. (https://arxiv.org/abs/1801.03924)

        Takes reference images and corrupted images as an input and outputs the perceptual
        distance between the image pairs.
        """

        # Concatenate images.
        images = tf.concat([images_a, images_b], axis=0)

        # Extract features.
        vgg_features = VGG16_featues_fn(images)

        # Normalize each feature vector to unit length over channel dimension.
        normalized_features = []
        for x in vgg_features:
            x = tf.reshape(x, (len(x), 1))
            n = tf.reduce_sum(x ** 2, axis=1, keepdims=True) ** 0.5
            normalized_features.append(x / (n + 1e-10))

        # Split and compute distances.
        diff = [tf.subtract(*tf.split(x, 2, axis=0)) ** 2 for x in normalized_features]

        return np.array(diff)

    def filter_distances_fn(distances):
        # Reject outliers.
        lo = np.percentile(distances, 1, interpolation='lower')
        hi = np.percentile(distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= distances, distances <= hi), distances)
        return filtered_distances

    def calculate_distances(images):
        images01, images02 = images[0::2], images[1::2]
        return learned_perceptual_image_patch_similarity(images01, images02) * (1 / epsilon ** 2)

    # prepare the ae model random_images_generator
    def model_random_images_generator():
        while True:
            # Generate latents from the data
            latents_real = model.encode(next(data_generator))

            # Generate random latents and interpolation t-values.
            latents_t = np.random.normal(size=latents_real.shape)
            lerp_t = np.random.uniform(size=1)[0]

            latents_e = slerp(lerp_t, latents_real, latents_t)
            images = model.decode(latents_e).numpy()
            # images = (images*255).astype(np.uint8)

            yield images[:batch_size]
            # calculate_distances(images[0::2], images[1::2])

    def stopping_fn(distances):
        # Reject outliers.
        filter_distances = filter_distances_fn(distances)
        return np.mean(distances)

    ppl_mean = bootstrapping_additive(
        data_generator=model_random_images_generator(), func=calculate_distances, \
        stopping_func=stopping_fn, tolerance_threshold=tolerance_threshold, max_iteration=max_iteration
    )

    return ppl_mean
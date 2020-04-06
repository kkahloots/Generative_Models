from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from evaluation.generativity_metrics.shared_api import mean_fn, sigma_fn, bootstrapping_additive, slerp
import numpy as np
import scipy as sp
epsilon = 1e-6
import dask.array as da
import dask.delayed as delayed

def inception_score(model, tolerance_threshold=1e-6, max_iteration=100):
    @delayed
    def calculate_is(x):
        kl = x * (np.log(x+epsilon) - np.log(np.expand_dims(np.mean(x+epsilon, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        return np.exp(kl+epsilon)

    # prepare the inception v3 model
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=model.get_inputs_shape())
    inception_predictions = lambda x: inception_model.predict(inception_preprocess_input(x))

    # prepare the ae model random_images_generator
    def model_random_images_generator():
        while True:
            predictions = []
            for _ in range(500):
                data = model.generate_random_images().numpy()
                data = (data * 255).astype(np.uint8)
                predictions += [inception_predictions(data)]
            yield da.from_array(np.vstack(predictions), chunks=100)

    print('calculating the inception_score mean ...')
    is_mean = bootstrapping_additive(
        data_generator=model_random_images_generator(), func=calculate_is, stopping_func=mean_fn,
        tolerance_threshold=tolerance_threshold, max_iteration=max_iteration
    )

    print('calculating the inception_score sigma ...')
    is_sigma = bootstrapping_additive(
        data_generator=model_random_images_generator(), func=calculate_is, stopping_func=sigma_fn,
        tolerance_threshold=tolerance_threshold, max_iteration=max_iteration
    )

    return is_mean, is_sigma


def frechet_inception_distance(model, data_generator, tolerance_threshold=1e-6, max_iteration=100, batch_size=10):
    @delayed
    def calculate_fid(generated_mean, generated_sigma, inception_mean, inception_sigma):
        """Numpy implementation of the Frechet Distance.
           The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
           and X_2 ~ N(mu_2, C_2) is
                    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
           Stable version by Dougal J. Sutherland.
           Params:
            -- mu1(generated_mean) : Numpy array containing the activations of the pool_3 layer of the
                     inception net ( like returned by the function 'get_predictions')
                     for generated samples.
            -- mu2(inception_mean)  : The sample mean over activations of the pool_3 layer, precalcualted
                       on an representive data set.
            -- sigma1(generated_sigma): The covariance matrix over activations of the pool_3 layer for
                       generated samples.
            -- sigma2(inception_sigma): The covariance matrix over activations of the pool_3 layer,
                       precalcualted on an representive data set.
           Returns:
            --   : The Frechet Distance.
        """
        generated_mean = da.from_array(generated_mean, chunks=100)
        inception_mean = da.from_array(inception_mean, chunks=100)
        generated_sigma = da.from_array(generated_sigma, chunks=100)
        inception_sigma = da.from_array(inception_sigma, chunks=100)

        # calculate sum squared difference between means
        ssdiff = np.sum((generated_mean - inception_mean) ** 2.0)

        # calculate sqrt of product between cov
        covmean = sp.linalg.sqrtm(generated_sigma.dot(inception_sigma))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + da.trace(generated_sigma + inception_sigma - 2.0 * covmean)
        return fid

    # prepare the inception v3 model
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=model.get_inputs_shape())
    inception_predictions = lambda x: inception_model.predict(inception_preprocess_input(x))

    def inception_predictions_generator():
        while True:
            images = next(data_generator)
            images = (images * 255).astype(np.uint8)
            yield inception_predictions(images)[:batch_size]

    print('calculating the inception images sigma ...')
    inception_images_sigma = bootstrapping_additive(
        data_generator=inception_predictions_generator(), func=delayed(lambda x: x), \
        stopping_func=sigma_fn, tolerance_threshold=tolerance_threshold, max_iteration=max_iteration
    )

    def inception_predictions_generator():
        while True:
            images = next(data_generator)
            images = (images * 255).astype(np.uint8)
            yield inception_predictions(images)[:batch_size]

    print('calculating the inception images mean ...')
    inception_images_mean = bootstrapping_additive(
        data_generator=inception_predictions_generator(), func=delayed(lambda x: x), \
        stopping_func=mean_fn, tolerance_threshold=tolerance_threshold, max_iteration=max_iteration
    )

    def generated_predictions_generator():
        while True:
            # Generate latents from the data
            data = next(data_generator)
            latents_real = model.encode(data)

            # Generate random latents and interpolation t-values.
            ln = np.random.normal(size=[latents_real.shape[1]])
            latents_t = np.array([ln for _ in range(latents_real.shape[0])])
            lerp_t = np.random.uniform()

            latents_e0 = slerp(lerp_t, latents_real[0::2], latents_t[1::2])
            latents_e1 = slerp(lerp_t + epsilon, latents_real[0::2], latents_t[1::2])

            latents_e = np.vstack([latents_e0, latents_e1])

            images = model.decode(latents_e).numpy()
            images = (images * 255).astype(np.uint8)
            yield inception_predictions(images)[:batch_size]

    print('calculating the generated images sigma ...')
    generated_images_sigma = bootstrapping_additive(
        data_generator=generated_predictions_generator(), func=delayed(lambda x: x), \
        stopping_func=sigma_fn, tolerance_threshold=tolerance_threshold, max_iteration=max_iteration
    )

    def generated_predictions_generator():
        while True:
            # Generate latents from the data
            data = next(data_generator)
            latents_real = model.encode(data)

            # Generate random latents and interpolation t-values.
            ln = np.random.normal(size=[latents_real.shape[1]])
            latents_t = np.array([ln for _ in range(latents_real.shape[0])])
            lerp_t = np.random.uniform()

            latents_e0 = slerp(lerp_t, latents_real[0::2], latents_t[1::2])
            latents_e1 = slerp(lerp_t+epsilon, latents_real[0::2], latents_t[1::2])

            latents_e = np.vstack([latents_e0, latents_e1])

            images = model.decode(latents_e).numpy()
            images = (images * 255).astype(np.uint8)
            yield inception_predictions(images)[:batch_size]

    print('calculating the generated images mean ...')
    generated_images_mean = bootstrapping_additive(
        data_generator=generated_predictions_generator(), func=delayed(lambda x: x), \
        stopping_func=mean_fn, tolerance_threshold=tolerance_threshold, max_iteration=max_iteration
    )

    return calculate_fid(generated_images_mean, generated_images_sigma, inception_images_mean, inception_images_sigma)
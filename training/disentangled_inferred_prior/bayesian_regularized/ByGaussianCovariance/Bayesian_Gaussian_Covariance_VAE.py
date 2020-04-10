import tensorflow as tf

from training.disentangled_inferred_prior.DIP_shared import infer_gaussian_prior
import tensorflow_probability as tfp
from training.disentangled_inferred_prior.regularized.ByCovariance.Covariance_VAE import Covariance_VAE

class Bayesian_Gaussian_Covariance_AE(Covariance_VAE):
    def __encode__(self, **kwargs):
        inputs = kwargs['inputs']
        for k, v in  inputs.items():
            if inputs[k].shape == self.get_inputs_shape():
                inputs[k] = tf.reshape(inputs[k], (1, ) + self.get_inputs_shape())
            inputs[k] = tf.cast(inputs[k], tf.float32)
        kwargs['model']  = self.get_variable
        kwargs['latents_shape'] = (self.batch_size, self.latents_dim)

        encoded = self.encode_fn(**kwargs)
        covariance_mean, covariance_regularizer = infer_gaussian_prior(
                                                        latent_mean=encoded['inference_mean'],
                                                        latent_logvariance=encoded['inference_logvariance'],\
                                                        regularize=True, lambda_d=self.lambda_d, lambda_od=self.lambda_od
        )

        covariance_sigma = tf.exp(encoded['inference_logvariance'])
        latents_sigma = covariance_sigma
        prior_distribution = tfp.distributions.Normal(loc=covariance_mean, scale=covariance_sigma)
        posterior_distribution = tfp.distributions.Normal(loc=encoded['inference_mean'], scale=latents_sigma)
        bayesian_divergent = tfp.distributions.kl_divergence(posterior_distribution, prior_distribution)
        bayesian_divergent = tf.identity(bayesian_divergent, name='bayesian_divergent' )

        return {**encoded,
                'covariance_regularized': covariance_regularizer,
                'bayesian_divergent': bayesian_divergent}


import tensorflow as tf
from training.disentangled_inferred_prior.regularized.ByCovariance.Covariance_VAE import Covariance_VAE
from training.disentangled_inferred_prior.DIP_shared import infer_gaussian_prior

class Gaussian_Covariance_VAE(Covariance_VAE):
    def __encode__(self, **kwargs):
        inputs = kwargs['inputs']
        for k, v in  inputs.items():
            if inputs[k].shape == self.get_inputs_shape():
                inputs[k] = tf.reshape(inputs[k], (1, ) + self.get_inputs_shape())
            inputs[k] = tf.cast(inputs[k], tf.float32)
        kwargs['model']  = self.get_variable
        kwargs['latents_shape'] = (self.batch_size, self.latents_dim)

        encoded = self.encode_fn(**kwargs)
        _, covariance_regularizer = infer_gaussian_prior(
                                                        latent_mean=encoded['z_latents'],
                                                        latent_logvariance=encoded['inference_logvariance'],\
                                                        regularize=True, lambda_d=self.lambda_d, lambda_od=self.lambda_od
        )
        return {**encoded, 'covariance_regularized': covariance_regularizer}
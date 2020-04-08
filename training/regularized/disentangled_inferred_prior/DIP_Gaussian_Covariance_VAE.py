import tensorflow as tf
from training.regularized.disentangled_inferred_prior.DIP_Covariance_VAE import DIP_Covariance_VAE
from training.regularized.DIP_shared import gaussian_regularize

class DIP_Gaussian_Covariance_VAE(DIP_Covariance_VAE):
    def __encode__(self, **kwargs):
        inputs = kwargs['inputs']
        for k, v in  inputs.items():
            if inputs[k].shape == self.get_inputs_shape():
                inputs[k] = tf.reshape(inputs[k], (1, ) + self.get_inputs_shape())
            inputs[k] = tf.cast(inputs[k], tf.float32)
        kwargs['model']  = self.get_variable
        kwargs['latents_shape'] = (self.batch_size, self.latents_dim)

        encoded = self.encode_fn(**kwargs)
        _, covariance_regularizer = gaussian_regularize(
                                                        latent_mean=encoded['z_latents'],
                                                        latent_logvariance=encoded['inference_logvariance'],\
                                                        regularize=True, lambda_d=self.lambda_d, lambda_od=self.lambda_od
        )
        return {**encoded, 'covariance_regularized': covariance_regularizer}
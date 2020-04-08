import tensorflow as tf
from training.regularized.disentangled_inferred_prior.DIP_Covariance_AE import DIP_Covariance_AE
from training.regularized.DIP_shared import gaussian_regularize

class DIP_Gaussian_Covariance_AE(DIP_Covariance_AE):
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
                                                        latent_logvariance=tf.sigmoid(encoded['z_latents']),\
                                                        regularize=True, lambda_d=self.lambda_d, d=self.d
        )
        return {**encoded, 'covariance_regularized': covariance_regularizer}



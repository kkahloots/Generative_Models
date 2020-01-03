
import tensorflow as tf

from graphs.basics.VAE_graph import make_vae, encode
from training.transformative.AE import AE


class VAE(AE):
    def __init__(
            self,
            model_name,
            inputs_shape,
            outputs_shape,
            latent_dim,
            variables_params,
            restore=None
    ):

        AE.__init__(self,
            model_name=model_name,
            inputs_shape=inputs_shape,
            outputs_shape=outputs_shape,
            latent_dim=latent_dim,
            variables_params=variables_params,
            restore=restore,
            make_ae=make_vae)

        self.encode_graph = encode

    @tf.function
    def feedforward(self, inputs):
        z, mean, logvar = self.encode(inputs)
        x_logit = self.decode(z)
        return {'x_logit': x_logit, 'latent': z, 'mean': mean, 'logvar':logvar}

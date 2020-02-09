import tensorflow as tf

from graphs.basics.VAE_graph import create_graph, encode_fn
from training.traditional.autoencoders.autoencoder import autoencoder


class VAE(autoencoder):
    def __init__(
            self,
            name,
            inputs_shape,
            outputs_shape,
            latent_dim,
            variables_params,
            filepath=None
    ):

        autoencoder.__init__(self,
                             name=name,
                             inputs_shape=inputs_shape,
                             outputs_shape=outputs_shape,
                             latent_dim=latent_dim,
                             variables_params=variables_params,
                             filepath=filepath,
                             model_fn=create_graph)

        self.encode_graph = encode_fn

    @tf.function
    def feedforwad(self, inputs):
        X = inputs[0]
        y = inputs[1]
        z, mean, logvar = self.encode(X)
        x_logit = self.decode(tf.concat[z, y])
        return {'x_logit': x_logit, 'latent': z, 'mean': mean, 'logvar':logvar}
    
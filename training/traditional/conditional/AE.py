import tensorflow as tf

from graphs.basics.AE_graph import make_ae, encode
from training.traditional.autoencoders.AE import AE as basicAE


class AE(basicAE):
    def __init__(
            self,
            model_name,
            inputs_shape,
            outputs_shape,
            latent_dim,
            variables_params,
            restore=None
    ):

        basicAE.__init__(self,
            model_name=model_name,
            inputs_shape=inputs_shape,
            outputs_shape=outputs_shape,
            latent_dim=latent_dim,
            variables_params=variables_params,
            restore=restore,
            make_ae=make_ae
                    )

        self.encode_graph = encode

    @tf.function
    def feedforward(self, inputs):
        X = inputs[0]
        y = inputs[1]
        z = self.encode(X)
        x_logit = self.decode(tf.concat[z, y])
        return {'x_logit': x_logit, 'latent': z}


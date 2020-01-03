
from graphs.denoising.Deno_AE_graph import make_deno_ae
from training.autoencoders.AE import AE, encode


class DenoAE(AE):
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
            make_ae=make_deno_ae)

        self.encode_graph = encode

    # @tf.function
    # def feedforward(self, inputs):
    #     z, mean, logvar = self.encode(inputs)
    #     x_logit = self.decode(z)
    #     return {'x_logit': x_logit, 'latent': z, 'mean': mean, 'logvar':logvar}

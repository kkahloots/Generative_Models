from graphs.basics_conditional.Cond_AE_graph import  make_cond_ae, encode
from training.autoencoders.AE import AE


class Cond_AE(AE):
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
            make_ae=make_cond_ae)

        self.encode_graph = encode

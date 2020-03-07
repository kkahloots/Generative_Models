from training.adversarial.autoencoders.AAE import AAE
from training.traditional.conditional.AE import autoencoder as TradAdapteeCondAE
from training.traditional.conditional.VAE import VAE as TradAdapteeCondVAE
from training.traditional.conditional_transformative.AE import autoencoder as TradAdapteeCondTranAE
from training.traditional.conditional_transformative.VAE import VAE as TradAdapteeCondTranVAE
from training.traditional.transformative.AE import autoencoder as TradAdapteeTranAE
from training.traditional.transformative.VAE import VAE as TradAdapteeTranVAE

def make_adver_VAE(
        model_name,
        inputs_shape,
        outputs_shape,
        latent_dim,
        variables_params,
        restore=None
    ):
    return AAE(
        model_name=model_name,
        inputs_shape=inputs_shape,
        outputs_shape=outputs_shape,
        latent_dim=latent_dim,
        variables_params=variables_params,
        restore=restore,
        model_fn=create_graph,
        encode_fn=encode_fn,
    )


def make_adver_CondAE(
        model_name,
        inputs_shape,
        outputs_shape,
        latent_dim,
        variables_params,
        restore=None
    ):
    return AE(
        model_name=model_name,
        inputs_shape=inputs_shape,
        outputs_shape=outputs_shape,
        latent_dim=latent_dim,
        variables_params=variables_params,
        restore=restore,
        AE=TradAdapteeCondAE
    )

def make_adver_CondVAE(
        model_name,
        inputs_shape,
        outputs_shape,
        latent_dim,
        variables_params,
        restore=None
    ):
    return AE(
        model_name=model_name,
        inputs_shape=inputs_shape,
        outputs_shape=outputs_shape,
        latent_dim=latent_dim,
        variables_params=variables_params,
        restore=restore,
        AE=TradAdapteeCondVAE
    )

def make_adver_CondTranAE(
        model_name,
        inputs_shape,
        outputs_shape,
        latent_dim,
        variables_params,
        restore=None
    ):
    return AE(
        model_name=model_name,
        inputs_shape=inputs_shape,
        outputs_shape=outputs_shape,
        latent_dim=latent_dim,
        variables_params=variables_params,
        restore=restore,
        AE=TradAdapteeCondTranAE
    )

def make_adver_CondTranVAE(
        model_name,
        inputs_shape,
        outputs_shape,
        latent_dim,
        variables_params,
        restore=None
    ):
    return AE(
        model_name=model_name,
        inputs_shape=inputs_shape,
        outputs_shape=outputs_shape,
        latent_dim=latent_dim,
        variables_params=variables_params,
        restore=restore,
        AE=TradAdapteeCondTranVAE
    )

def make_adver_TranAE(
        model_name,
        inputs_shape,
        outputs_shape,
        latent_dim,
        variables_params,
        restore=None
    ):
    return AE(
        model_name=model_name,
        inputs_shape=inputs_shape,
        outputs_shape=outputs_shape,
        latent_dim=latent_dim,
        variables_params=variables_params,
        restore=restore,
        AE=TradAdapteeTranAE
    )

def make_adver_TranVAE(
        model_name,
        inputs_shape,
        outputs_shape,
        latent_dim,
        variables_params,
        restore=None
    ):
    return AE(
        model_name=model_name,
        inputs_shape=inputs_shape,
        outputs_shape=outputs_shape,
        latent_dim=latent_dim,
        variables_params=variables_params,
        restore=restore,
        AE=TradAdapteeTranVAE
    )


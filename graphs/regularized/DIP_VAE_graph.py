
from graphs.regularized.DIP_AE_graph import covariance_regularized
from graphs.basics.VAE_graph import logpx_z_fn, logpz_fn, logqz_x_fn

def create_losses():
    return {
        'x_logits': logpx_z_fn,
        'z_latents': logpz_fn,
        'x_logpdf': logqz_x_fn,
        'covariance_regularized': covariance_regularized
    }



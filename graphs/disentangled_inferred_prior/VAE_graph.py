
from graphs.disentangled_inferred_prior.AE_graph import covariance_regularized, bayesian_divergence
from graphs.basics.VAE_graph import logpx_z_fn, logpz_fn, logqz_x_fn

def create_regularized_losses():
    return {
        'x_logits': logpx_z_fn,
        'z_latents': logpz_fn,
        'x_logpdf': logqz_x_fn,
        'covariance_regularized': covariance_regularized
    }

def create_Bayesian_losses():
    return {
        'x_logits': logpx_z_fn,
        'z_latents': logpz_fn,
        'x_logpdf': logqz_x_fn,
        'bayesian_divergent': bayesian_divergence
            }


def create_regularized_Bayesian_losses():
    return {
        'x_logits': logpx_z_fn,
        'z_latents': logpz_fn,
        'x_logpdf': logqz_x_fn,
        'covariance_regularized': covariance_regularized,
        'bayesian_divergent': bayesian_divergence
            }

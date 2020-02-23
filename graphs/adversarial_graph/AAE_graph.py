import tensorflow as tf
from graphs.basics.AE_graph import encode_fn


def latent_discriminate_encode_fn(**kwargs):
    # swapping the true by random
    ae_encoded = encode_fn(**kwargs)
    fake_latent = ae_encoded['x_latent']
    real_latent = tf.random.normal(shape=tf.shape(fake_latent))
    real_pred = kwargs['model']('latent_real_discriminator', [real_latent])
    fake_pred = kwargs['model']('latent_fake_discriminator', [fake_latent])
    return {**ae_encoded, 'fake_pred': fake_pred, 'real_pred':  real_pred}


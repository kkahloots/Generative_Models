import tensorflow as tf
from graphs.basics.AE_graph import encode_fn


def latent_discriminate_encode_fn(**kwargs):
    # swapping the true by random
    ae_encoded = encode_fn(**kwargs)
    fake_latent = ae_encoded['x_latent']
    real_latent = tf.random.normal(shape=tf.shape(fake_latent))
    real_pred = kwargs['model']('latent_real_discriminator', [real_latent])
    fake_pred = kwargs['model']('latent_fake_discriminator', [fake_latent])
    return {**ae_encoded, 'latent_fake_pred': fake_pred, 'latent_real_pred':  real_pred}


def inputs_discriminate_encode_fn(**kwargs):
    # swapping the true by random
    fake_inputs = kwargs['inputs']['inputs']
    ae_encoded = encode_fn(**kwargs)
    real_inputs = tf.random.normal(shape=tf.shape(fake_inputs))
    inputs_real_pred = kwargs['model']('inputs_real_discriminator', [real_inputs])
    inputs_fake_pred = kwargs['model']('inputs_fake_discriminator', [fake_inputs])
    return {**ae_encoded, 'inputs_fake_pred': inputs_fake_pred, 'inputs_real_pred':  inputs_real_pred}


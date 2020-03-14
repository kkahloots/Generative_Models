import tensorflow as tf
from graphs.basics.VAE_graph import encode_fn

def inputs_discriminate_encode_fn(**kwargs):
    # swapping the true by random
    fake_inputs = kwargs['inputs']['x_mean']
    ae_encoded = encode_fn(**kwargs)
    real_inputs = tf.random.normal(shape=tf.shape(fake_inputs))
    inputs_real_pred = kwargs['model']('inputs_real_discriminator', [real_inputs])
    inputs_fake_pred = kwargs['model']('inputs_fake_discriminator', [fake_inputs])
    return {**ae_encoded, 'inputs_fake_pred': inputs_fake_pred, 'inputs_real_pred':  inputs_real_pred}


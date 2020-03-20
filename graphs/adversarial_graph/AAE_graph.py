import tensorflow as tf
from graphs.basics.AE_graph import encode_fn

def latent_discriminate_encode_fn(**kwargs):
    # swapping the true by random
    ae_encoded = encode_fn(**kwargs)
    fake_latent = ae_encoded['x_latent']
    real_latent = tf.random.normal(shape=tf.shape(fake_latent))
    real_discriminator_pred = kwargs['model']('latent_discriminator_real', [real_latent])
    fake_discriminator_pred = kwargs['model']('latent_discriminator_fake', [fake_latent])
    fake_generator_pred = kwargs['model']('latent_generator_fake', [fake_latent])
    return {**ae_encoded, 'latent_discriminator_fake_pred': fake_discriminator_pred,
                          'latent_generator_fake_pred': fake_generator_pred,
                          'latent_discriminator_real_pred':  real_discriminator_pred}

def inputs_discriminate_encode_fn(**kwargs):
    # swapping the true by random
    fake_inputs = kwargs['inputs']['inputs']
    ae_encoded = encode_fn(**kwargs)
    real_inputs = tf.random.normal(shape=tf.shape(fake_inputs))
    inputs_discriminator_real_pred = kwargs['model']('inputs_discriminator_real', [real_inputs])
    inputs_discriminator_fake_pred = kwargs['model']('inputs_discriminator_fake', [fake_inputs])
    inputs_generator_fake_pred = kwargs['model']('inputs_generator_fake', [fake_inputs])
    return {**ae_encoded, 'inputs_discriminator_fake_pred': inputs_discriminator_fake_pred,
                          'inputs_generator_fake_pred': inputs_generator_fake_pred,
                          'inputs_discriminator_real_pred':  inputs_discriminator_real_pred}


def inputs_latent_discriminate_encode_fn(**kwargs):
    # swapping the true by random
    fake_inputs = kwargs['inputs']['inputs']
    ae_encoded = encode_fn(**kwargs)
    real_inputs = tf.random.normal(shape=tf.shape(fake_inputs))
    inputs_discriminator_real_pred = kwargs['model']('inputs_discriminator_real', [real_inputs])
    inputs_discriminator_fake_pred = kwargs['model']('inputs_discriminator_fake', [fake_inputs])
    inputs_generator_fake_pred = kwargs['model']('inputs_generator_fake', [fake_inputs])

    # swapping the true by random
    fake_latent = ae_encoded['x_latent']
    real_latent = tf.random.normal(shape=tf.shape(fake_latent))
    real_discriminator_pred = kwargs['model']('latent_discriminator_real', [real_latent])
    fake_discriminator_pred = kwargs['model']('latent_discriminator_fake', [fake_latent])
    fake_generator_pred = kwargs['model']('latent_generator_fake', [fake_latent])

    return {**ae_encoded, 'inputs_discriminator_fake_pred': inputs_discriminator_fake_pred, \
                          'inputs_generator_fake_pred': inputs_generator_fake_pred,
                          'inputs_discriminator_real_pred':  inputs_discriminator_real_pred,
                          'latent_discriminator_fake_pred': fake_discriminator_pred,
                          'latent_generator_fake_pred': fake_generator_pred,
                          'latent_discriminator_real_pred':  real_discriminator_pred
            }


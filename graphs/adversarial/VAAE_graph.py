import tensorflow as tf
from graphs.basics.VAE_graph import encode_fn

def inputs_discriminate_encode_fn(**kwargs):
    # swapping the true by random
    fake_inputs = kwargs['inputs']['x_mean']
    ae_encoded = encode_fn(**kwargs)
    real_inputs = tf.random.normal(shape=tf.shape(fake_inputs))
    inputs_discriminator_real_predictions = kwargs['model']('inputs_discriminator_real', [real_inputs])
    inputs_discriminator_fake_predictions = kwargs['model']('inputs_discriminator_fake', [fake_inputs])
    inputs_generator_fake_predictions = kwargs['model']('inputs_generator_fake', [fake_inputs])
    return {**ae_encoded, 'inputs_discriminator_fake_predictions': inputs_discriminator_fake_predictions,
                          'inputs_generator_fake_predictions': inputs_generator_fake_predictions,
                          'inputs_discriminator_real_predictions':  inputs_discriminator_real_predictions
            }

def inputs_latent_discriminate_encode_fn(**kwargs):
    # swapping the true by random
    fake_inputs = kwargs['inputs']['x_mean']
    ae_encoded = encode_fn(**kwargs)
    real_inputs = tf.random.normal(shape=tf.shape(fake_inputs))
    inputs_discriminator_real_predictions = kwargs['model']('inputs_discriminator_real', [real_inputs])
    inputs_discriminator_fake_predictions = kwargs['model']('inputs_discriminator_fake', [fake_inputs])
    inputs_generator_fake_predictions = kwargs['model']('inputs_generator_fake', [fake_inputs])

    # swapping the true by random
    fake_latent = ae_encoded['z_latent']
    real_latent = tf.random.normal(shape=tf.shape(fake_latent))
    latent_discriminator_real_predictions = kwargs['model']('latent_discriminator_real', [real_latent])
    latent_discriminator_fake_predictions = kwargs['model']('latent_discriminator_fake', [fake_latent])
    latent_generator_fake_predictions = kwargs['model']('latent_generator_fake', [fake_latent])

    return {**ae_encoded, 'inputs_discriminator_fake_predictions': inputs_discriminator_fake_predictions, \
                          'inputs_generator_fake_predictions': inputs_generator_fake_predictions,
                          'inputs_discriminator_real_predictions':  inputs_discriminator_real_predictions,
                          'latent_discriminator_fake_predictions': latent_discriminator_fake_predictions,
                          'latent_generator_fake_predictions': latent_generator_fake_predictions,
                          'latent_discriminator_real_predictions':  latent_discriminator_real_predictions
            }


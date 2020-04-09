import tensorflow as tf
from graphs.basics.AE_graph import encode_fn

def inference_discriminate_encode_fn(**kwargs):
    # swapping the true by random
    ae_encoded = encode_fn(**kwargs)
    fake_latents = ae_encoded['z_latents']
    real_latents = tf.random.normal(shape=tf.shape(fake_latents))
    real_discriminator_predictions = kwargs['model']('inference_discriminator_real', [real_latents])
    fake_discriminator_predictions = kwargs['model']('inference_discriminator_fake', [fake_latents])
    fake_generator_predictions = kwargs['model']('inference_generator_fake', [fake_latents])
    return {**ae_encoded, 'inference_discriminator_fake_predictions': fake_discriminator_predictions,
                          'inference_generator_fake_predictions': fake_generator_predictions,
                          'inference_discriminator_real_predictions':  real_discriminator_predictions}

def generative_discriminate_encode_fn(**kwargs):
    # swapping the true by random
    fake_inputs = kwargs['inputs']['inputs']
    ae_encoded = encode_fn(**kwargs)
    real_inputs = tf.random.normal(shape=tf.shape(fake_inputs))
    generative_discriminator_real_predictions = kwargs['model']('generative_discriminator_real', [real_inputs])
    generative_discriminator_fake_predictions = kwargs['model']('generative_discriminator_fake', [fake_inputs])
    generative_generator_fake_predictions = kwargs['model']('generative_generator_fake', [fake_inputs])
    return {**ae_encoded, 'generative_discriminator_fake_predictions': generative_discriminator_fake_predictions,
                          'generative_generator_fake_predictions': generative_generator_fake_predictions,
                          'generative_discriminator_real_predictions':  generative_discriminator_real_predictions}


def generative_inference_discriminate_encode_fn(**kwargs):
    # swapping the true by random
    fake_inputs = kwargs['inputs']['inputs']
    ae_encoded = encode_fn(**kwargs)
    real_inputs = tf.random.normal(shape=tf.shape(fake_inputs))
    generative_discriminator_real_predictions = kwargs['model']('generative_discriminator_real', [real_inputs])
    generative_discriminator_fake_predictions = kwargs['model']('generative_discriminator_fake', [fake_inputs])
    generative_generator_fake_predictions = kwargs['model']('generative_generator_fake', [fake_inputs])

    # swapping the true by random
    fake_latents = ae_encoded['z_latents']
    real_latents = tf.random.normal(shape=tf.shape(fake_latents))
    real_discriminator_predictions = kwargs['model']('inference_discriminator_real', [real_latents])
    fake_discriminator_predictions = kwargs['model']('inference_discriminator_fake', [fake_latents])
    fake_generator_predictions = kwargs['model']('inference_generator_fake', [fake_latents])

    return {**ae_encoded, 'generative_discriminator_fake_predictions': generative_discriminator_fake_predictions, \
                          'generative_generator_fake_predictions': generative_generator_fake_predictions,
                          'generative_discriminator_real_predictions':  generative_discriminator_real_predictions,
                          'inference_discriminator_fake_predictions': fake_discriminator_predictions,
                          'inference_generator_fake_predictions': fake_generator_predictions,
                          'inference_discriminator_real_predictions':  real_discriminator_predictions
            }


import tensorflow as tf
from graphs.basics.AE_graph import bce


def create_adversarial_losses():
    return {
        'latent_real_discriminator_outputs': real_bce_fn,
        'latent_fake_discriminator_outputs': fake_bce_fn,
         'x_logits': bce
    }

def create_adversarial_real_losses():
    return {
        'latent_real_discriminator_outputs': real_bce_fn,
    }

def create_adversarial_fake_losses():
    return {
        'latent_fake_discriminator_outputs': fake_bce_fn,
    }

def real_bce_fn(real_true, real_pred):
    real_loss = tf.losses.binary_crossentropy(y_true=real_true, y_pred=real_pred)
    return  0.5 * real_loss

def fake_bce_fn(fake_true, fake_pred):
    fake_loss = tf.losses.binary_crossentropy(y_true=fake_true, y_pred=fake_pred)
    return  0.5 * fake_loss
import tensorflow as tf

def create_inputs_latent_adversarial_losses():
    return {
        'inputs_real_discriminator_outputs': real_bce_fn,
        'inputs_fake_discriminator_outputs': fake_bce_fn,
        'latent_real_discriminator_outputs': real_bce_fn,
        'latent_fake_discriminator_outputs': fake_bce_fn,
    }

def create_inputs_adversarial_losses():
    return {
        'inputs_real_discriminator_outputs': real_bce_fn,
        'inputs_fake_discriminator_outputs': fake_bce_fn,
    }

def create_inputs_adversarial_real_losses():
    return {
        'inputs_real_discriminator_outputs': real_bce_fn,
    }

def create_inputs_adversarial_fake_losses():
    return {
        'inputs_fake_discriminator_outputs': fake_bce_fn,
    }

def create_latent_adversarial_losses():
    return {
        'latent_real_discriminator_outputs': real_bce_fn,
        'latent_fake_discriminator_outputs': fake_bce_fn,
    }

def create_latent_adversarial_real_losses():
    return {
        'latent_real_discriminator_outputs': real_bce_fn,
    }

def create_latent_adversarial_fake_losses():
    return {
        'latent_fake_discriminator_outputs': fake_bce_fn,
    }

def real_bce_fn(real_true, real_pred_logits):
    real_loss = tf.losses.binary_crossentropy(y_true=real_true, y_pred=tf.sigmoid(real_pred_logits))
    return  0.5 * real_loss

def fake_bce_fn(fake_true, fake_pred_logits):
    fake_loss = tf.losses.binary_crossentropy(y_true=fake_true, y_pred=tf.sigmoid(fake_pred_logits))
    return  0.5 * fake_loss
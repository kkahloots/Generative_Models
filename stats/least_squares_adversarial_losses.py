# https://arxiv.org/abs/1611.04076
import tensorflow as tf

def create_inputs_adversarial_discriminator_real_losses():
    return {
        'inputs_adversarial_discriminator_real_outputs': real_ls_fn,
    }

def create_inputs_adversarial_discriminator_fake_losses():
    return {
        'inputs_adversarial_discriminator_fake_outputs': fake_ls_fn,
    }

def create_latent_adversarial_discriminator_real_losses():
    return {
        'latent_adversarial_discriminator_real_outputs': real_ls_fn,
    }

def create_latent_adversarial_discriminator_fake_losses():
    return {
        'latent_adversarial_discriminator_fake_outputs': fake_ls_fn,
    }

def create_latent_adversarial_generator_fake_losses():
    return {
        'latent_adversarial_generator_fake_outputs': gfake_ls_fn,
    }

def real_ls_fn(real_true, real_pred_logits):
    real_loss = mse(y_true=real_true, y_pred=real_pred_logits)
    return  0.5 * real_loss

def gfake_ls_fn(fake_true, fake_pred_logits):
    fake_loss = mse(y_true=fake_true, y_pred=fake_pred_logits)
    return  fake_loss

def fake_ls_fn(fake_true, fake_pred_logits):
    fake_loss = mse(y_true=fake_true, y_pred=fake_pred_logits)
    return  0.5 * fake_loss

def mse(y_true, y_pred):
    loss_val = tf.sqrt(2 * tf.nn.l2_loss(y_pred - y_true))
    return loss_val
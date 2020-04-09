import tensorflow as tf

def create_generative_discriminator_real_losses():
    return  real_bce_fn
def create_generative_discriminator_fake_losses():
    return fake_bce_fn
def create_generative_generator_fake_losses():
    return gfake_bce_fn

def create_inference_discriminator_real_losses():
    return  real_bce_fn
def create_inference_discriminator_fake_losses():
    return fake_bce_fn
def create_inference_generator_fake_losses():
    return gfake_bce_fn

def real_bce_fn(real_true, real_pred_logits):
    real_loss = tf.losses.binary_crossentropy(y_true=real_true, y_pred=tf.sigmoid(real_pred_logits))
    return  0.5 * real_loss

def fake_bce_fn(fake_true, fake_pred_logits):
    fake_loss = tf.losses.binary_crossentropy(y_true=fake_true, y_pred=tf.sigmoid(fake_pred_logits))
    return  0.5 * fake_loss

def gfake_bce_fn(fake_true, fake_pred_logits):
    fake_loss = tf.losses.binary_crossentropy(y_true=fake_true, y_pred=tf.sigmoid(fake_pred_logits))
    return  fake_loss
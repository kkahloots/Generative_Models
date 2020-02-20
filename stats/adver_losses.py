import tensorflow as tf

def discr_bce_fn(inputs, predictions):
    real_pred = predictions['real_pred']
    fake_pred = predictions['fake_pred']
    real_loss = tf.losses.binary_crossentropy(y_true=tf.ones_like(real_pred),\
                                                        y_pred=real_pred)
    fake_loss = tf.losses.binary_crossentropy(y_true=tf.zeros_like(fake_pred), \
                                                        y_pred=fake_pred)
    return 0.5 * (real_loss + fake_loss)

def create_adversarial_losses():
    return {
        'latent_real_outputs': real_bce_fn,
        'latent_fake_outputs': fake_bce_fn
    }

def fake_bce_fn(fake_true, fake_pred):
    fake_loss = tf.losses.binary_crossentropy(y_true=fake_true, y_pred=fake_pred)
    return 0.5 * fake_loss

def real_bce_fn(real_true, real_pred):
    real_loss = tf.losses.binary_crossentropy(y_true=real_true, y_pred=real_pred)
    return 0.5 * real_loss
import tensorflow as tf

@tf.function
def compute_discr_bce(inputs, predictions):
    real_pred = predictions['real_pred']
    fake_pred = predictions['fake_pred']
    real_loss = tf.losses.binary_crossentropy(y_true=tf.ones_like(real_pred),\
                                                        y_pred=real_pred)
    fake_loss = tf.losses.binary_crossentropy(y_true=tf.zeros_like(fake_pred), \
                                                        y_pred=fake_pred)
    return 0.5 * (real_loss + fake_loss)

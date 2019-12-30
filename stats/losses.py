import tensorflow as tf
import tensorflow_probability as tfp

def reconstuction_loss(true_x, pred_x):
    """expected log-likelihood of the lower bound. For this we use a bernouilli lower bound
    Computes the Bernoulli loss."""
    # Because true images are not binary, the lower bound in the xent is not zero:
    # the lower bound in the xent is the entropy of the true images.
    dist = tfp.distributions.Bernoulli(
        probs=tf.clip_by_value(true_x, 1e-6, 1 - 1e-6))
    loss_lower_bound = tf.reduce_sum(dist.entropy(), axis=[1,2,3])

    ell = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pred_x, labels=true_x),
        axis=[1,2,3])

    return ell - loss_lower_bound

#def reconstuction_loss(true_x, pred_x):
#    return tf.losses.binary_crossentropy(y_true=true_x, y_pred=pred_x, from_logits=True)
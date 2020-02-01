import tensorflow as tf
def ssmi(inputs, x_logit):
    imageB = tf.sigmoid(x_logit)
    imageA  = inputs

    ssmi = tf.reduce_mean(tf.image.ssim(imageA, imageB, max_val=1.0))
    return ssmi

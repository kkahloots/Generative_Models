import tensorflow as tf
def ssmi(inputs, x_logits):
    imageB = tf.sigmoid(x_logits)
    imageA  = inputs

    ssmi = 1-tf.reduce_mean(tf.image.ssim(imageA, imageB, max_val=1.0))
    ssmi = 0.5*ssmi
    return ssmi

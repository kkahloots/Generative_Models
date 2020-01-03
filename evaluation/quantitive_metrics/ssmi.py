import tensorflow as tf
def ssmi_metric(inputs):
    imageB = inputs['y']
    imageA  = inputs['X']

    ssmi = tf.reduce_mean(tf.image.ssim(imageA, imageB, max_val=1.0))
    return ssmi

import tensorflow as tf
from skimage.measure import compare_ssim
def ssim_multiscale(imageA, imageB):
    shapeB = list(imageB.shape)
    if len(shapeB) > 4:
        shapeA = list(imageA.shape)
        imageA = tf.reshape(imageA, tf.TensorShape([shapeA[0] * shapeA[1]] + shapeA[2:]))
        imageB = tf.reshape(imageB, tf.TensorShape([shapeB[0] * shapeB[1]] + shapeB[2:]))

    score = tf.reduce_sum(tf.cast(compare_ssim(imageA, imageB, full=False,  multichannel=True)), dtype='float')
    return score

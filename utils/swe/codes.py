import tensorflow as tf

def assert_codes(instance, Class, message):
    assert instance in properties(Class), Messages_CD.USV.format(instance, 'Layers', properties(Class))
    pass

def properties(cls):
    return [i for i in cls.__dict__.keys() if i[:1] != '_']

class Messages_CD:
    USV = '{} is unsupported value for {}, Please choose out of {}'

class Layer_CD:
    Conv   = tf.keras.layers.Conv2D
    Deconv = tf.keras.layers.Conv2DTranspose
    Dense  = tf.keras.layers.Dense

class Sampling_CD:
    UpSampling   = tf.keras.layers.UpSampling2D
    DownSampling = tf.keras.layers.MaxPool2D

class Activate_CD:
    relu = 'relu'
    softmax = 'softmax'
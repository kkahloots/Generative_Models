epsilon = 1e-6

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


import types
import functools

def copy_fn(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
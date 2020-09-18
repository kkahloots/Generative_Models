import tensorflow as tf
import six
import json
import numpy as np

def get_shape(image_size, color_mode='rgb', data_format=tf.keras.backend.image_data_format()):
    if color_mode == 'rgb':
        if data_format == 'channels_last':
            image_shape = image_size + (3,)
        else:
            image_shape = (3,) + image_size
    else:
        if data_format == 'channels_last':
            image_shape = image_size + (1,)
        else:
            image_shape = (1,) + image_size
    return image_shape


def as_bytes(bytes_or_text, encoding='utf-8'):
    """Converts bytes or unicode to `bytes`, using utf-8 encoding for text.

    # Arguments
        bytes_or_text: A `bytes`, `str`, or `unicode` object.
        encoding: A string indicating the charset for encoding unicode.

    # Returns
        A `bytes` object.

    # Raises
        TypeError: If `bytes_or_text` is not a binary or unicode string.
    """
    if isinstance(bytes_or_text, six.text_type):
        return bytes_or_text.encode(encoding)
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text
    else:
        raise TypeError('Expected binary or unicode string, got %r' %
                        (bytes_or_text,))


def array_to_generator(x, y=None, batch_size=32):
    x = x.astype('float32')
    # Normalizing the images to the range of [0., 1.]
    x /= 255.
    if y is not None:
        data = (x, y)
    else:
        data = x

    return tf.data.Dataset.from_tensor_slices(data).shuffle(x.shape[0]).batch(batch_size)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)




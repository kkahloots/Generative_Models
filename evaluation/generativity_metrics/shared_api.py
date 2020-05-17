from tqdm import tqdm
import numpy as np
import dask.array as da
mean_fn = lambda x: np.mean(da.from_array(np.array(x).reshape((len(x), -1)), chunks=100), axis=0)
sigma_fn = lambda x: np.cov(da.from_array(np.array(x).reshape((len(x), -1)), chunks=100), rowvar=False)

import dask.delayed as delayed
from scipy.spatial import cKDTree

def distance_fn(x, y, metric='l2'):
    l1 = lambda a, b: delayed((a - b).sum(1))
    l2 = lambda a, b: delayed(np.sqrt(((a - b) ** 2).sum(1)))
    metric = l1 if metric=='l1' else l2

    # construct a k-D tree
    tree = cKDTree(x)
    Nm=32
    num_batches = len(y)//Nm
    distances = []
    for batch_ix in range(num_batches):
        # Get next batch for training (4D tensor is returned;
        # our first layer after the input layer will flatten the input)
        batch_y = y[batch_ix * Nm: (batch_ix + 1) * Nm]
        idx = tree.query_ball_point(y, r=4.2, p=2)
        d = metric(batch_y[None, :], x[idx])
        distances += [d.compute()]
    return np.vstack(distances)


def bootstrapping_additive(data_generator, func, stopping_func, tolerance_threshold=1e-6, max_iteration=50):
    data = next(data_generator)
    outputs = [func(data).compute()]
    results = stopping_func(outputs)

    for i, data in tqdm(enumerate(data_generator), leave=False, position=0):
        if i >= max_iteration: break
        outputs += [func(data).compute()]
        new_results = stopping_func(outputs)

        diff_results = np.absolute(results - new_results)
        results = new_results

        if np.max(diff_results) <= tolerance_threshold:
            break

    return results

import tensorflow as tf
def slerp(val, low, high):
    # Val must be Batch_size, n_timesteps
    # low must be batch_size, n_dimensions
    # high must be batch_size, n_dimensions
    if len(low.shape)==1:
        low = np.expand_dims(low, 1)
    if len(high.shape)==1:
        high = np.expand_dims(high, 1)

    low = tf.cast(low, 'float32')
    high = tf.cast(high, 'float32')

    dim_size = low.shape[-1]
    time_steps = 1#val.shape[-1]

    p1 = low / tf.tile(tf.expand_dims(tf.norm(low, axis=1), axis=1), [1, dim_size])
    p2 = high / tf.tile(tf.expand_dims(tf.norm(high, axis=1), axis=1), [1, dim_size])
    dot = tf.reduce_sum(p1 * p2, axis=-1)  # batchwise dot of our Batch*num_dims.

    omega = tf.acos(tf.clip_by_value(dot, -1, 1, ))
    so = tf.sin(omega)
    # if (so == 0):
    # return (1.0-val)*low + val * high
    so = tf.tile(tf.expand_dims(tf.expand_dims(so, axis=1), axis=2), [1, time_steps, dim_size])
    omega = tf.tile(tf.expand_dims(tf.expand_dims(omega, axis=1), axis=2), [1, time_steps, dim_size])
    #val = tf.tile(tf.expand_dims(val, axis=2), [1, 1, dim_size])
    low = tf.tile(tf.expand_dims(low, axis=1), [1, time_steps, 1])
    high = tf.tile(tf.expand_dims(high, axis=1), [1, time_steps, 1])
    lerp = np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high
    return lerp

# def slerp(val, low, high):
#     omega = np.arccos(np.dot(low/np.linalg.norm(low), high.transpose()/np.linalg.norm(high)))
#     so = np.sin(omega)
#     if np.all(so == 0):
#         return (1.0-val) * low + val * high # L'Hopital's rule/LERP
#     print('xxx', 'val',val,  val.shape)
#     print('xxx', 'so', so.shape)
#     print('xxx', 'omega', omega.shape)
#     print('xxx', 'high', high.shape)
#     print('xxx', 'low', low.shape)
#     return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

# def slerp(val, low, high):
#     """Code from https://github.com/soumith/dcgan.torch/issues/14"""
#     omega = np.arccos(np.dot(low / np.linalg.norm(low), (high / np.linalg.norm(high)).transpose()))
#     so = np.max(np.sin(omega),axis=1)
#     # try:
#     #    if omega.shape[1] != low.shape[1]:
#     #        omega = np.hstack([omega, omega])
#     # except:
#     #    pass
#     print('xxx', 'omega', omega.shape)
#     print('xxx', 'so', so.shape)
#     print('xxx', 'high', high.shape)
#     print('xxx', 'low', low.shape)
#
#     if np.all(so == 0):
#         return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
#     return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high
#

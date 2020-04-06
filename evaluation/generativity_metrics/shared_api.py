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

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high.transpose() / np.linalg.norm(high)), -1, 1))
    so = np.max(np.sin(omega))
    try:
        if omega.shape[1] != low.shape[1]:
            omega = np.hstack([omega, omega])
    except:
        pass

    # l1 = lambda low, high, val: (1.0-val) * low + val * high
    # l2 = lambda low, high, val, so, omega: np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
    if np.all(so == 0):
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


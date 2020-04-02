from tqdm import tqdm
import numpy as np
mean_fn = lambda x: np.mean(x, axis=0)
sigma_fn = lambda x: np.cov(np.array(x).reshape((len(x), -1)), rowvar=False)

def bootstrapping_additive(data_generator, func, stopping_func, tolerance_threshold=1e-6, max_iteration=50):
    data = next(data_generator)
    outputs = [func(data)]
    results = stopping_func(outputs)

    for i, data in tqdm(enumerate(data_generator)):
        if i >= max_iteration: break
        outputs += [func(data)]
        new_results = stopping_func(outputs)

        diff_results = np.absolute(results - new_results)
        results = new_results

        if np.max(diff_results) <= tolerance_threshold:
            break

    return results

# Normalize batch of vectors.
def normalize(v):
    return v / np.sqrt(np.sum(np.square(v), axis=-1, keepdims=True))

# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = np.sum(a * b, axis=-1, keepdims=True)

    p = t * np.arccos(d)
    c = normalize(b - d * a)
    d = a * np.cos(p) + c * np.sin(p)
    return normalize(d)
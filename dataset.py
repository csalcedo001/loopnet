import numpy as np

def xor_sampler(n):
    points = np.random.uniform(0, 1, [n, 2])

    d = 0.5
    x = points
    labels = np.logical_or(
        np.logical_and(x[:, 0] > d, x[:, 1] > d),
        np.logical_and(x[:, 0] < d, x[:, 1] < d))
    
    labels = labels.astype(int)

    return points, labels

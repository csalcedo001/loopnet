import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def xor_samples(n):
    points = np.random.uniform(-1, 1, [n, 2])

    x = points
    labels = np.logical_or(
        np.logical_and(x[:, 0] > 0., x[:, 1] > 0.),
        np.logical_and(x[:, 0] < 0., x[:, 1] < 0.))
    
    labels = labels.astype(int)

    return points, labels

x, y = xor_samples(100)

fig = plt.figure()
plt.scatter(x[:,0], x[:,1], c=y)
plt.savefig('scatter.png')
plt.close(fig)

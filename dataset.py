import numpy as np

class XORSampler():
    def __init__(self, batch_size=32, dims=2):
        self.batch_size = batch_size
        self.dims = dims

    def sample(self):
        points = np.random.uniform(0, 1, [self.batch_size, self.dims])
        labels = np.sum(np.round(points), axis=1) % 2

        return points, labels
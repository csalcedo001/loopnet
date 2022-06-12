import matplotlib.pyplot as plt

from dataset import XORSampler


xor_sampler = XORSampler(
    batch_size=100)

x, y = xor_sampler.sample()

fig = plt.figure()
plt.scatter(x[:,0], x[:,1], c=y)
plt.savefig('sample_scatter.png')
plt.close(fig)

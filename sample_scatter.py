import matplotlib.pyplot as plt

from dataset import xor_samples


x, y = xor_samples(100)

fig = plt.figure()
plt.scatter(x[:,0], x[:,1], c=y)
plt.savefig('sample_scatter.png')
plt.close(fig)

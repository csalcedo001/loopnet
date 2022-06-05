import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import xor_sampler
from model import LoopNet, NaiveNet


epochs = 10000
batch_size = 64

n_h = 32
n_l = 5
n_hs = [n_h for _ in range(n_l)]

model = LoopNet(
    n_x=2,
    n_h=sum(n_hs),
    n_y=1,
    loops=n_l + 1)

model = NaiveNet(
    n_x=2,
    n_y=1,
    n_hs=n_hs)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

losses = []
for epoch in tqdm(range(epochs)):
    x, y = xor_sampler(batch_size)

    x = torch.from_numpy(x).float()
    y = torch.unsqueeze(torch.from_numpy(y), axis=1).float()

    y_hat = model(x)
    loss = criterion(y_hat, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

fig = plt.figure()
plt.plot(losses)
plt.savefig('training_loss.png')
plt.close(fig)

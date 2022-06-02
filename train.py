import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import xor_sampler

epochs = 1000
batch_size = 128

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid(),
)

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

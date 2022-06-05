import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import xor_sampler
from model import LoopNet, NaiveNet


epochs = 1000
batch_size = 64

n_h = 32
n_l = 5
n_hs = [n_h for _ in range(n_l)]


model_names = ['loopnet', 'naivenet']
losses = {}
for model_name in model_names:
    if model_name == 'loopnet': 
        model = LoopNet(
            n_x=2,
            n_h=sum(n_hs),
            n_y=1,
            loops=n_l + 1)
    elif model_name == 'naivenet':
        model = NaiveNet(
            n_x=2,
            n_y=1,
            n_hs=n_hs)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    model_losses = []
    for epoch in tqdm(range(epochs)):
        x, y = xor_sampler(batch_size)

        x = torch.from_numpy(x).float()
        y = torch.unsqueeze(torch.from_numpy(y), axis=1).float()

        y_hat = model(x)
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_losses.append(loss.item())
    
    losses[model_name] = model_losses


fig = plt.figure()
plt.title('Training loss')
for model_name in model_names:
    plt.plot(losses[model_name], label=model_name)
plt.ylabel('Loss')
plt.xlabel('Timestep')
plt.legend()
plt.savefig('training_loss.png')
plt.close(fig)

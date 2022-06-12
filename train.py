import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import XORSampler
from model import LoopNet, NaiveNet


epochs = 1000
batch_size = 64
dims=3

n_h = 32
n_l = 5
n_hs = [n_h for _ in range(n_l)]

xor_sampler = XORSampler(
    batch_size=batch_size,
    dims=dims)


model_names = ['loopnet', 'naivenet']
losses = {}
accuracy = {}
for model_name in model_names:
    if model_name == 'loopnet': 
        model = LoopNet(
            n_x=dims,
            n_h=sum(n_hs),
            n_y=1,
            loops=n_l + 1)
    elif model_name == 'naivenet':
        model = NaiveNet(
            n_x=dims,
            n_y=1,
            n_hs=n_hs)
    
    model.train()

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    model_losses = []
    model_accuracy = []
    for epoch in tqdm(range(epochs)):
        x, y = xor_sampler.sample()

        x = torch.from_numpy(x).float()
        y = torch.unsqueeze(torch.from_numpy(y), axis=1).float()

        y_hat = model(x)
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_losses.append(loss.item())

        # Computer accuracy
        acc = torch.sum(torch.round(y_hat) == y)

        model_accuracy.append(acc)
    
    losses[model_name] = model_losses
    accuracy[model_name] = model_accuracy


### Plots

# Plot training loss
fig = plt.figure()
plt.title('Training loss')
for model_name in model_names:
    plt.plot(losses[model_name], label=model_name)
plt.ylabel('Loss')
plt.xlabel('Timestep')
plt.legend()
plt.savefig('training_loss.png')
plt.close(fig)


# Plot training accuracy
fig = plt.figure()
plt.title('Training accuracy')
for model_name in model_names:
    plt.plot(accuracy[model_name], label=model_name)
plt.ylabel('Accuracy')
plt.xlabel('Timestep')
plt.legend()
plt.savefig('training_accuracy.png')
plt.close(fig)

import torch
import torch.nn as nn


class LoopNet(nn.Module):
    def __init__(self, n_x, n_h, n_y, loops):
        super().__init__()

        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.loops = loops

        if loops == 0:
            raise Exception('Error: loops=0 results in y_hat=0.')

        self.n_l = n_x + n_h + n_y
        self.model = nn.Sequential(
            nn.Linear(self.n_l, self.n_l),
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        n = x.shape[0]

        # Create empty observation
        z = torch.zeros(n, self.n_l)

        # Fill in first range of nodes with input
        z[:, :self.n_x] = x

        for _ in range(self.loops - 1):
            z = self.relu(self.model(z))

        z = self.sigmoid(self.model(z))

        return z[:, -self.n_y:]


class NaiveNet(nn.Module):
    def __init__(self, n_x, n_y, n_hs:list):
        super().__init__()
    
        sizes = [n_x] + n_hs + [n_y]
        n_l = len(sizes)

        layers = []
        for i in range(n_l - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        y_hat = self.model(x)
        return y_hat
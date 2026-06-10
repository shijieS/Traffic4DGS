"""OPT-38: Iterative mask refiner with geometric feedback"""
import torch
import torch.nn as nn

class IterativeMaskRefiner(nn.Module):
    def __init__(self, n_iter=3, momentum=0.9):
        super().__init__()
        self.n_iter = n_iter
        self.momentum = momentum
        self.update_net = nn.Sequential(
            nn.Conv2d(5, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )

    def forward(self, mask, rgb, depth, prev_mask=None):
        m = mask
        for _ in range(self.n_iter):
            x = torch.cat([rgb, depth, m], dim=1)
            delta = self.update_net(x)
            m = (self.momentum * m + (1 - self.momentum) * (m + 0.1 * delta)).clamp(0, 1)
        return m

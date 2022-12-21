import torch
from torch import nn


class LayerNorm(nn.Module):
    """Specifically for FashionMNIST"""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.mult = nn.Parameter(torch.tensor(1.0))
        self.add = nn.Parameter(torch.tensor(0.0))
        self.eps = eps

    def forward(self, x):
        # Recall: x.shape == (B x C x H x W)
        m = x.mean((1, 2, 3), keepdim=True)
        v = x.var((1, 2, 3), keepdim=True)
        x = (x - m) / (v + self.eps).sqrt()
        return x * self.mult + self.add

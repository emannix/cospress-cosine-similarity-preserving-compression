
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm

class LinearHead(nn.Module):
    def __init__(
        self,
        in_dim = 192,
        output_dim = 1000,
        **kwargs
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, output_dim, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return x
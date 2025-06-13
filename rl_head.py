import torch
import torch.nn as nn
from settings import RL_ALPHA

class RLHead(nn.Module):
    """
    Policy-head: DynamicBrain’den gelen 2-D vektörü
    µ ∈ [-1,1]^2’lık normal dağılıma çevirir.
    """
    def __init__(self, in_dim=2):
        super().__init__()
        self.mu      = nn.Linear(in_dim, 2)
        self.log_std = nn.Parameter(torch.zeros(2))
        self.opt     = torch.optim.Adam(self.parameters(), lr=RL_ALPHA)

    def forward(self, x):
        mu  = torch.tanh(self.mu(x))
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mtgnn_layers import GraphLearner, MixHopGCN, DilatedInceptionTCN


class STBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_set, dropout):
        super().__init__()
        self.tcn1 = DilatedInceptionTCN(in_channels, hidden_channels, kernel_set, dropout)
        self.gcn = MixHopGCN(hidden_channels, hidden_channels, dropout)
        self.tcn2 = DilatedInceptionTCN(hidden_channels, hidden_channels, kernel_set, dropout)
        self.bn = nn.BatchNorm2d(hidden_channels)

    def forward(self, x, A_hat):
        y = self.tcn1(x)
        y = self.gcn(y, A_hat)
        y = self.tcn2(y)
        return self.bn(F.relu(y) + x)


class MTGNN(nn.Module):
    def __init__(self, num_nodes, in_feat, layers=3, hidden=64,
                 kernel_set=(1, 3, 5, 7), dropout=0.1):
        super().__init__()
        self.graph = GraphLearner(num_nodes)
        self.blocks = nn.ModuleList([
            STBlock(in_feat if i == 0 else hidden, hidden, kernel_set, dropout)
            for i in range(layers)
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Flatten(start_dim=2)
        )

    def forward(self, x):
        # x: (B, N, F, T)
        A_hat = self.graph(x)
        for blk in self.blocks:
            x = blk(x, A_hat)
        out = self.proj(x)[..., -1]
        return out

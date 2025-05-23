import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLearner(nn.Module):
    """Adaptive adjacency learner producing row-normalised graphs."""

    def __init__(self, num_nodes, embed_dim=10, sparsity=0.002, eps=1e-5):
        super().__init__()
        self.num_nodes = num_nodes
        self.eps = eps
        self.sparsity = sparsity
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.nodevec2 = nn.Parameter(torch.randn(embed_dim, num_nodes))

    def forward(self, x):
        # x is only used for batch size
        batch_size = x.size(0)
        a = torch.matmul(self.nodevec1, self.nodevec2)
        a = F.softplus(a)
        a = a / (a.sum(dim=1, keepdim=True) + self.eps)
        if self.training and self.sparsity > 0:
            drop_mask = torch.rand_like(a) > self.sparsity
            a = a * drop_mask
        return a.unsqueeze(0).expand(batch_size, -1, -1)


class MixHopGCN(nn.Module):
    """Mix-hop graph convolution using A and A^2."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.fc = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: (B, N, C, T), adj: (B, N, N)
        x1 = torch.einsum("bnm,bmct->bnct", adj, x)
        adj2 = torch.bmm(adj, adj)
        x2 = torch.einsum("bnm,bmct->bnct", adj2, x)
        out = torch.cat([x1, x2], dim=2)  # concat on feature dim
        out = self.fc(out)
        return self.dropout(out)


class DilatedInceptionTCN(nn.Module):
    """Causal temporal convolution with multiple kernel sizes."""

    def __init__(self, in_channels, out_channels, kernel_set, dropout=0.0):
        super().__init__()
        layers = []
        for k in kernel_set:
            padding = (k - 1)
            layers.append(
                nn.Conv2d(in_channels, out_channels, (1, k),
                           dilation=1, padding=(0, padding))
            )
        self.convs = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (B, N, C, T)
        outputs = []
        for conv in self.convs:
            out = conv(x)
            out = out[..., : x.size(-1)]  # ensure causality
            outputs.append(out)
        out = sum(outputs) / len(outputs)
        out = self.bn(F.relu(out))
        return self.dropout(out)

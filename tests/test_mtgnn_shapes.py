import torch
from model.mtgnn import MTGNN
from model.mtgnn_layers import GraphLearner


def test_graph_learner_shape():
    N = 5
    gl = GraphLearner(N)
    x = torch.randn(2, N, 3, 4)
    out = gl(x)
    assert out.shape == (2, N, N)


def test_mtgnn_forward():
    B, N, F, T = 2, 4, 3, 6
    model = MTGNN(num_nodes=N, in_feat=F, layers=2, hidden=8)
    x = torch.randn(B, N, F, T)
    out = model(x)
    assert out.shape == (B, N)

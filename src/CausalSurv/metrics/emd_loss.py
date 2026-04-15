import ot
import torch
import torch.nn as nn


class EMDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, y):
        a = torch.ones(x.shape[0]) / x.shape[0]
        b = torch.ones(y.shape[0]) / y.shape[0]
        M = ot.dist(x, y)
        ot_dist = ot.bregman.sinkhorn(a, b, M, reg=0.1)

        return ot_dist

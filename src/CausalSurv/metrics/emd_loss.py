import ot
import torch
import torch.nn as nn


class EMDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, y):
        # a/b must live on the same device and dtype as M, which follows x -- POT's torch
        # backend rejects a mix, and this path only became reachable once the IPM group-size
        # gate was fixed, so the mismatch had never surfaced.
        a = torch.full((x.shape[0],), 1.0 / x.shape[0], device=x.device, dtype=x.dtype)
        b = torch.full((y.shape[0],), 1.0 / y.shape[0], device=y.device, dtype=y.dtype)
        M = ot.dist(x, y)
        ot_dist = ot.emd2(a, b, M, log=False)

        return ot_dist

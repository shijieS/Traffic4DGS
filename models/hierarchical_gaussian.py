"""OPT-44: Hierarchical Gaussian with LOD"""
import torch
import torch.nn as nn

class HierarchicalGaussianField(nn.Module):
    def __init__(self, n_levels=3, base_n=50000):
        super().__init__()
        self.levels = nn.ModuleList()
        for l in range(n_levels):
            n = base_n // (4**l) if l > 0 else base_n
            self.levels.append(nn.ModuleDict({
                "pos": nn.Parameter(torch.randn(n, 3) * 5),
                "scale": nn.Parameter(torch.ones(n, 3) * (2**l) * 0.01),
                "opa": nn.Parameter(torch.ones(n, 1) * 0.5),
            }))

    def forward(self, cam_dist=None):
        lvl = min(int(cam_dist / 20), len(self.levels)-1) if cam_dist else 0
        l = self.levels[lvl]
        return l["pos"], l["scale"].softmax(-1), l["opa"].sigmoid()

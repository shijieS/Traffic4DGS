"""OPT-46: View-dependent appearance model"""
import torch
import torch.nn as nn

class ViewDependentAppearance(nn.Module):
    def __init__(self, sh_degree=3, res_dim=32):
        super().__init__()
        self.sh_w = nn.Parameter(torch.randn(1, 9, 3) * 0.01)
        self.res_mlp = nn.Sequential(
            nn.Linear(6, res_dim), nn.ReLU(), nn.Linear(res_dim, 3), nn.Tanh()
        )

    def forward(self, view_dirs, normals=None):
        x, y, z = view_dirs[:,0], view_dirs[:,1], view_dirs[:,2]
        sh = torch.stack([torch.ones_like(x), x, y, z, x*y, x*z, y*z, x**2-y**2, 3*z**2-1], dim=-1)
        color = (sh.unsqueeze(-1) * self.sh_w).sum(1)
        inp = torch.cat([view_dirs, normals if normals is not None else view_dirs], dim=-1)
        return color + self.res_mlp(inp) * 0.1

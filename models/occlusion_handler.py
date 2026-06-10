"""OPT-47: Occlusion-aware rendering"""
import torch
import torch.nn as nn

class OcclusionHandler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s_depth, d_depth, s_rgb, d_rgb, s_opa, d_opa):
        closer = (d_depth < s_depth).float()
        a_d = d_opa * closer
        a_s = s_opa * (1 - a_d)
        total = (a_s + a_d).clamp(min=1e-6)
        return ((a_s.unsqueeze(-1) * s_rgb + a_d.unsqueeze(-1) * d_rgb) / total.unsqueeze(-1)).clamp(0,1)

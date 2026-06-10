"""OPT-35: Depth regularizer"""
import torch
import torch.nn as nn

class DepthRegularizer(nn.Module):
    def __init__(self, depth_w=0.1):
        super().__init__()
        self.depth_w = depth_w

    def forward(self, rendered_depth, mono_depth):
        valid = (rendered_depth > 0) & (mono_depth > 0)
        if not valid.any():
            return torch.tensor(0.0, device=rendered_depth.device)
        rd, md = rendered_depth[valid], mono_depth[valid]
        log_diff = rd.log() - md.log()
        return self.depth_w * ((log_diff**2).mean() - 0.5 * log_diff.mean()**2)

"""OPT-42: Foreground-background separation loss"""
import torch
import torch.nn as nn

class ForegroundBackgroundLoss(nn.Module):
    def __init__(self, sep_w=1.0, overlap_w=2.0):
        super().__init__()
        self.sep_w = sep_w
        self.overlap_w = overlap_w

    def forward(self, static_sem, dynamic_sem, static_opa, dynamic_opa):
        s_fg = static_sem.softmax(-1)[:,:5].sum(-1).mean()
        d_bg = dynamic_sem.softmax(-1)[:,5:].sum(-1).mean()
        overlap = static_opa.mean() * dynamic_opa.mean()
        return self.sep_w * (s_fg + d_bg) + self.overlap_w * overlap

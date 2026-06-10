"""OPT-29: Temporal consistency loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConsistencyLoss(nn.Module):
    def __init__(self, rgb_w=1.0, depth_w=0.5, sem_w=0.3):
        super().__init__()
        self.rgb_w, self.depth_w, self.sem_w = rgb_w, depth_w, sem_w

    def forward(self, out_t, out_t1, flow=None):
        loss = torch.tensor(0.0, device=out_t["rgb"].device)
        loss += self.rgb_w * F.l1_loss(out_t["rgb"], out_t1["rgb"]) * 0.1
        loss += self.depth_w * (out_t["depth"] - out_t1["depth"]).abs().mean()
        loss += self.sem_w * F.kl_div(out_t1["semantic"].softmax(-1).log(), out_t["semantic"].softmax(-1), reduction="batchmean")
        return loss

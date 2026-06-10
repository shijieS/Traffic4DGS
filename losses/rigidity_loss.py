"""OPT-30: Rigidity enforcement loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RigidityLoss(nn.Module):
    def __init__(self, weight=1.0, threshold=0.01):
        super().__init__()
        self.weight = weight
        self.threshold = threshold

    def forward(self, pos_t, pos_t1, obj_ids, rigidity):
        loss = torch.tensor(0.0, device=pos_t.device)
        for oid in range(obj_ids.max().item() + 1):
            m = obj_ids == oid
            if m.sum() < 2:
                continue
            d0 = torch.cdist(pos_t[m], pos_t[m])
            d1 = torch.cdist(pos_t1[m], pos_t1[m])
            violation = F.relu((d1 - d0).abs() - self.threshold)
            loss += rigidity[m].mean() * violation.mean()
        return self.weight * loss / max(obj_ids.max().item() + 1, 1)

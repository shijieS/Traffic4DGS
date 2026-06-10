"""OPT-45: Multi-object tracking loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrackingLoss(nn.Module):
    def __init__(self, det_w=1.0, id_w=0.3):
        super().__init__()
        self.det_w = det_w
        self.id_w = id_w

    def forward(self, pred_boxes, gt_boxes, pred_tracks, gt_tracks):
        T = min(pred_boxes.shape[0], gt_boxes.shape[0])
        N = min(pred_boxes.shape[1], gt_boxes.shape[1])
        loss = self.det_w * F.l1_loss(pred_boxes[:T,:N], gt_boxes[:T,:N])
        for t in range(1, T):
            loss += self.id_w * (pred_tracks[t-1,:N] != pred_tracks[t,:N]).float().mean()
        return loss

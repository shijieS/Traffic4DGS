"""OPT-48: UAV aerial viewpoint extension"""
import torch
import torch.nn as nn

class UAVAerialExtension(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_view = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 6)
        )

    def forward(self, ground_feat, aerial_feat):
        combined = torch.cat([ground_feat.mean(dim=[2,3], keepdim=True).expand_as(aerial_feat), aerial_feat], dim=1)
        return self.cross_view(combined[:, :256])

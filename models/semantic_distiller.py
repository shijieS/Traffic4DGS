"""OPT-26: Semantic feature distiller from SAM2 to Gaussians"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticFeatureDistiller(nn.Module):
    def __init__(self, sam2_dim=256, gauss_dim=64, n_classes=10):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(sam2_dim, 128), nn.ReLU(), nn.Linear(128, gauss_dim))
        self.classifier = nn.Linear(gauss_dim, n_classes)
        self.boundary_head = nn.Sequential(nn.Linear(gauss_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, sam2_feats, ids):
        compact = self.projector(sam2_feats)[ids]
        return compact, self.classifier(compact), self.boundary_head(compact)

"""OPT-43: GIS coordinate aligner"""
import torch
import torch.nn as nn

class GISCoordinateAligner(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.rotation = nn.Parameter(torch.eye(3))
        self.translation = nn.Parameter(torch.zeros(3))

    def forward(self, local_pts):
        R = self._orth(self.rotation)
        return self.scale * (local_pts @ R.T) + self.translation

    def align_loss(self, local_pts, gis_pts, corr):
        transformed = self.forward(local_pts[corr[:,0]])
        return ((transformed - gis_pts[corr[:,1]])**2).mean()

    @staticmethod
    def _orth(R):
        U, _, Vt = torch.linalg.svd(R)
        return U @ Vt

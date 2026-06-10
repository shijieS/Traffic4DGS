"""OPT-34: Pose graph optimizer"""
import torch
import torch.nn as nn

class PoseGraphOptimizer(nn.Module):
    def __init__(self, n_keyframes=100):
        super().__init__()
        self.keyframe_twists = nn.Parameter(torch.zeros(n_keyframes, 6))
        self.constraints = {}

    def add_constraint(self, i, j, rel_T):
        self.constraints[(i,j)] = rel_T

    def forward(self):
        loss = torch.tensor(0.0, device=self.keyframe_twists.device)
        for (i,j), rel_gt in self.constraints.items():
            Ti = self._to_matrix(self.keyframe_twists[i])
            Tj = self._to_matrix(self.keyframe_twists[j])
            rel_est = torch.inverse(Ti) @ Tj
            loss += ((rel_est - rel_gt)**2).sum()
        return loss / max(len(self.constraints), 1)

    def _to_matrix(self, xi):
        o, v = xi[:3], xi[3:]
        th = o.norm().clamp(min=1e-8)
        K = self._skew(o/th)
        R = torch.eye(3, device=xi.device) + torch.sin(th)*K + (1-torch.cos(th))*K@K
        T = torch.eye(4, device=xi.device); T[:3,:3] = R; T[:3,3] = v
        return T

    @staticmethod
    def _skew(v):
        return torch.tensor([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], device=v.device)

"""OPT-24: B-spline SE(3) trajectory representation"""
import torch
import torch.nn as nn

class BSplineTrajectory(nn.Module):
    def __init__(self, n_control=16, degree=3):
        super().__init__()
        self.degree = degree
        self.n_control = n_control
        self.control_twists = nn.Parameter(torch.randn(n_control, 6) * 0.01)

    def basis(self, t, i, k, knots):
        if k == 0:
            return ((t >= knots[i]) & (t < knots[i+1])).float()
        d1 = (knots[i+k] - knots[i]).clamp(min=1e-8)
        d2 = (knots[i+k+1] - knots[i+1]).clamp(min=1e-8)
        return (t - knots[i])/d1 * self.basis(t,i,k-1,knots) + (knots[i+k+1]-t)/d2 * self.basis(t,i+1,k-1,knots)

    def forward(self, t_norm):
        n, d = self.n_control, self.degree
        knots = torch.zeros(n+d+1, device=self.control_twists.device)
        knots[-(d+1):] = 1.0
        for j in range(d+1, n):
            knots[j] = (j - d) / (n - d)
        w = torch.stack([self.basis(t_norm, i, d, knots) for i in range(n)], dim=-1)
        return (w.unsqueeze(-1) * self.control_twists).sum(dim=0)

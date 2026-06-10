"""OPT-20: Weighted SE(3) trajectory smoothing with B-spline basis"""
import torch
import torch.nn as nn


class BSplineTrajectory(nn.Module):
    """B-spline based SE(3) trajectory representation.
    Provides C2 continuous trajectories for rigid body motion,
    superior to linear interpolation for vehicle tracking."""

    def __init__(self, n_control_points=16, degree=3):
        super().__init__()
        self.degree = degree
        self.n_control = n_control_points
        # Learnable SE(3) control points as twists
        self.control_twists = nn.Parameter(
            torch.randn(n_control_points, 6) * 0.01
        )

    def basis_function(self, t, i, k, knots):
        """Cox-de Boor recursion for B-spline basis."""
        if k == 0:
            return ((t >= knots[i]) & (t < knots[i + 1])).float()
        denom1 = (knots[i + k] - knots[i]).clamp(min=1e-8)
        denom2 = (knots[i + k + 1] - knots[i + 1]).clamp(min=1e-8)
        c1 = (t - knots[i]) / denom1 * self.basis_function(t, i, k - 1, knots)
        c2 = (knots[i + k + 1] - t) / denom2 * self.basis_function(t, i + 1, k - 1, knots)
        return c1 + c2

    def forward(self, t_norm):
        """Evaluate SE(3) trajectory at normalized time t_norm in [0, 1].
        Args:
            t_norm: scalar or [B] normalized time
        Returns:
            xi: [6] interpolated twist
        """
        n = self.n_control
        d = self.degree
        # Create uniform knot vector
        knots = torch.zeros(n + d + 1, device=self.control_twists.device)
        knots[:d + 1] = 0.0
        knots[-(d + 1):] = 1.0
        for j in range(d + 1, n):
            knots[j] = (j - d) / (n - d)

        # Compute basis weights
        basis_vals = []
        for i in range(n):
            b = self.basis_function(t_norm, i, d, knots)
            basis_vals.append(b)
        basis = torch.stack(basis_vals, dim=-1)  # [n]

        # Weighted combination of control twists
        xi = (basis.unsqueeze(-1) * self.control_twists).sum(dim=0)  # [6]
        return xi

    def get_trajectory_batch(self, n_samples=64):
        """Sample full trajectory at n_samples time points."""
        t_vals = torch.linspace(0, 1, n_samples, device=self.control_twists.device)
        twists = []
        for t in t_vals:
            twists.append(self.forward(t))
        return torch.stack(twists)  # [n_samples, 6]

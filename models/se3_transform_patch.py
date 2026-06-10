"""OPT-18: SE(3) dual quaternion representation for smoother interpolation"""
import torch
import torch.nn as nn


class DualQuaternionSE3(nn.Module):
    """Dual quaternion representation of SE(3) for seamless rigid body interpolation.
    More stable than SLERP for long sequences with large rotations."""

    def __init__(self):
        super().__init__()

    def se3_to_dual_quaternion(self, xi):
        """Convert SE(3) twist to dual quaternion.
        Args:
            xi: [..., 6] twist (omega, v)
        Returns:
            dq: [..., 8] dual quaternion (qr, qd)
        """
        omega = xi[..., :3]
        v = xi[..., 3:]
        theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        omega_hat = omega / theta

        # Real part (rotation quaternion)
        qr = torch.cat([
            torch.cos(theta / 2),
            omega_hat * torch.sin(theta / 2)
        ], dim=-1)

        # Dual part (translation)
        t_quat = torch.cat([torch.zeros_like(theta), v], dim=-1)
        qd = 0.5 * self._quat_mult(t_quat, qr)
        return torch.cat([qr, qd], dim=-1)

    def _quat_mult(self, q1, q2):
        """Hamilton product of two quaternions."""
        w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
        w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
        return torch.cat([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ], dim=-1)

    def dual_quaternion_interpolate(self, dq1, dq2, t):
        """Screw-linear interpolation (ScLERP) via dual quaternions."""
        qr1, qd1 = dq1[..., :4], dq1[..., 4:8]
        qr2, qd2 = dq2[..., :4], dq2[..., 4:8]
        # Ensure shortest path
        dot = (qr1 * qr2).sum(dim=-1, keepdim=True)
        qr2 = torch.where(dot < 0, -qr2, qr2)
        qd2 = torch.where(dot < 0, -qd2, qd2)
        # Linear interpolation + normalization
        dq_interp = (1 - t) * dq1 + t * dq2
        dq_norm = dq_interp[..., :4].norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return dq_interp / dq_norm

"""OPT-19: Lie algebra exponential map caching for repeated SE(3) computations"""
import torch
import torch.nn as nn
from functools import lru_cache


class ExpMapCache(nn.Module):
    """Cache SE(3) exponential map results to avoid redundant computation.
    Useful when the same rigid body pose is queried across multiple Gaussians."""

    def __init__(self, max_cache_size=1024):
        super().__init__()
        self.cache = {}
        self.max_cache_size = max_cache_size

    def forward(self, xi, object_id=None):
        """Compute or retrieve cached SE(3) exp map.
        Args:
            xi: [6] twist vector
            object_id: optional identifier for cache key
        Returns:
            T: [4, 4] rigid transformation matrix
        """
        if object_id is not None and self.training:
            key = (object_id, xi.data_ptr())
            if key in self.cache:
                return self.cache[key]

        omega = xi[:3]
        v = xi[3:]
        theta = omega.norm().clamp(min=1e-8)
        omega_hat = omega / theta
        K = self._skew(omega_hat)

        R = torch.eye(3, device=xi.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        t = (torch.eye(3, device=xi.device) - R) @ (omega.cross(v) / (theta ** 2)) + v * theta / theta

        T = torch.eye(4, device=xi.device)
        T[:3, :3] = R
        T[:3, 3] = t

        if object_id is not None and self.training:
            if len(self.cache) >= self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = T

        return T

    @staticmethod
    def _skew(v):
        return torch.zeros(3, 3, device=v.device).indexed_fill(
            [0, 1, 2, 0, 1, 2],
            [1, 2, 0, 2, 0, 1],
            [-v[2], -v[0], -v[1], v[1], v[2], v[0]]
        )

    def clear_cache(self):
        self.cache.clear()

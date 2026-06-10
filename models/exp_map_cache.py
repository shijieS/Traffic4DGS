"""OPT-23: Lie algebra exp map caching"""
import torch
import torch.nn as nn

class ExpMapCache(nn.Module):
    def __init__(self, max_size=1024):
        super().__init__()
        self.cache = {}
        self.max_size = max_size

    def forward(self, xi, obj_id=None):
        if obj_id is not None and self.training:
            key = (obj_id, round(xi[0].item(), 4))
            if key in self.cache:
                return self.cache[key]
        omega, v = xi[:3], xi[3:]
        theta = omega.norm().clamp(min=1e-8)
        K = self._skew(omega / theta)
        R = torch.eye(3, device=xi.device) + torch.sin(theta)*K + (1-torch.cos(theta))*(K@K)
        t = (torch.eye(3, device=xi.device) - R) @ (omega.cross(v)/(theta**2)) + v
        T = torch.eye(4, device=xi.device)
        T[:3,:3] = R
        T[:3,3] = t
        if obj_id is not None and self.training:
            if len(self.cache) >= self.max_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = T
        return T

    @staticmethod
    def _skew(v):
        return torch.tensor([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], device=v.device)

    def clear(self):
        self.cache.clear()

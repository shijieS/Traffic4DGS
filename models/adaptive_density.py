"""OPT-25: Adaptive Gaussian density controller"""
import torch
import torch.nn as nn

class AdaptiveGaussianDensity(nn.Module):
    def __init__(self, split_th=0.0002, prune_th=0.005, max_gaussians=2_000_000):
        super().__init__()
        self.split_th = split_th
        self.prune_th = prune_th
        self.max_gaussians = max_gaussians
        self.grad_accum = None

    def update_grad(self, positions, grads):
        if self.grad_accum is None:
            self.grad_accum = torch.zeros(positions.shape[0], device=positions.device)
        self.grad_accum += grads.norm(dim=-1).detach()

    def densify_and_prune(self, positions, scales, opacities, sem_weights=None):
        if self.grad_accum is None:
            return positions, scales, opacities
        avg = self.grad_accum / self.grad_accum.clamp(min=1)
        th = self.split_th / (1 + sem_weights.max(dim=-1).values) if sem_weights is not None else self.split_th
        split_mask = avg > th
        prune_mask = opacities.squeeze(-1) < self.prune_th
        n_split = split_mask.sum().item()
        if n_split > 0 and positions.shape[0] + n_split < self.max_gaussians:
            positions = torch.cat([positions, positions[split_mask]])
            scales = torch.cat([scales, scales[split_mask] * 0.5])
            opacities = torch.cat([opacities, opacities[split_mask]])
        keep = ~prune_mask
        self.grad_accum = None
        return positions[keep], scales[keep], opacities[keep]

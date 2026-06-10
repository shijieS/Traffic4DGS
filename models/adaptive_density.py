"""OPT-21: Multi-resolution Gaussian field with adaptive density control"""
import torch
import torch.nn as nn


class AdaptiveGaussianDensity(nn.Module):
    """Adaptive Gaussian density controller inspired by 3DGS densification.
    Splits/prunes Gaussians based on gradient statistics and geometric complexity.
    Extended for semantic-aware density: densify more around object boundaries."""

    def __init__(self, split_threshold=0.0002, prune_threshold=0.005,
                 max_gaussians=2_000_000, densify_interval=100):
        super().__init__()
        self.split_threshold = split_threshold
        self.prune_threshold = prune_threshold
        self.max_gaussians = max_gaussians
        self.densify_interval = densify_interval
        self.grad_accum = None
        self.step_count = 0

    def update_grad_stats(self, positions, grads):
        """Accumulate positional gradient statistics for density control.
        Args:
            positions: [N, 3] Gaussian positions
            grads: [N, 3] gradients w.r.t. positions
        """
        if self.grad_accum is None:
            self.grad_accum = torch.zeros(positions.shape[0], device=positions.device)
            self.grad_count = torch.zeros(positions.shape[0], device=positions.device)

        grad_norm = grads.norm(dim=-1)
        self.grad_accum += grad_norm.detach()
        self.grad_count += 1

    def should_densify(self, step):
        """Check if densification should occur at current step."""
        self.step_count = step
        return step > 0 and step % self.densify_interval == 0

    def densify_and_prune(self, positions, scales, opacities, semantic_weights=None):
        """Perform adaptive densification and pruning.
        Args:
            positions: [N, 3]
            scales: [N, 3]
            opacities: [N, 1]
            semantic_weights: [N, C] optional semantic importance weights
        Returns:
            Updated positions, scales, opacities
        """
        if self.grad_accum is None:
            return positions, scales, opacities

        avg_grad = self.grad_accum / self.grad_count.clamp(min=1)

        # Semantic-aware threshold adjustment
        if semantic_weights is not None:
            semantic_importance = semantic_weights.max(dim=-1).values
            threshold = self.split_threshold / (1 + semantic_importance)
        else:
            threshold = self.split_threshold

        # Split: high gradient Gaussians
        split_mask = avg_grad > threshold
        # Prune: low opacity Gaussians
        prune_mask = opacities.squeeze(-1) < self.prune_threshold

        # Clone high-gradient Gaussians
        n_split = split_mask.sum().item()
        if n_split > 0 and positions.shape[0] + n_split < self.max_gaussians:
            split_pos = positions[split_mask]
            split_scales = scales[split_mask] * 0.5
            split_opacities = opacities[split_mask]
            positions = torch.cat([positions, split_pos], dim=0)
            scales = torch.cat([scales, split_scales], dim=0)
            opacities = torch.cat([opacities, split_opacities], dim=0)

        # Prune low-opacity Gaussians
        keep_mask = ~prune_mask
        positions = positions[keep_mask]
        scales = scales[keep_mask]
        opacities = opacities[keep_mask]

        # Reset accumulators
        self.grad_accum = None
        self.grad_count = None

        return positions, scales, opacities

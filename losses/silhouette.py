"""
Silhouette and Edge Loss Functions.

This module implements silhouette-aware loss functions for
better boundary rendering in 4D Gaussian Splatting.

Mathematical Foundation:
    Silhouette Loss:
        L_silhouette = ||∇α - ∇α*||₁
        
    where ∇ is the gradient operator and α is the rendered alpha.
    
    Edge-aware Loss:
        L_edge = ||w · (E - E*)||₁
        
    where E is the edge map and w is edge confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SilhouetteLoss(nn.Module):
    """Silhouette and edge loss for boundary rendering.
    
    Computes loss based on alpha channel edges.
    """
    
    def __init__(
        self,
        edge_weight: float = 1.0,
        smooth_weight: float = 0.1,
        use_canny: bool = True,
    ) -> None:
        """Initialize silhouette loss.
        
        Args:
            edge_weight: Weight for edge consistency
            smooth_weight: Weight for smoothness
            use_canny: Use Canny edge detector
        """
        super().__init__()
        self.edge_weight = edge_weight
        self.smooth_weight = smooth_weight
        self.use_canny = use_canny
    
    def forward(
        self,
        rendered_alpha: torch.Tensor,
        target_silhouette: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute silhouette loss.
        
        Args:
            rendered_alpha: Rendered alpha [B, H, W] or [H, W]
            target_silhouette: Target silhouette [B, H, W] or [H, W]
            mask: Optional validity mask
            
        Returns:
            loss: Silhouette loss
            info: Loss breakdown
        """
        if rendered_alpha.dim() == 2:
            rendered_alpha = rendered_alpha.unsqueeze(0)
        if target_silhouette.dim() == 2:
            target_silhouette = target_silhouette.unsqueeze(0)
        
        B, H, W = rendered_alpha.shape
        
        # Compute rendered edges
        gy = torch.diff(rendered_alpha, dim=1)
        gx = torch.diff(rendered_alpha, dim=2)
        
        rendered_edge = torch.sqrt(
            F.pad(gy, (0, 0, 0, 1)) ** 2 +
            F.pad(gx, (0, 1, 0, 0)) ** 2
        )
        
        # Normalize
        rendered_edge = rendered_edge / (rendered_edge.max() + 1e-8)
        
        # Edge consistency loss
        diff = rendered_edge - target_silhouette
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            # Dilate mask for edge region
            mask_edge = F.max_pool2d(
                mask.float().unsqueeze(1),
                kernel_size=5, stride=1, padding=2
            ).squeeze(1)
            diff = diff * mask_edge
        
        loss_edge = torch.abs(diff).mean()
        
        # Smoothness loss (TV)
        loss_smooth = self._total_variation(rendered_alpha)
        
        loss = self.edge_weight * loss_edge + self.smooth_weight * loss_smooth
        
        return loss, {'edge': loss_edge.item(), 'smooth': loss_smooth.item()}
    
    def _total_variation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss."""
        gy = torch.diff(x, dim=1)
        gx = torch.diff(x, dim=2)
        return torch.mean(torch.abs(gy)) + torch.mean(torch.abs(gx))


class MultiScaleSilhouetteLoss(nn.Module):
    """Multi-scale silhouette loss for finer edge details."""
    
    def __init__(
        self,
        scales: Tuple[int, ...] = (1, 2, 4),
        weights: Tuple[float, ...] = (1.0, 0.5, 0.25),
    ) -> None:
        """Initialize multi-scale silhouette loss.
        
        Args:
            scales: Downsampling factors
            weights: Loss weights per scale
        """
        super().__init__()
        self.scales = scales
        self.weights = weights
        self.base_loss = SilhouetteLoss()
    
    def forward(
        self,
        rendered_alpha: torch.Tensor,
        target_silhouette: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-scale silhouette loss."""
        total_loss = 0.0
        info = {}
        
        for scale, weight in zip(self.scales, self.weights):
            if scale > 1:
                # Downsample
                rendered_scaled = F.avg_pool2d(
                    rendered_alpha.unsqueeze(1),
                    kernel_size=scale,
                    stride=scale
                ).squeeze(1)
                target_scaled = F.avg_pool2d(
                    target_silhouette.unsqueeze(1).float(),
                    kernel_size=scale,
                    stride=scale
                ).squeeze(1)
                mask_scaled = F.avg_pool2d(
                    mask.unsqueeze(1).float(),
                    kernel_size=scale,
                    stride=scale
                ).squeeze(1) > 0.5 if mask is not None else None
            else:
                rendered_scaled = rendered_alpha
                target_scaled = target_silhouette
                mask_scaled = mask
            
            loss_scale, info_scale = self.base_loss(
                rendered_scaled, target_scaled, mask_scaled
            )
            total_loss = total_loss + weight * loss_scale
            info[f'edge_{scale}x'] = info_scale.get('edge', 0)
        
        info['multi_scale'] = total_loss.item()
        return total_loss, info


class DepthSilhouetteConsistencyLoss(nn.Module):
    """Combined depth and silhouette consistency loss.
    
    Ensures depth discontinuities align with silhouette edges.
    """
    
    def __init__(
        self,
        depth_weight: float = 1.0,
        silhouette_weight: float = 1.0,
        consistency_weight: float = 0.5,
    ) -> None:
        """Initialize depth-silhouette consistency loss.
        
        Args:
            depth_weight: Weight for depth loss
            silhouette_weight: Weight for silhouette loss
            consistency_weight: Weight for consistency term
        """
        super().__init__()
        self.depth_weight = depth_weight
        self.silhouette_weight = silhouette_weight
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        rendered_depth: torch.Tensor,
        target_depth: torch.Tensor,
        rendered_alpha: torch.Tensor,
        target_silhouette: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute depth-silhouette consistency loss."""
        info = {}
        losses = []
        
        # Depth L1 loss
        if self.depth_weight > 0:
            diff_depth = torch.abs(rendered_depth - target_depth)
            if mask is not None:
                diff_depth = diff_depth * mask
            loss_depth = diff_depth.mean()
            losses.append(self.depth_weight * loss_depth)
            info['depth'] = loss_depth.item()
        
        # Silhouette loss
        if self.silhouette_weight > 0:
            gy = torch.diff(rendered_alpha, dim=0)
            gx = torch.diff(rendered_alpha, dim=1)
            rendered_edge = torch.sqrt(
                F.pad(gy, (0, 0, 0, 1)) ** 2 +
                F.pad(gx, (0, 1, 0, 0)) ** 2
            )
            diff_sil = torch.abs(rendered_edge - target_silhouette)
            if mask is not None:
                diff_sil = diff_sil * F.max_pool2d(
                    mask.float().unsqueeze(0),
                    kernel_size=3, stride=1, padding=1
                ).squeeze(0)
            loss_sil = diff_sil.mean()
            losses.append(self.silhouette_weight * loss_sil)
            info['silhouette'] = loss_sil.item()
        
        # Consistency loss: depth edges should match silhouette edges
        if self.consistency_weight > 0:
            depth_gy = torch.abs(torch.diff(rendered_depth, dim=0))
            depth_gx = torch.abs(torch.diff(rendered_depth, dim=1))
            depth_edge = depth_gy[:-1, :] + depth_gy[1:, :] + \
                         depth_gx[:, :-1] + depth_gx[:, 1:]
            
            # Silhouette edges
            sil_gy = torch.abs(torch.diff(target_silhouette, dim=0))
            sil_gx = torch.abs(torch.diff(target_silhouette, dim=1))
            sil_edge = sil_gy[:-1, :] + sil_gy[1:, :] + \
                      sil_gx[:, :-1] + sil_gx[:, 1:]
            
            # Consistency: both should have edges at same locations
            loss_consistency = torch.abs(depth_edge - sil_edge).mean()
            losses.append(self.consistency_weight * loss_consistency)
            info['consistency'] = loss_consistency.item()
        
        total_loss = sum(losses) if losses else torch.tensor(0.0)
        return total_loss, info

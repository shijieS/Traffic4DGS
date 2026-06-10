"""
Photometric Loss Functions.

This module implements photometric loss functions for 4D Gaussian Splatting,
including L1, L2, SSIM, and perceptual losses.

Mathematical Foundation:
    L1 Loss:
        L₁(I, Ī) = |I - Ī|₁
        
    L2 Loss (MSE):
        L₂(I, Ī) = ||I - Ī||²₂
        
    SSIM (Structural Similarity):
        SSIM(x, y) = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / (μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂)
        
    Perceptual Loss (LPIPS):
        Uses pretrained network features:
        L_percep = Σₗ ||φₗ(I) - φₗ(Ī)||₂
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable
from torchvision import models


class PhotometricLoss(nn.Module):
    """Combined photometric loss with multiple terms.
    
    Loss = λ₁L₁ + λ₂L₂ + λ_ssim SSIM + λ_perc LPIPS
    """
    
    def __init__(
        self,
        weight_l1: float = 0.8,
        weight_l2: float = 0.2,
        weight_ssim: float = 0.0,
        weight_perceptual: float = 0.0,
        use_mask: bool = False,
        reduction: str = "mean",
    ) -> None:
        """Initialize photometric loss.
        
        Args:
            weight_l1: Weight for L1 loss
            weight_l2: Weight for L2 loss
            weight_ssim: Weight for SSIM loss
            weight_perceptual: Weight for perceptual loss
            use_mask: Apply masking for valid pixels
            reduction: Loss reduction method
        """
        super().__init__()
        
        self.weight_l1 = weight_l1
        self.weight_l2 = weight_l2
        self.weight_ssim = weight_ssim
        self.weight_perceptual = weight_perceptual
        self.use_mask = use_mask
        self.reduction = reduction
        
        # SSIM calculator
        if weight_ssim > 0:
            self.ssim = SSIMLoss()
        
        # Perceptual loss network
        if weight_perceptual > 0:
            self.perceptual = LPIPSLoss()
    
    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute photometric loss.
        
        Args:
            rendered: Rendered image [B, 3, H, W] or [3, H, W]
            target: Target image [B, 3, H, W] or [3, H, W]
            mask: Optional validity mask [B, H, W] or [H, W]
            
        Returns:
            loss: Total photometric loss
            info: Dictionary of individual loss terms
        """
        # Ensure batch dimension
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        
        info = {}
        losses = {}
        
        # Apply mask if provided
        if mask is not None and self.use_mask:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        
        def apply_mask(loss_tensor):
            if mask is not None:
                return (loss_tensor * mask).sum() / (mask.sum() + 1e-8)
            return loss_tensor.mean()
        
        # L1 loss
        if self.weight_l1 > 0:
            loss_l1 = torch.abs(rendered - target)
            loss_l1 = apply_mask(loss_l1)
            losses['l1'] = loss_l1 * self.weight_l1
            info['l1'] = loss_l1.item()
        
        # L2 loss
        if self.weight_l2 > 0:
            loss_l2 = ((rendered - target) ** 2)
            loss_l2 = apply_mask(loss_l2)
            losses['l2'] = loss_l2 * self.weight_l2
            info['l2'] = loss_l2.item()
        
        # SSIM loss
        if self.weight_ssim > 0:
            loss_ssim = 1 - self.ssim(rendered, target)
            loss_ssim = apply_mask(loss_ssim)
            losses['ssim'] = loss_ssim * self.weight_ssim
            info['ssim'] = loss_ssim.item()
        
        # Perceptual loss
        if self.weight_perceptual > 0:
            loss_perc = self.perceptual(rendered, target)
            loss_perc = apply_mask(loss_perc)
            losses['perceptual'] = loss_perc * self.weight_perceptual
            info['perceptual'] = loss_perc.item()
        
        total_loss = sum(losses.values())
        
        return total_loss, info


class SSIMLoss(nn.Module):
    """SSIM (Structural Similarity Index) Loss.
    
    SSIM measures perceptual difference between two images based on:
    - Luminance: μ
    - Contrast: σ
    - Structure: σₓᵧ
    """
    
    def __init__(
        self,
        window_size: int = 11,
        reduction: str = "mean",
    ) -> None:
        """Initialize SSIM loss.
        
        Args:
            window_size: Size of Gaussian window
            reduction: Loss reduction method
        """
        super().__init__()
        self.window_size = window_size
        self.reduction = reduction
        
        # Constants for stability
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        
        # Create Gaussian window
        self.register_buffer(
            'window',
            self._create_gaussian_window(window_size, 1.5)
        )
    
    def _create_gaussian_window(
        self,
        window_size: int,
        sigma: float
    ) -> torch.Tensor:
        """Create 2D Gaussian window."""
        import math
        
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window / window.sum()
        
        return window
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SSIM between two images.
        
        Args:
            img1: First image [B, C, H, W]
            img2: Second image [B, C, H, W]
            
        Returns:
            SSIM value
        """
        # Ensure channel dimension for window application
        channel = img1.shape[1]
        
        # Compute local statistics
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 ** 2, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        if self.reduction == "mean":
            return ssim_map.mean()
        elif self.reduction == "sum":
            return ssim_map.sum()
        else:
            return ssim_map


class LPIPSLoss(nn.Module):
    """LPIPS (Learned Perceptual Image Patch Similarity) Loss.
    
    Uses pretrained network (VGG, etc.) to compute perceptual similarity.
    """
    
    def __init__(
        self,
        network: str = "vgg",
        reduction: str = "mean",
    ) -> None:
        """Initialize LPIPS loss.
        
        Args:
            network: Feature network ('vgg', 'alex', 'squeeze')
            reduction: Loss reduction method
        """
        super().__init__()
        self.reduction = reduction
        
        # Load pretrained network
        if network == "vgg":
            self.net = models.vgg16(pretrained=True).features
        elif network == "alex":
            self.net = models.alexnet(pretrained=True).features
        else:
            self.net = models.squeezenet1_1(pretrained=True).features
        
        # Freeze parameters
        for param in self.net.parameters():
            param.requires_grad = False
        
        # Normalize to [-1, 1]
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LPIPS between two images.
        
        Args:
            img1: First image [B, 3, H, W]
            img2: Second image [B, 3, H, W]
            
        Returns:
            LPIPS distance
        """
        # Normalize
        img1 = (img1 - self.mean) / self.std
        img2 = (img2 - self.mean) / self.std
        
        # Extract features
        feat1 = self.net(img1)
        feat2 = self.net(img2)
        
        # Compute normalized distance
        diff = (feat1 - feat2) ** 2
        distance = diff.mean(dim=(1, 2, 3))
        
        if self.reduction == "mean":
            return distance.mean()
        elif self.reduction == "sum":
            return distance.sum()
        else:
            return distance


class DepthPhotometricLoss(nn.Module):
    """Photometric loss with depth-guided masking.
    
    Uses depth consistency to weight the photometric loss.
    """
    
    def __init__(
        self,
        depth_weight: float = 0.1,
        angle_weight: float = 0.05,
    ) -> None:
        """Initialize depth photometric loss.
        
        Args:
            depth_weight: Weight for depth consistency term
            angle_weight: Weight for angle consistency term
        """
        super().__init__()
        self.depth_weight = depth_weight
        self.angle_weight = angle_weight
    
    def forward(
        self,
        rendered_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        rendered_depth: torch.Tensor,
        target_depth: torch.Tensor,
        rendered_normal: Optional[torch.Tensor] = None,
        target_normal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute depth-guided photometric loss.
        
        Args:
            rendered_rgb: Rendered RGB [B, 3, H, W]
            target_rgb: Target RGB [B, 3, H, W]
            rendered_depth: Rendered depth [B, H, W]
            target_depth: Target depth [B, H, W]
            rendered_normal: Optional rendered normals [B, 3, H, W]
            target_normal: Optional target normals [B, 3, H, W]
            
        Returns:
            loss: Total loss
            info: Loss breakdown
        """
        # RGB photometric loss
        loss_rgb = F.l1_loss(rendered_rgb, target_rgb)
        
        info = {'rgb': loss_rgb.item()}
        total_loss = loss_rgb
        
        # Depth consistency
        if self.depth_weight > 0:
            depth_diff = torch.abs(rendered_depth - target_depth)
            # Weight by validity (non-zero depth)
            valid_mask = (target_depth > 0).float()
            loss_depth = (depth_diff * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            total_loss = total_loss + self.depth_weight * loss_depth
            info['depth'] = loss_depth.item()
        
        # Normal consistency
        if self.angle_weight > 0 and rendered_normal is not None and target_normal is not None:
            # Cosine similarity
            cos_sim = F.cosine_similarity(rendered_normal, target_normal, dim=1)
            loss_normal = (1 - cos_sim).mean()
            total_loss = total_loss + self.angle_weight * loss_normal
            info['normal'] = loss_normal.item()
        
        return total_loss, info

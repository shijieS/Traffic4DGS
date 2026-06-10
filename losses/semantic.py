"""
Semantic Feature Rendering Loss.

This module implements semantic loss functions for instance-aware
semantic segmentation in 4D Gaussian Splatting.

Mathematical Foundation:
    Cross-Entropy Loss:
        L_CE = -Σᵢ yᵢ log(σ(zᵢ))
        
    where σ is softmax and z is the logit.
    
    Focal Loss (for class imbalance):
        L_focal = -αₜ(1 - pₜ)ᵞ log(pₜ)
        
    Instance-Aware Loss:
        L_instance = Σ_{c ∈ classes} w_c · L_CE(c)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class SemanticLoss(nn.Module):
    """Semantic segmentation loss with class weighting.
    
    Supports:
    - Cross-entropy loss
    - Focal loss for class imbalance
    - Instance-aware weighting
    """
    
    def __init__(
        self,
        num_classes: int = 23,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        use_focal: bool = False,
        reduction: str = "mean",
    ) -> None:
        """Initialize semantic loss.
        
        Args:
            num_classes: Number of semantic classes
            weight: Class weights [C]
            ignore_index: Index to ignore in loss
            focal_gamma: Focal loss gamma parameter
            focal_alpha: Focal loss alpha parameter
            use_focal: Use focal loss
            reduction: Loss reduction method
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.use_focal = use_focal
        self.reduction = reduction
        
        if weight is not None:
            self.register_buffer('class_weight', weight)
        else:
            self.register_buffer('class_weight', torch.ones(num_classes))
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute semantic loss.
        
        Args:
            logits: Predicted logits [B, C, H, W] or [C, H, W]
            targets: Target class indices [B, H, W] or [H, W]
            mask: Optional validity mask [B, H, W] or [H, W]
            
        Returns:
            loss: Semantic loss
            info: Loss breakdown
        """
        # Ensure batch dimension
        if logits.dim() == 3:
            logits = logits.unsqueeze(0)
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        B, C, H, W = logits.shape
        
        # Flatten spatial dimensions
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        targets_flat = targets.reshape(-1)  # [B*H*W]
        
        if mask is not None:
            mask_flat = mask.reshape(-1)  # [B*H*W]
        else:
            mask_flat = torch.ones_like(targets_flat, dtype=torch.bool)
        
        # Ignore specified indices
        valid_mask = (targets_flat != self.ignore_index) & mask_flat
        
        logits_flat = logits_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]
        
        if len(targets_flat) == 0:
            return torch.tensor(0.0, device=logits.device), {'semantic': 0.0}
        
        # Compute loss
        if self.use_focal:
            loss = self._focal_loss(logits_flat, targets_flat)
        else:
            loss = F.cross_entropy(
                logits_flat, targets_flat,
                weight=self.class_weight,
                reduction=self.reduction
            )
        
        # Per-class accuracy
        with torch.no_grad():
            preds = torch.argmax(logits_flat, dim=-1)
            accuracy = (preds == targets_flat).float().mean()
        
        return loss, {'semantic': loss.item(), 'accuracy': accuracy.item()}
    
    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss for class imbalance.
        
        Args:
            logits: [N, C] logits
            targets: [N] class indices
            
        Returns:
            focal loss
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        focal_loss = self.focal_alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


class InstanceAwareSemanticLoss(nn.Module):
    """Instance-aware semantic loss.
    
    Computes semantic loss with per-instance weighting
    for better handling of object instances.
    """
    
    def __init__(
        self,
        num_classes: int = 23,
        instance_weight: float = 1.0,
        background_weight: float = 0.5,
    ) -> None:
        """Initialize instance-aware semantic loss.
        
        Args:
            num_classes: Number of semantic classes
            instance_weight: Weight for instance classes
            background_weight: Weight for background classes
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.instance_weight = instance_weight
        self.background_weight = background_weight
        
        # Background class indices
        self.background_classes = {0, 10, 11, 12, 13, 14, 15}  # unlabeled, sky, vegetation, etc.
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        instance_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute instance-aware semantic loss.
        
        Args:
            logits: Predicted logits [B, C, H, W]
            targets: Target class indices [B, H, W]
            instance_ids: Instance ID map [B, H, W]
            mask: Optional validity mask [B, H, W]
            
        Returns:
            loss: Instance-aware semantic loss
            info: Loss breakdown
        """
        if logits.dim() == 3:
            logits = logits.unsqueeze(0)
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)
        
        # Base semantic loss
        base_loss = F.cross_entropy(
            logits, targets,
            reduction='none',
            ignore_index=-100
        )
        
        # Apply instance weighting
        if instance_ids is not None:
            instance_mask = instance_ids > 0  # Non-zero = object instance
            weight_map = torch.where(
                instance_mask,
                torch.tensor(self.instance_weight, device=logits.device),
                torch.tensor(self.background_weight, device=logits.device)
            )
            loss = (base_loss * weight_map).mean()
        else:
            loss = base_loss.mean()
        
        # Per-class accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == targets).float().mean()
        
        return loss, {'instance_semantic': loss.item(), 'accuracy': accuracy.item()}


class SemanticFeatureContrastiveLoss(nn.Module):
    """Contrastive loss for semantic feature learning.
    
    Encourages features of the same class to be close,
    and different classes to be far apart.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        num_classes: int = 23,
    ) -> None:
        """Initialize contrastive loss.
        
        Args:
            temperature: Temperature for softmax
            num_classes: Number of semantic classes
        """
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute contrastive loss.
        
        Args:
            features: Semantic features [B, D, H, W]
            labels: Class labels [B, H, W]
            mask: Optional validity mask [B, H, W]
            
        Returns:
            loss: Contrastive loss
            info: Loss info
        """
        if features.dim() == 3:
            features = features.unsqueeze(0)
            labels = labels.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        B, D, H, W = features.shape
        
        # Flatten
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, D)  # [BHW, D]
        labels_flat = labels.reshape(-1)  # [BHW]
        
        if mask is not None:
            mask_flat = mask.reshape(-1)
            features_flat = features_flat[mask_flat]
            labels_flat = labels_flat[mask_flat]
        
        # Normalize features
        features_flat = F.normalize(features_flat, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features_flat, features_flat.T) / self.temperature
        
        # Create positive/negative masks
        labels_equal = labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)
        
        # Numerical stability
        logits_max = sim_matrix.max(dim=-1, keepdim=True)[0]
        logits = sim_matrix - logits_max
        
        # Exponentiate
        exp_logits = torch.exp(logits)
        
        # Loss
        exp_logits_sum = exp_logits.sum(dim=-1, keepdim=True)
        log_prob = logits - torch.log(exp_logits_sum + 1e-8)
        
        # Mean log-likelihood
        loss = -log_prob.mean()
        
        return loss, {'contrastive': loss.item()}

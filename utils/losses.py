"""
Semantic-4DGS-Traffic: Combined Loss Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class CombinedLoss(nn.Module):
    """
    Combined loss for Semantic-4DGS-Traffic training.
    """
    
    def __init__(self, loss_config: Dict):
        super().__init__()
        self.config = loss_config
        
        # Loss weights
        self.rgb_weight = loss_config.get("rgb_weight", 1.0)
        self.depth_weight = loss_config.get("depth_weight", 0.5)
        self.semantic_weight = loss_config.get("semantic_weight", 0.3)
        self.silhouette_weight = loss_config.get("silhouette_weight", 0.2)
        
    def rgb_loss(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
    ) -> torch.Tensor:
        """RGB rendering loss (L1 + SSIM)"""
        loss = torch.tensor(0.0, device=predictions[0].device)
        
        for pred, target in zip(predictions, targets):
            # L1 loss
            l1 = F.l1_loss(pred, target)
            
            # SSIM loss (simplified)
            ssim = self._ssim(pred, target)
            ssim_loss = 1 - ssim
            
            loss = loss + l1 + 0.1 * ssim_loss
            
        return loss / len(predictions)
        
    def depth_loss(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Depth rendering loss"""
        loss = torch.tensor(0.0, device=predictions[0].device)
        
        for pred in predictions:
            # L1 on valid depths
            mask = targets > 0
            if mask.sum() > 0:
                l1 = F.l1_loss(pred[mask], targets[mask])
                loss = loss + l1
                
        return loss / len(predictions) if len(predictions) > 0 else loss
        
    def semantic_loss(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Semantic segmentation loss (cross-entropy)"""
        loss = torch.tensor(0.0, device=predictions[0].device)
        
        for pred in predictions:
            # Flatten spatial dimensions
            pred_flat = pred.permute(1, 0, 2, 3).reshape(-1, pred.shape[0])
            target_flat = targets.reshape(-1)
            
            # Cross-entropy
            ce = F.cross_entropy(pred_flat, target_flat)
            loss = loss + ce
            
        return loss / len(predictions) if len(predictions) > 0 else loss
        
    def silhouette_loss(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
    ) -> torch.Tensor:
        """Silhouette/binary segmentation loss"""
        if len(predictions) == 0 or len(targets) == 0:
            return torch.tensor(0.0)
            
        loss = torch.tensor(0.0, device=predictions[0].device)
        
        for pred, target in zip(predictions, targets):
            if pred.shape != target.shape:
                continue
            # Binary cross-entropy
            bce = F.binary_cross_entropy(pred, target.float())
            loss = loss + bce
            
        return loss / len(predictions)
        
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
    ) -> torch.Tensor:
        """Simplified SSIM calculation"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
                   
        return ssim_map.mean()

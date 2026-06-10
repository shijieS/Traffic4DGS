"""
Joint Loss Module with Uncertainty Weighting.

Implements automatic loss weight balancing using learned uncertainty
for multi-task learning in Semantic 4D Gaussian Splatting.

Mathematical Foundation:
    Uncertainty Weighting (Kendall et al., 2018):
        L_joint = Σ_i (1 / (2σᵢ²)) Lᵢ + log(σᵢ)
        
    where σᵢ is the learned uncertainty for task i.
    
    Progressive SE(3) Activation:
        α(t) = min(1, t / T_warmup) · sigmoid((t - T_start) / τ)
        
    Trajectory Consistency:
        L_traj = ||p_4dgs(t) - p_track(t)||²
        
@author Semantic 4DGS Team
@version 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Callable
import math


class UncertaintyWeightedLoss(nn.Module):
    """Joint loss with automatic uncertainty-based weight balancing.
    
    This module implements the uncertainty weighting approach from
    "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018).
    
    Each task has a learnable log variance parameter σ:
        - Larger σ² → smaller weight → easier task
        - Smaller σ² → larger weight → harder task
    """
    
    def __init__(
        self,
        num_losses: int,
        init_log_vars: Optional[List[float]] = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize uncertainty-weighted loss.
        
        Args:
            num_losses: Number of loss components
            init_log_vars: Initial log variance values (negative = high uncertainty)
            reduction: Loss reduction method
        """
        super().__init__()
        self.num_losses = num_losses
        self.reduction = reduction
        
        # Initialize log variance parameters
        if init_log_vars is None:
            init_log_vars = [0.0] * num_losses  # Equal weights initially
        
        self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))
    
    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        loss_names: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute uncertainty-weighted combined loss.
        
        Args:
            losses: Dictionary of loss values
            loss_names: Optional ordered list of loss names
            
        Returns:
            weighted_loss: Combined loss with uncertainty weighting
            info: Detailed loss information
        """
        if loss_names is None:
            loss_names = list(losses.keys())
        
        total_loss = torch.tensor(0.0, device=self.log_vars.device)
        info = {}
        
        for i, name in enumerate(loss_names):
            if name not in losses:
                continue
            
            L_i = losses[name]
            if isinstance(L_i, torch.Tensor) and L_i.numel() > 1:
                L_i = L_i.mean()
            
            # Uncertainty weighting: (1/2σ²)L + log(σ)
            # We use log_var = log(σ²) for numerical stability
            log_var = self.log_vars[i]
            precision = torch.exp(-log_var)
            
            weighted_loss = precision * L_i + 0.5 * log_var
            total_loss = total_loss + weighted_loss
            
            info[f'{name}_raw'] = L_i.item() if isinstance(L_i, torch.Tensor) else L_i
            info[f'{name}_weight'] = precision.item()
            info[f'uncertainty_{name}'] = math.sqrt(torch.exp(log_var).item())
        
        info['total_weighted'] = total_loss.item()
        
        return total_loss, info
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights based on uncertainties."""
        weights = {}
        for i, name in enumerate(self.log_vars):
            sigma_sq = torch.exp(self.log_vars[i])
            weight = 1.0 / (2 * sigma_sq + 1e-8)
            weights[name] = weight.item()
        return weights


class ProgressiveLossScheduler:
    """Scheduler for progressive loss activation.
    
    Implements warmup and progressive activation of loss terms
    to stabilize training in early stages.
    """
    
    def __init__(
        self,
        loss_config: Dict[str, Dict],
        warmup_steps: int = 1000,
        total_steps: int = 100000,
    ) -> None:
        """Initialize progressive loss scheduler.
        
        Args:
            loss_config: Configuration for each loss term
            warmup_steps: Steps for initial warmup
            total_steps: Total training steps
        """
        self.loss_config = loss_config
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def get_scales(self, step: int) -> Dict[str, float]:
        """Get loss scales for current step.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary of loss name -> scale factor
        """
        scales = {}
        
        # Global warmup factor
        warmup_factor = min(1.0, step / max(1, self.warmup_steps))
        
        for name, config in self.loss_config.items():
            if 'start_step' not in config:
                # Always active with warmup
                scales[name] = warmup_factor * config.get('weight', 1.0)
            else:
                # Progressive activation with optional delay
                start = config['start_step']
                tau = config.get('tau', 1000)  # Sigmoid steepness
                
                if step < start:
                    scales[name] = 0.0
                else:
                    # Sigmoid activation after start
                    t = (step - start) / tau
                    sigmoid_factor = 1.0 / (1.0 + math.exp(-t))
                    scales[name] = sigmoid_factor * config.get('weight', 1.0)
        
        return scales


class SE3ConstraintLoss(nn.Module):
    """SE(3) rigid body constraint loss with progressive activation.
    
    Ensures physical validity of rigid transformations:
    - Rotation matrix orthonormalization
    - Determinant = +1 (proper rotation)
    - Scale/position bounds
    """
    
    def __init__(
        self,
        rotation_weight: float = 0.1,
        det_weight: float = 0.05,
        bounds_weight: float = 0.01,
        progressive_start: int = 0,
        progressive_end: int = 5000,
    ) -> None:
        """Initialize SE(3) constraint loss.
        
        Args:
            rotation_weight: Weight for orthonormalization loss
            det_weight: Weight for determinant loss
            bounds_weight: Weight for parameter bounds
            progressive_start: Start step for progressive activation
            progressive_end: End step for full activation
        """
        super().__init__()
        self.rotation_weight = rotation_weight
        self.det_weight = det_weight
        self.bounds_weight = bounds_weight
        self.progressive_start = progressive_start
        self.progressive_end = progressive_end
    
    def forward(
        self,
        rotations: torch.Tensor,
        translations: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SE(3) constraint loss.
        
        Args:
            rotations: Rotation matrices [N, 3, 3] or quaternions [N, 4]
            translations: Optional translations [N, 3]
            scales: Optional scales [N, 3]
            step: Current training step (for progressive activation)
            
        Returns:
            loss: SE(3) constraint loss
            info: Loss breakdown
        """
        # Progressive activation
        if step < self.progressive_start:
            return torch.tensor(0.0, device=rotations.device), {'se3': 0.0}
        
        if step < self.progressive_end:
            alpha = (step - self.progressive_start) / (self.progressive_end - self.progressive_start)
        else:
            alpha = 1.0
        
        info = {'activation_alpha': alpha}
        losses = []
        
        # Convert quaternions to rotation matrices if needed
        if rotations.shape[-1] == 4:
            rot_mats = self._quaternion_to_matrix(rotations)
        else:
            rot_mats = rotations
        
        # 1. Orthogonality: R @ R^T = I
        rt_r = torch.matmul(rot_mats, rot_mats.transpose(-2, -1))
        I = torch.eye(3, device=rot_mats.device).unsqueeze(0).expand(rot_mats.shape[0], -1, -1)
        loss_ortho = ((rt_r - I) ** 2).mean()
        losses.append(alpha * self.rotation_weight * loss_ortho)
        info['orthogonality'] = loss_ortho.item()
        
        # 2. Determinant = +1 (proper rotation, not reflection)
        det = torch.det(rot_mats)
        loss_det = ((det - 1) ** 2).mean()
        losses.append(alpha * self.det_weight * loss_det)
        info['determinant'] = loss_det.item()
        
        # 3. Translation bounds (if provided)
        if translations is not None and self.bounds_weight > 0:
            # Penalize extremely large translations
            loss_trans = (translations ** 2).mean()
            losses.append(alpha * self.bounds_weight * loss_trans)
            info['translation'] = loss_trans.item()
        
        # 4. Scale bounds (if provided)
        if scales is not None and self.bounds_weight > 0:
            # Ensure scales are positive and reasonable
            scale_vals = torch.exp(scales) if scales.min() < 0 else scales
            loss_scale = ((torch.clamp(scale_vals, min=1e-3) - scale_vals) ** 2).mean()
            losses.append(alpha * self.bounds_weight * loss_scale)
            info['scale'] = loss_scale.item()
        
        total_loss = sum(losses) if losses else torch.tensor(0.0, device=rot_mats.device)
        info['se3_total'] = total_loss.item()
        
        return total_loss, info
    
    def _quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        R = torch.zeros(*q.shape[:-1], 3, 3, device=q.device)
        R[..., 0, 0] = 1 - 2*(y*y + z*z)
        R[..., 0, 1] = 2*(x*y - z*w)
        R[..., 0, 2] = 2*(x*z + y*w)
        R[..., 1, 0] = 2*(x*y + z*w)
        R[..., 1, 1] = 1 - 2*(x*x + z*z)
        R[..., 1, 2] = 2*(y*z - x*w)
        R[..., 2, 0] = 2*(x*z - y*w)
        R[..., 2, 1] = 2*(y*z + x*w)
        R[..., 2, 2] = 1 - 2*(x*x + y*y)
        
        return R


class TrajectoryConsistencyLoss(nn.Module):
    """Trajectory consistency loss between 4DGS and external trackers.
    
    Ensures consistency between:
    - 4DGS Gaussian trajectories
    - TAPIR/CoTracker point trajectories
    
    Math:
        L_traj = Σ_t ||p_4dgs(t) - p_track(t)||²
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        occlusion_threshold: float = 0.5,
        distance_threshold: float = 1.0,
    ) -> None:
        """Initialize trajectory consistency loss.
        
        Args:
            weight: Loss weight
            occlusion_threshold: Visibility threshold (0-1)
            distance_threshold: Max distance for matching
        """
        super().__init__()
        self.weight = weight
        self.occlusion_threshold = occlusion_threshold
        self.distance_threshold = distance_threshold
    
    def forward(
        self,
        gauss_trajectories: torch.Tensor,
        track_trajectories: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute trajectory consistency loss.
        
        Args:
            gauss_trajectories: 4DGS trajectories [T, N, 3] or [T, N, 4] (with time)
            track_trajectories: External tracker trajectories [T, M, 3]
            visibility: Point visibility [T, M]
            mask: Optional validity mask
            
        Returns:
            loss: Trajectory consistency loss
            info: Loss info
        """
        T, N, D = gauss_trajectories.shape[:3]
        
        info = {}
        
        # Visibility filtering
        if visibility is not None:
            valid = visibility > self.occlusion_threshold
        else:
            valid = torch.ones(T, track_trajectories.shape[1], device=track_trajectories.device, dtype=torch.bool)
        
        # Point-to-trajectory matching (simplified: nearest neighbor)
        # For each track point, find nearest Gaussian trajectory
        losses = []
        
        for t in range(T):
            gt_pos = gauss_trajectories[t]  # [N, 3]
            track_pos = track_trajectories[t]  # [M, 3]
            
            if track_pos.shape[0] == 0:
                continue
            
            # Compute distances
            dists = torch.cdist(track_pos.unsqueeze(0), gt_pos.unsqueeze(0)).squeeze(0)  # [M, N]
            
            # Find nearest Gaussian for each track point
            min_dists, _ = dists.min(dim=1)  # [M]
            
            # Apply visibility mask
            valid_mask = valid[t]
            
            if valid_mask.sum() > 0:
                loss_t = (min_dists[valid_mask] ** 2).mean()
                losses.append(loss_t)
        
        if losses:
            total_loss = self.weight * torch.stack(losses).mean()
            info['trajectory_consistency'] = total_loss.item()
        else:
            total_loss = torch.tensor(0.0, device=gauss_trajectories.device)
            info['trajectory_consistency'] = 0.0
        
        return total_loss, info


class DepthConsistencyLoss(nn.Module):
    """Depth consistency loss for multi-view reconstruction.
    
    Ensures consistency between:
    - Rendered depth maps
    - LiDAR point depth
    - Predicted depth maps
    
    Math:
        L_depth = ||d_render - d_lidar||² + ||d_render - d_pred||²
    """
    
    def __init__(
        self,
        lidar_weight: float = 1.0,
        pred_weight: float = 0.5,
        threshold: float = 0.1,
        use_outlier_rejection: bool = True,
    ) -> None:
        """Initialize depth consistency loss.
        
        Args:
            lidar_weight: Weight for LiDAR depth supervision
            pred_weight: Weight for predicted depth supervision
            threshold: Depth threshold for outlier rejection
            use_outlier_rejection: Whether to reject large errors
        """
        super().__init__()
        self.lidar_weight = lidar_weight
        self.pred_weight = pred_weight
        self.threshold = threshold
        self.use_outlier_rejection = use_outlier_rejection
    
    def forward(
        self,
        rendered_depth: torch.Tensor,
        lidar_depth: Optional[torch.Tensor] = None,
        pred_depth: Optional[torch.Tensor] = None,
        lidar_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute depth consistency loss.
        
        Args:
            rendered_depth: Rendered depth map [B, 1, H, W] or [B, H, W]
            lidar_depth: LiDAR depth values [N] or projected depth [B, H, W]
            pred_depth: Predicted depth map [B, 1, H, W]
            lidar_coords: LiDAR point coordinates for reprojection
            
        Returns:
            loss: Depth consistency loss
            info: Loss breakdown
        """
        if rendered_depth.dim() == 3:
            rendered_depth = rendered_depth.unsqueeze(1)
        
        info = {}
        losses = []
        
        # LiDAR depth consistency
        if lidar_depth is not None and self.lidar_weight > 0:
            if lidar_coords is not None:
                # Project LiDAR to depth and compare
                lidar_proj = self._project_lidar_to_depth(lidar_depth, lidar_coords, rendered_depth.shape)
                
                diff = rendered_depth - lidar_proj
                
                if self.use_outlier_rejection:
                    # Huber-like loss with outlier rejection
                    abs_diff = torch.abs(diff)
                    loss_lidar = torch.where(
                        abs_diff < self.threshold,
                        0.5 * diff ** 2,
                        self.threshold * (abs_diff - 0.5 * self.threshold)
                    )
                else:
                    loss_lidar = diff ** 2
                
                loss_lidar = loss_lidar.mean()
            else:
                # Direct depth comparison
                loss_lidar = ((rendered_depth - lidar_depth.unsqueeze(1)) ** 2).mean()
            
            losses.append(self.lidar_weight * loss_lidar)
            info['lidar_depth'] = loss_lidar.item()
        
        # Predicted depth consistency
        if pred_depth is not None and self.pred_weight > 0:
            diff = rendered_depth - pred_depth
            loss_pred = (diff ** 2).mean()
            losses.append(self.pred_weight * loss_pred)
            info['pred_depth'] = loss_pred.item()
        
        if losses:
            total_loss = sum(losses)
            info['depth_total'] = total_loss.item()
        else:
            total_loss = torch.tensor(0.0, device=rendered_depth.device)
            info['depth_total'] = 0.0
        
        return total_loss, info
    
    def _project_lidar_to_depth(
        self,
        lidar_depth: torch.Tensor,
        lidar_coords: torch.Tensor,
        target_shape: Tuple,
    ) -> torch.Tensor:
        """Project LiDAR points to depth map coordinates."""
        B, C, H, W = target_shape
        
        # Simplified: create depth map from LiDAR points
        depth_map = torch.zeros(B, 1, H, W, device=lidar_depth.device)
        
        # This is a placeholder - actual implementation would use
        # camera intrinsics/extrinsics for proper projection
        
        return depth_map


class FocalLossVariant(nn.Module):
    """Focal Loss variant for semantic segmentation with class imbalance.
    
    Original Focal Loss (Lin et al., 2017):
        L_focal = -αₜ(1 - pₜ)ᵞ log(pₜ)
    
    Extensions:
    - Class-balanced focal loss
    - Quality focal loss
    - Distribution focal loss
    """
    
    def __init__(
        self,
        num_classes: int = 23,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        variant: str = "standard",
    ) -> None:
        """Initialize Focal Loss variant.
        
        Args:
            num_classes: Number of classes
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma (focusing) parameter
            class_weights: Class-balanced weights
            reduction: Loss reduction method
            variant: Loss variant ('standard', 'balanced', 'quality')
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction
        self.variant = variant
        
        if class_weights is not None:
            self.register_buffer('cb_weights', class_weights)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        quality: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Focal Loss.
        
        Args:
            logits: [B, C, H, W] or [C, H, W]
            targets: [B, H, W] or [H, W]
            quality: Optional quality scores for quality focal loss
            mask: Optional validity mask
            
        Returns:
            loss: Focal loss
            info: Loss information
        """
        # Handle dimensions
        if logits.dim() == 3:
            logits = logits.unsqueeze(0)
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        B, C, H, W = logits.shape
        
        # Flatten
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        
        if mask is not None:
            mask_flat = mask.reshape(-1)
            valid = mask_flat & (targets_flat >= 0) & (targets_flat < C)
        else:
            valid = (targets_flat >= 0) & (targets_flat < C)
        
        logits_flat = logits_flat[valid]
        targets_flat = targets_flat[valid]
        
        if len(targets_flat) == 0:
            return torch.tensor(0.0, device=logits.device), {'focal': 0.0}
        
        # Compute probabilities
        probs = F.softmax(logits_flat, dim=-1)
        pt = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        
        # Standard focal loss
        focal_weight = (1 - pt) ** self.gamma
        
        # Class-balanced weighting
        if self.class_weights is not None:
            class_weights = self.cb_weights.to(logits.device)
            alpha_t = class_weights[targets_flat]
        else:
            alpha_t = self.alpha
        
        # Quality focal loss extension
        if self.variant == "quality" and quality is not None:
            # Weight by quality score
            quality_flat = quality.reshape(-1)[valid]
            focal_weight = focal_weight * quality_flat
        
        # Cross entropy
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Final focal loss
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        with torch.no_grad():
            accuracy = (pt.argmax() == targets_flat).float().mean()
        
        info = {
            'focal': loss.item(),
            'accuracy': accuracy.item(),
            'mean_pt': pt.mean().item(),
        }
        
        return loss, info


class CombinedJointLoss(nn.Module):
    """Combined joint loss for P4 (joint optimization of tracking and reconstruction).
    
    Integrates all loss components with uncertainty weighting and progressive activation.
    """
    
    def __init__(
        self,
        loss_components: Dict[str, nn.Module],
        uncertainty_weighting: bool = True,
        progressive_activation: bool = True,
    ) -> None:
        """Initialize combined joint loss.
        
        Args:
            loss_components: Dictionary of loss modules
            uncertainty_weighting: Use uncertainty-based weighting
            progressive_activation: Use progressive loss activation
        """
        super().__init__()
        
        self.loss_components = nn.ModuleDict(loss_components)
        self.uncertainty_weighting = uncertainty_weighting
        self.progressive_activation = progressive_activation
        
        # Uncertainty weighting for all losses
        if uncertainty_weighting:
            self.uncertainty_weights = UncertaintyWeightedLoss(
                num_losses=len(loss_components),
                init_log_vars=[0.0] * len(loss_components),
            )
        
        # Progressive scheduler
        if progressive_activation:
            loss_config = {
                'photometric': {'weight': 1.0},
                'semantic': {'weight': 1.0, 'start_step': 1000},
                'silhouette': {'weight': 0.5},
                'regularization': {'weight': 0.1},
                'se3_constraint': {'weight': 0.1, 'start_step': 500, 'tau': 500},
                'trajectory': {'weight': 1.0, 'start_step': 2000},
                'depth': {'weight': 0.5, 'start_step': 1000},
            }
            self.progressive_scheduler = ProgressiveLossScheduler(
                loss_config=loss_config,
                warmup_steps=1000,
                total_steps=100000,
            )
    
    def forward(
        self,
        predictions: Dict,
        targets: Dict,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined joint loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            step: Current training step
            
        Returns:
            total_loss: Combined loss
            all_info: Detailed loss information
        """
        all_losses = {}
        all_info = {}
        
        # Get progressive scales
        if self.progressive_activation:
            scales = self.progressive_scheduler.get_scales(step)
        else:
            scales = {name: 1.0 for name in self.loss_components.keys()}
        
        # Compute each loss component
        for name, loss_module in self.loss_components.items():
            if not hasattr(loss_module, 'forward'):
                continue
            
            try:
                loss_val, info = loss_module(
                    predictions=predictions,
                    targets=targets,
                    step=step,
                )
                
                # Apply scale
                scale = scales.get(name, 1.0)
                weighted_loss = scale * loss_val
                
                all_losses[name] = loss_val
                all_info[f'{name}_weighted'] = weighted_loss.item()
                all_info[f'{name}_scale'] = scale
                all_info.update({f'{name}_{k}': v for k, v in info.items()})
                
            except Exception as e:
                # Skip failed loss components
                print(f"Warning: Loss component {name} failed: {e}")
                all_losses[name] = torch.tensor(0.0)
                all_info[f'{name}_weighted'] = 0.0
        
        # Uncertainty weighting
        if self.uncertainty_weighting and hasattr(self, 'uncertainty_weights'):
            total_loss, uncertainty_info = self.uncertainty_weights(all_losses)
            all_info.update(uncertainty_info)
        else:
            total_loss = sum(all_losses.values())
            all_info['total_unweighted'] = total_loss.item()
        
        all_info['step'] = step
        
        return total_loss, all_info


# Utility functions for loss computation
def compute_multi_scale_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable,
    scales: List[int] = [1, 2, 4],
) -> torch.Tensor:
    """Compute multi-scale loss.
    
    Args:
        predictions: Multi-scale predictions
        targets: Target tensor
        loss_fn: Loss function to apply
        scales: List of downsampling factors
        
    Returns:
        Combined multi-scale loss
    """
    total_loss = 0.0
    
    for scale in scales:
        if scale > 1:
            pred = F.interpolate(predictions, scale_factor=1/scale, mode='bilinear')
            target = F.interpolate(targets, scale_factor=1/scale, mode='bilinear')
        else:
            pred = predictions
            target = targets
        
        loss = loss_fn(pred, target)
        total_loss = total_loss + loss / len(scales)
    
    return total_loss

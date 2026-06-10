"""
Semantic-4DGS-Traffic: Evaluation Metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_metrics(
    predictions: Dict[str, List[torch.Tensor]],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute evaluation metrics for Semantic-4DGS-Traffic.
    
    Args:
        predictions: Dict of predicted tensors
        targets: Dict of target tensors
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    # RGB metrics
    if "rgb" in predictions and "rgb" in targets:
        metrics["psnr"] = compute_psnr(predictions["rgb"][0], targets["rgb"])
        metrics["ssim"] = compute_ssim(predictions["rgb"][0], targets["rgb"])
        
    # Depth metrics
    if "depth" in predictions and "depth" in targets:
        metrics["depth_mae"] = compute_depth_mae(predictions["depth"][0], targets["depth"])
        metrics["depth_rmse"] = compute_depth_rmse(predictions["depth"][0], targets["depth"])
        
    # Semantic metrics
    if "semantic" in predictions and "semantic" in targets:
        metrics["mIoU"] = compute_mean_iou(predictions["semantic"][0], targets["semantic"])
        metrics["accuracy"] = compute_semantic_accuracy(predictions["semantic"][0], targets["semantic"])
        
    # Silhouette metrics
    if "silhouette" in predictions and "silhouette" in targets:
        metrics["IoU"] = compute_iou(predictions["silhouette"][0], targets["silhouette"])
        
    return metrics


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Compute SSIM"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = pred.mean()
    mu2 = target.mean()
    sigma1 = pred.var()
    sigma2 = target.var()
    sigma12 = ((pred - mu1) * (target - mu2)).mean()
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
           
    return ssim.item()


def compute_depth_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute depth MAE"""
    mask = target > 0
    if mask.sum() == 0:
        return 0.0
    return torch.abs(pred[mask] - target[mask]).mean().item()


def compute_depth_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute depth RMSE"""
    mask = target > 0
    if mask.sum() == 0:
        return 0.0
    return torch.sqrt(((pred[mask] - target[mask]) ** 2).mean()).item()


def compute_mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 20) -> float:
    """Compute mean IoU for semantic segmentation"""
    pred_labels = pred.argmax(dim=0)
    
    ious = []
    for cls in range(num_classes):
        pred_mask = pred_labels == cls
        target_mask = target == cls
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(float('nan'))
            
    return np.nanmean(ious)


def compute_semantic_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute semantic segmentation accuracy"""
    pred_labels = pred.argmax(dim=0)
    correct = (pred_labels == target).sum().item()
    total = target.numel()
    return correct / total if total > 0 else 0.0


def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute binary IoU for silhouette"""
    intersection = (pred.bool() & target.bool()).sum().item()
    union = (pred.bool() | target.bool()).sum().item()
    
    if union == 0:
        return 0.0
    return intersection / union


def compute_trajectory_metrics(
    predicted_trajectories: List,
    ground_truth_trajectories: List,
) -> Dict[str, float]:
    """Compute trajectory tracking metrics"""
    metrics = {}
    
    if len(predicted_trajectories) == 0:
        return metrics
        
    # Average trajectory error
    errors = []
    for pred_traj, gt_traj in zip(predicted_trajectories, ground_truth_trajectories):
        if len(pred_traj) == len(gt_traj):
            error = torch.norm(pred_traj - gt_traj, dim=-1).mean().item()
            errors.append(error)
            
    if len(errors) > 0:
        metrics["ATE"] = np.mean(errors)  # Average Trajectory Error
        metrics["RTE"] = np.std(errors)   # RMSE of Trajectory Error
        
    return metrics

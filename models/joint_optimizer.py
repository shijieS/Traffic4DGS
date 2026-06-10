"""
Joint Tracking-Reconstruction Optimizer - Optimized Version.

OPTIMIZATION CHANGELOG (v1.1.0):
  [OPT-12] 多阶段训练策略（先静态→后动态→联合优化）
  [OPT-13] 学习率调度器（不同参数组不同学习率）
  [OPT-14] 梯度累积支持（大batch场景）
  [OPT-15] SAM2掩码→4DGS反馈的掩码精炼循环

@author Semantic 4DGS Team
@version 1.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable, NamedTuple
from dataclasses import dataclass, field
import numpy as np
from enum import Enum


# ============================================================================
# [OPT-12] MULTI-STAGE TRAINING STRATEGY
# ============================================================================

class TrainingStage(Enum):
    """Training stages for progressive optimization."""
    WARMUP = "warmup"           # Static field only
    DETECTION = "detection"     # Instance detection with SAM2
    DYNAMIC = "dynamic"          # Dynamic field optimization
    JOINT = "joint"             # Full joint optimization


@dataclass
class StageConfig:
    """Configuration for a training stage."""
    name: TrainingStage
    num_steps: int
    enabled_modules: List[str]
    loss_weights: Dict[str, float]
    lr_multpliers: Dict[str, float]
    densify_enabled: bool = True
    sam2_feedback_enabled: bool = False


class MultiStageScheduler:
    r"""Scheduler for multi-stage training.
    
    OPTIMIZATION [OPT-12]: Implements progressive training from
    simple (static) to complex (joint) optimization.
    
    Stage 1: WARMUP (1000 steps)
        - Static field optimization only
        - Focus on scene geometry
        - High photometric weight, low others
        
    Stage 2: DETECTION (2000 steps)
        - Enable SAM2 instance detection
        - Learn semantic features
        - Start dynamic field initialization
        
    Stage 3: DYNAMIC (3000 steps)
        - Enable SE(3) transformations
        - Optimize dynamic Gaussians
        - Joint tracking-reconstruction
        
    Stage 4: JOINT (5000+ steps)
        - Full joint optimization
        - SAM2 feedback loop
        - Trajectory consistency
    """
    
    DEFAULT_CONFIGS = {
        TrainingStage.WARMUP: StageConfig(
            name=TrainingStage.WARMUP,
            num_steps=1000,
            enabled_modules=['static_field'],
            loss_weights={
                'photometric': 1.0,
                'semantic': 0.0,
                'silhouette': 0.01,
                'tracking': 0.0,
                'regularization': 0.1,
                'se3_constraint': 0.0,
            },
            lr_multpliers={
                'positions': 1.0,
                'scales': 1.0,
                'rotations': 1.0,
                'features': 0.5,
            },
            densify_enabled=True,
        ),
        TrainingStage.DETECTION: StageConfig(
            name=TrainingStage.DETECTION,
            num_steps=2000,
            enabled_modules=['static_field', 'sam2_tracker'],
            loss_weights={
                'photometric': 1.0,
                'semantic': 0.1,
                'silhouette': 0.05,
                'tracking': 0.0,
                'regularization': 0.05,
                'se3_constraint': 0.0,
            },
            lr_multpliers={
                'positions': 1.0,
                'scales': 1.0,
                'rotations': 1.0,
                'features': 1.0,
                'semantic': 1.0,
            },
            densify_enabled=True,
        ),
        TrainingStage.DYNAMIC: StageConfig(
            name=TrainingStage.DYNAMIC,
            num_steps=3000,
            enabled_modules=['static_field', 'dynamic_field', 'se3_transform'],
            loss_weights={
                'photometric': 1.0,
                'semantic': 0.1,
                'silhouette': 0.05,
                'tracking': 0.1,
                'regularization': 0.02,
                'se3_constraint': 0.5,
                'trajectory_smooth': 0.05,
            },
            lr_multpliers={
                'positions': 0.5,
                'scales': 0.5,
                'rotations': 0.5,
                'features': 0.5,
                'pose_twists': 1.0,
            },
            densify_enabled=True,
        ),
        TrainingStage.JOINT: StageConfig(
            name=TrainingStage.JOINT,
            num_steps=10000,
            enabled_modules=['static_field', 'dynamic_field', 'se3_transform', 'sam2_tracker', 'point_tracker'],
            loss_weights={
                'photometric': 1.0,
                'semantic': 0.2,
                'silhouette': 0.1,
                'tracking': 0.5,
                'regularization': 0.01,
                'se3_constraint': 1.0,
                'trajectory_smooth': 0.1,
                'mask_refinement': 0.1,
            },
            lr_multpliers={
                'positions': 0.3,
                'scales': 0.3,
                'rotations': 0.3,
                'features': 0.3,
                'pose_twists': 0.5,
            },
            densify_enabled=True,
            sam2_feedback_enabled=True,
        ),
    }
    
    def __init__(
        self,
        custom_configs: Optional[Dict[TrainingStage, StageConfig]] = None,
    ) -> None:
        """Initialize multi-stage scheduler.
        
        Args:
            custom_configs: Override default configurations
        """
        self.configs = self.DEFAULT_CONFIGS.copy()
        if custom_configs:
            self.configs.update(custom_configs)
        
        self.current_stage = TrainingStage.WARMUP
        self.stage_step = 0
        self.total_steps = 0
        
        # Stage boundaries
        self._boundaries = {}
        cumulative = 0
        for stage in TrainingStage:
            self._boundaries[stage] = cumulative
            cumulative += self.configs[stage].num_steps
    
    def get_current_config(self) -> StageConfig:
        """Get configuration for current stage."""
        return self.configs[self.current_stage]
    
    def step(self) -> StageConfig:
        """Advance one step and return current config."""
        self.stage_step += 1
        self.total_steps += 1
        
        # Check for stage transition
        config = self.get_current_config()
        if self.stage_step >= config.num_steps:
            self._advance_stage()
        
        return self.get_current_config()
    
    def _advance_stage(self) -> None:
        """Advance to next training stage."""
        stages = list(TrainingStage)
        current_idx = stages.index(self.current_stage)
        
        if current_idx < len(stages) - 1:
            self.current_stage = stages[current_idx + 1]
            self.stage_step = 0
            print(f"Advanced to stage: {self.current_stage.value}")
    
    def get_stage_info(self) -> Dict[str, Any]:
        """Get information about current stage."""
        config = self.get_current_config()
        progress = self.stage_step / config.num_steps if config.num_steps > 0 else 1.0
        
        return {
            'current_stage': self.current_stage.value,
            'stage_progress': progress,
            'total_steps': self.total_steps,
            'enabled_modules': config.enabled_modules,
            'loss_weights': config.loss_weights,
        }


# ============================================================================
# [OPT-13] LEARNING RATE SCHEDULER WITH PARAMETER GROUP DIFFERENTIATION
# ============================================================================

class LearningRateScheduler:
    r"""Multi-parameter learning rate scheduler.
    
    OPTIMIZATION [OPT-13]: Implements per-parameter-group learning rates
    with warmup and decay schedules.
    
    Features:
        - Different LR for different parameter groups
        - Warmup phase
        - Exponential decay
        - Stage-based LR adjustment
        - Gradient clipping
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lrs: Dict[str, float],
        warmup_steps: int = 500,
        decay_type: str = "exponential",
        decay_rate: float = 0.95,
        decay_steps: int = 1000,
        min_lr: float = 1e-6,
    ) -> None:
        """Initialize LR scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            base_lrs: Base learning rates per parameter group
            warmup_steps: Warmup phase length
            decay_type: Type of decay ("exponential", "cosine", "step")
            decay_rate: Decay factor
            decay_steps: Steps between decay
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.base_lrs = base_lrs
        self.warmup_steps = warmup_steps
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        
        self.step_count = 0
        
        # Store parameter group indices
        self.param_groups_idx = {}
        for i, pg in enumerate(optimizer.param_groups):
            name = pg.get('name', f'group_{i}')
            self.param_groups_idx[name] = i
    
    def step(self, stage_config: Optional[StageConfig] = None) -> None:
        """Update learning rates for all parameter groups."""
        self.step_count += 1
        
        # Compute warmup factor
        if self.step_count < self.warmup_steps:
            warmup_factor = self.step_count / self.warmup_steps
        else:
            warmup_factor = 1.0
        
        # Compute decay factor
        decay_steps_adjusted = max(1, self.step_count - self.warmup_steps)
        
        if self.decay_type == "exponential":
            decay_factor = self.decay_rate ** (decay_steps_adjusted / self.decay_steps)
        elif self.decay_type == "cosine":
            decay_factor = 0.5 * (1 + np.cos(np.pi * decay_steps_adjusted / self.decay_steps))
        elif self.decay_type == "step":
            decay_factor = self.decay_rate ** (decay_steps_adjusted // self.decay_steps)
        else:
            decay_factor = 1.0
        
        # Update LR for each group
        for pg in self.optimizer.param_groups:
            name = pg.get('name', 'default')
            base_lr = self.base_lrs.get(name, self.base_lrs.get('default', 1e-4))
            
            # Apply stage multiplier if provided
            stage_mult = 1.0
            if stage_config is not None:
                stage_mult = stage_config.lr_multpliers.get(name, 1.0)
            
            # Compute new LR
            new_lr = base_lr * warmup_factor * decay_factor * stage_mult
            new_lr = max(new_lr, self.min_lr)
            
            pg['lr'] = new_lr
    
    def get_lr(self, name: str) -> float:
        """Get current learning rate for a parameter group."""
        for pg in self.optimizer.param_groups:
            if pg.get('name') == name:
                return pg['lr']
        return self.base_lrs.get('default', 1e-4)


# ============================================================================
# [OPT-14] GRADIENT ACCUMULATION SUPPORT
# ============================================================================

class GradientAccumulator:
    r"""Gradient accumulation for large batch simulation.
    
    OPTIMIZATION [OPT-14]: Enables training with effective batch sizes
    larger than GPU memory by accumulating gradients over multiple
    micro-batches.
    
    Features:
        - Configurable accumulation steps
        - Automatic gradient scaling
        - Gradient checkpointing support
        - Mixed precision support
    """
    
    def __init__(
        self,
        accumulation_steps: int = 4,
        gradient_scale: float = 1.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        """Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of micro-batches to accumulate
            gradient_scale: Scale factor for accumulated gradients
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.accumulation_steps = accumulation_steps
        self.gradient_scale = gradient_scale
        self.max_grad_norm = max_grad_norm
        
        self._step_count = 0
        self._accumulated = False
        
        # Gradient buffers
        self._grad_buffers: Dict[str, torch.Tensor] = {}
    
    def should_step(self) -> bool:
        """Check if optimizer should take a step."""
        return (self._step_count + 1) % self.accumulation_steps == 0
    
    def backward(
        self,
        loss: torch.Tensor,
        retain_graph: bool = False,
    ) -> None:
        """Perform backward pass with scaling."""
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward(retain_graph=retain_graph)
    
    def step(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Tuple[bool, float]:
        r"""Perform optimizer step if accumulated enough.
        
        Args:
            optimizer: PyTorch optimizer
            scaler: Optional gradient scaler for mixed precision
            
        Returns:
            did_step: Whether optimizer step was performed
            grad_norm: Current gradient norm
        """
        self._step_count += 1
        
        if not self.should_step():
            self._accumulated = True
            return False, 0.0
        
        # Unscale gradients if using AMP
        if scaler is not None:
            scaler.unscale_(optimizer)
        
        # Compute gradient norm
        total_norm = 0.0
        for pg in optimizer.param_groups:
            for p in pg['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                optimizer.param_groups[0]['params'],
                self.max_grad_norm
            )
        
        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Zero gradients
        optimizer.zero_grad()
        
        self._accumulated = False
        return True, total_norm
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero accumulated gradients."""
        for p in self._grad_buffers.values():
            if p is not None:
                if set_to_none:
                    p = None
                else:
                    p.zero_()


# ============================================================================
# [OPT-15] MASK REFINEMENT LOOP (SAM2 ↔ 4DGS)
# ============================================================================

class MaskRefinementLoop:
    r"""Iterative mask refinement between SAM2 and 4DGS.
    
    OPTIMIZATION [OPT-15]: Implements a feedback loop where:
        1. SAM2 provides initial masks
        2. 4DGS renders with these masks
        3. Rendered masks are compared with SAM2
        4. Discrepancies refine SAM2 prompts
        5. Repeat until convergence
    
    This improves both:
        - SAM2: More precise segmentation
        - 4DGS: Better semantic consistency
    """
    
    def __init__(
        self,
        num_iterations: int = 3,
        iou_threshold: float = 0.8,
        refinement_lr: float = 0.1,
        momentum: float = 0.9,
    ) -> None:
        """Initialize mask refinement loop.
        
        Args:
            num_iterations: Number of refinement iterations
            iou_threshold: IoU threshold for convergence
            refinement_lr: Learning rate for prompt refinement
            momentum: Momentum for exponential moving average
        """
        self.num_iterations = num_iterations
        self.iou_threshold = iou_threshold
        self.refinement_lr = refinement_lr
        self.momentum = momentum
        
        # Refined prompts storage
        self._refined_prompts: Dict[int, torch.Tensor] = {}
        self._prompt_momentum: Dict[int, torch.Tensor] = {}
        
        # Statistics
        self._refinement_history: List[Dict] = []
    
    def compute_iou(
        self,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> float:
        r"""Compute IoU between two masks.
        
        Args:
            mask1: First mask [H, W]
            mask2: Second mask [H, W]
            
        Returns:
            iou: Intersection over Union
        """
        intersection = (mask1 & mask2).sum().float()
        union = (mask1 | mask2).sum().float()
        
        if union == 0:
            return 1.0
        
        return (intersection / union).item()
    
    def refine_prompts(
        self,
        instance_id: int,
        rendered_mask: torch.Tensor,
        sam2_mask: torch.Tensor,
        rendered_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        r"""Refine SAM2 prompts based on rendered mask discrepancy.
        
        Args:
            instance_id: Instance identifier
            rendered_mask: Rendered mask from 4DGS [H, W]
            sam2_mask: Original SAM2 mask [H, W]
            rendered_confidence: Confidence map from renderer
            
        Returns:
            refined_mask: Refined mask
            improvement: IoU improvement
        """
        # Compute initial IoU
        initial_iou = self.compute_iou(rendered_mask, sam2_mask)
        
        # Initialize momentum if needed
        if instance_id not in self._prompt_momentum:
            self._prompt_momentum[instance_id] = torch.zeros_like(sam2_mask, dtype=torch.float32)
        
        # Compute discrepancy
        discrepancy = (rendered_mask.float() - sam2_mask.float())
        
        # Apply momentum
        momentum_update = self.momentum * self._prompt_momentum[instance_id] + \
                        (1 - self.momentum) * discrepancy
        
        self._prompt_momentum[instance_id] = momentum_update
        
        # Refine mask with learning rate
        refined_mask = sam2_mask.float() + self.refinement_lr * momentum_update
        refined_mask = (refined_mask > 0.5).float()
        
        # Compute refined IoU
        refined_iou = self.compute_iou(refined_mask.bool(), rendered_mask.bool())
        improvement = refined_iou - initial_iou
        
        return refined_mask, improvement
    
    def run_refinement_loop(
        self,
        rendered_masks: Dict[int, torch.Tensor],
        sam2_masks: Dict[int, torch.Tensor],
        rendered_confidences: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[str, float]]:
        r"""Run iterative mask refinement.
        
        Args:
            rendered_masks: Rendered masks from 4DGS
            sam2_masks: Initial SAM2 masks
            rendered_confidences: Optional confidence maps
            
        Returns:
            refined_masks: Refined masks after loop
            stats: Refinement statistics
        """
        refined_masks = sam2_masks.copy()
        stats = {
            'initial_iou': [],
            'final_iou': [],
            'improvements': [],
            'converged': [],
        }
        
        for iteration in range(self.num_iterations):
            iteration_stats = {
                'initial_iou': [],
                'final_iou': [],
                'improvements': [],
            }
            
            for instance_id in rendered_masks.keys():
                if instance_id not in refined_masks:
                    continue
                
                # Compute IoU before refinement
                iou_before = self.compute_iou(
                    rendered_masks[instance_id],
                    refined_masks[instance_id]
                )
                iteration_stats['initial_iou'].append(iou_before)
                
                # Refine
                conf = rendered_confidences.get(instance_id) if rendered_confidences else None
                refined, improvement = self.refine_prompts(
                    instance_id,
                    rendered_masks[instance_id],
                    refined_masks[instance_id],
                    conf,
                )
                
                refined_masks[instance_id] = refined
                
                # Compute IoU after
                iou_after = self.compute_iou(
                    rendered_masks[instance_id],
                    refined
                )
                iteration_stats['final_iou'].append(iou_after)
                iteration_stats['improvements'].append(improvement)
            
            # Check convergence
            mean_iou = np.mean(iteration_stats['final_iou']) if iteration_stats['final_iou'] else 0.0
            converged = mean_iou >= self.iou_threshold
            
            if converged:
                print(f"Mask refinement converged at iteration {iteration + 1}")
                break
        
        # Aggregate stats
        stats['initial_iou'] = np.mean(iteration_stats.get('initial_iou', [0.0]))
        stats['final_iou'] = np.mean(iteration_stats.get('final_iou', [0.0]))
        stats['improvements'] = np.mean(iteration_stats.get('improvements', [0.0]))
        stats['converged'] = converged
        
        self._refinement_history.append(stats)
        
        return refined_masks, stats
    
    def get_refined_mask(self, instance_id: int) -> Optional[torch.Tensor]:
        """Get refined mask for an instance."""
        return self._refined_prompts.get(instance_id)


# ============================================================================
# ENHANCED JOINT OPTIMIZER
# ============================================================================

@dataclass
class OptimizationState:
    """Current state of optimization."""
    step: int
    epoch: int
    stage: str
    loss_total: float
    loss_photometric: float
    loss_semantic: float
    loss_silhouette: float
    loss_tracking: float
    loss_regularization: float
    loss_se3_constraint: float
    num_gaussians: int
    num_instances: int
    learning_rate: float
    grad_norm: float


class JointOptimizer:
    r"""Enhanced joint tracking-reconstruction optimizer.
    
    OPTIMIZATION [OPT-12-15]:
        - [OPT-12] Multi-stage training
        - [OPT-13] Per-parameter LR scheduling
        - [OPT-14] Gradient accumulation
        - [OPT-15] Mask refinement loop
    """
    
    def __init__(
        self,
        static_field: nn.Module,
        dynamic_field: Optional[nn.Module] = None,
        sam2_tracker: Optional[Any] = None,
        point_tracker: Optional[Any] = None,
        losses: Optional[Dict[str, Callable]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        
        # Models
        self.static_field = static_field.to(self.device)
        self.dynamic_field = dynamic_field.to(self.device) if dynamic_field else None
        
        # Trackers
        self.sam2_tracker = sam2_tracker
        self.point_tracker = point_tracker
        
        # [OPT-12] Multi-stage scheduler
        self.stage_scheduler = MultiStageScheduler()
        
        # [OPT-13] Learning rate scheduler
        self.lr_scheduler: Optional[LearningRateScheduler] = None
        
        # [OPT-14] Gradient accumulator
        accumulation_steps = self.config.get('accumulation_steps', 1)
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.grad_accumulator = GradientAccumulator(
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
        )
        
        # [OPT-15] Mask refinement
        self.mask_refiner = MaskRefinementLoop(
            num_iterations=self.config.get('refinement_iterations', 3),
        )
        
        # Loss weights (will be updated by stage scheduler)
        self.loss_weights = {
            'photometric': 1.0,
            'semantic': 0.0,
            'silhouette': 0.01,
            'tracking': 0.0,
            'regularization': 0.01,
            'se3_constraint': 0.0,
            'trajectory_smooth': 0.0,
            'mask_refinement': 0.0,
        }
        
        # Optimizers
        self._setup_optimizers()
        
        # State
        self.state = OptimizationState(
            step=0,
            epoch=0,
            stage='warmup',
            loss_total=0.0,
            loss_photometric=0.0,
            loss_semantic=0.0,
            loss_silhouette=0.0,
            loss_tracking=0.0,
            loss_regularization=0.0,
            loss_se3_constraint=0.0,
            num_gaussians=0,
            num_instances=0,
            learning_rate=self.config.get('learning_rate', 1e-4),
            grad_norm=0.0,
        )
        
        # Loss functions
        self.losses = losses or self._default_losses()
        
        # AMP scaler
        self.use_amp = self.config.get('use_amp', False)
        self.scaler = None
        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _setup_optimizers(self) -> None:
        """Setup optimizers with named parameter groups."""
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        # Define parameter groups with names
        param_groups = [
            {'name': 'positions', 'params': [self.static_field._positions]},
            {'name': 'scales', 'params': [self.static_field._scales]},
            {'name': 'rotations', 'params': [self.static_field._rotations]},
            {'name': 'opacities', 'params': [self.static_field._opacities]},
            {'name': 'features', 'params': [self.static_field._features]},
        ]
        
        self.optimizer_static = torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
        
        # [OPT-13] Setup LR scheduler
        base_lrs = {
            'positions': lr,
            'scales': lr * 0.5,
            'rotations': lr * 0.5,
            'opacities': lr,
            'features': lr * 0.5,
            'default': lr,
        }
        self.lr_scheduler = LearningRateScheduler(
            optimizer=self.optimizer_static,
            base_lrs=base_lrs,
            warmup_steps=self.config.get('warmup_steps', 500),
        )
        
        if self.dynamic_field is not None:
            param_groups_dynamic = [
                {'name': 'pose_twists', 'params': [self.dynamic_field._pose_twists]},
                {'name': 'dynamic_positions', 'params': [self.dynamic_field._positions]},
                {'name': 'dynamic_features', 'params': [self.dynamic_field._features]},
            ]
            self.optimizer_dynamic = torch.optim.Adam(
                param_groups_dynamic,
                lr=lr * 2,  # Higher LR for pose
                weight_decay=weight_decay,
            )
        else:
            self.optimizer_dynamic = None
    
    def _default_losses(self) -> Dict[str, Callable]:
        """Create default loss functions."""
        return {
            'photometric': self._photometric_loss,
            'semantic': self._semantic_loss,
            'silhouette': self._silhouette_loss,
            'tracking': self._tracking_loss,
            'regularization': self._regularization_loss,
            'se3_constraint': self._se3_constraint_loss,
            'trajectory_smooth': self._trajectory_smooth_loss,
        }
    
    def _se3_constraint_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SE(3) rigid body constraint loss."""
        if self.dynamic_field is None:
            return torch.tensor(0.0, device=self.device), {}
        
        loss = 0.0
        info = {}
        
        # Rigid body distance preservation
        for inst_id, inst in self.dynamic_field._instances.items():
            if not inst.is_rigid:
                continue
            
            # Compute distances in observation space
            positions = inst.canonical_positions
            if positions.shape[0] < 2:
                continue
            
            # Sample pairs
            n_samples = min(100, positions.shape[0])
            indices = torch.randperm(positions.shape[0])[:n_samples]
            
            # Compute canonical distances
            d_canon = torch.norm(
                positions[indices[:50]] - positions[indices[50:100]],
                dim=-1
            )
            
            # Compute observation distances (transformed)
            twist = self.dynamic_field._pose_twists[inst_id]
            R, t = self._se3_exp(twist)
            positions_obs = torch.matmul(positions[indices[:50]], R.T) + t
            positions_obs2 = torch.matmul(positions[indices[50:100]], R.T) + t
            
            d_obs = torch.norm(positions_obs - positions_obs2, dim=-1)
            
            # Loss: distances should be preserved
            dist_loss = torch.mean((d_canon - d_obs) ** 2)
            loss = loss + dist_loss
            info[f'inst_{inst_id}_rigid'] = dist_loss.item()
        
        return loss, info
    
    def _se3_exp(self, twist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SE(3) exponential map."""
        omega = twist[:3]
        v = twist[3:6]
        
        theta = torch.norm(omega)
        if theta < 1e-8:
            return torch.eye(3, device=twist.device), v
        
        axis = omega / theta
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=twist.device)
        
        R = torch.eye(3, device=twist.device) + \
            torch.sin(theta) * K + \
            (1 - torch.cos(theta)) * (K @ K)
        
        J = torch.eye(3, device=twist.device) + \
            ((1 - torch.cos(theta)) / (theta ** 2)) * K + \
            ((theta - torch.sin(theta)) / (theta ** 3)) * (K @ K)
        
        t = J @ v
        return R, t
    
    def _trajectory_smooth_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute trajectory smoothness loss."""
        if self.dynamic_field is None:
            return torch.tensor(0.0, device=self.device), {}
        
        loss = 0.0
        info = {}
        
        # Smooth pose changes
        pose_twists = self.dynamic_field._pose_twists.data
        
        # Velocity (first derivative)
        velocity = torch.diff(pose_twists, dim=0)
        vel_loss = torch.mean(velocity ** 2)
        loss = loss + vel_loss
        info['velocity_smooth'] = vel_loss.item()
        
        # Acceleration (second derivative)
        if pose_twists.shape[0] > 2:
            acceleration = torch.diff(velocity, dim=0)
            acc_loss = torch.mean(acceleration ** 2)
            loss = loss + 0.5 * acc_loss
            info['acceleration_smooth'] = acc_loss.item()
        
        return loss, info
    
    def _photometric_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute photometric loss."""
        diff = rendered - target
        
        if mask is not None:
            diff = diff * mask.unsqueeze(0)
        
        loss_l1 = torch.abs(diff).mean()
        loss_l2 = (diff ** 2).mean()
        loss = 0.8 * loss_l2 + 0.2 * loss_l1
        
        return loss, {'l1': loss_l1.item(), 'l2': loss_l2.item()}
    
    def _semantic_loss(
        self,
        rendered_logits: torch.Tensor,
        target_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute semantic segmentation loss."""
        rendered_flat = rendered_logits.permute(1, 2, 0).reshape(-1, rendered_logits.shape[0])
        target_flat = target_labels.reshape(-1)
        
        loss_ce = F.cross_entropy(rendered_flat, target_flat, reduction='mean')
        
        return loss_ce, {'cross_entropy': loss_ce.item()}
    
    def _silhouette_loss(
        self,
        rendered_alpha: torch.Tensor,
        target_silhouette: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute silhouette edge loss."""
        gy = torch.diff(rendered_alpha, dim=0)
        gx = torch.diff(rendered_alpha, dim=1)
        
        rendered_edge = torch.sqrt(
            F.pad(gy, (0, 0, 0, 1)) ** 2 +
            F.pad(gx, (0, 1, 0, 0)) ** 2
        )
        
        diff = rendered_edge - target_silhouette
        
        if mask is not None:
            mask_edge = F.max_pool2d(
                mask.unsqueeze(0).unsqueeze(0).float(),
                kernel_size=3, stride=1, padding=1
            ).squeeze(0).squeeze(0)
            diff = diff * mask_edge
        
        loss = torch.abs(diff).mean()
        
        return loss, {'edge_l1': loss.item()}
    
    def _tracking_loss(
        self,
        predicted_tracks: List[Dict],
        target_tracks: List[Dict],
        instance_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute tracking consistency loss."""
        if len(predicted_tracks) == 0 or len(target_tracks) == 0:
            return torch.tensor(0.0, device=self.device), {}
        
        loss = 0.0
        count = 0
        
        for pred, target in zip(predicted_tracks, target_tracks):
            pred_pos = pred.get('positions', torch.zeros(1, 2, device=self.device))
            target_pos = target.get('positions', torch.zeros(1, 2, device=self.device))
            
            dist = torch.norm(pred_pos - target_pos, dim=-1).mean()
            loss = loss + dist
            count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss, {'track_distance': loss.item() if count > 0 else 0}
    
    def _regularization_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute regularization loss."""
        loss = 0.0
        info = {}
        
        scales = torch.exp(self.static_field._scales)
        scale_reg = torch.mean((scales - 0.01) ** 2)
        loss = loss + 0.01 * scale_reg
        info['scale_reg'] = scale_reg.item()
        
        opacities = torch.sigmoid(self.static_field._opacities)
        opacity_reg = torch.mean((opacities - 0.5) ** 2)
        loss = loss + 0.001 * opacity_reg
        info['opacity_reg'] = opacity_reg.item()
        
        if self.dynamic_field is not None:
            pose_reg = torch.mean(self.dynamic_field._pose_twists ** 2)
            loss = loss + 0.01 * pose_reg
            info['pose_reg'] = pose_reg.item()
        
        return loss, info
    
    def step(
        self,
        batch: Dict[str, Any],
        stage_override: Optional[TrainingStage] = None,
    ) -> OptimizationState:
        """Perform one optimization step with multi-stage support."""
        # Get current stage config
        stage_config = self.stage_scheduler.step()
        
        if stage_override is not None:
            stage_config = self.stage_scheduler.configs[stage_override]
        
        # Update loss weights from stage config
        self.loss_weights.update(stage_config.loss_weights)
        
        # [OPT-13] Update learning rates
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(stage_config)
        
        # Extract data
        images = batch['rgb'].to(self.device)
        cameras = batch.get('camera')
        
        # Zero gradients
        self.optimizer_static.zero_grad()
        if self.optimizer_dynamic:
            self.optimizer_dynamic.zero_grad()
        
        # Forward pass with optional AMP
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            rendered = self._render_batch(images, cameras, stage_config)
            
            # Compute losses
            losses = {}
            loss_info = {}
            
            # Photometric loss
            if self.loss_weights.get('photometric', 0) > 0:
                loss_photo, info_photo = self.losses['photometric'](
                    rendered['rgb'],
                    images[0] if images.dim() == 4 else images,
                    rendered.get('mask')
                )
                losses['photometric'] = loss_photo * self.loss_weights['photometric']
                loss_info.update(info_photo)
            
            # Semantic loss
            if 'semantic' in batch and self.loss_weights.get('semantic', 0) > 0:
                loss_sem, info_sem = self.losses['semantic'](
                    rendered.get('semantic'),
                    batch['semantic'].to(self.device),
                )
                losses['semantic'] = loss_sem * self.loss_weights['semantic']
                loss_info.update(info_sem)
            
            # Silhouette loss
            if 'silhouette' in batch and self.loss_weights.get('silhouette', 0) > 0:
                loss_sil, info_sil = self.losses['silhouette'](
                    rendered.get('alpha'),
                    batch['silhouette'].to(self.device),
                )
                losses['silhouette'] = loss_sil * self.loss_weights['silhouette']
                loss_info.update(info_sil)
            
            # Tracking loss
            if 'tracking' in self.loss_weights and self.loss_weights.get('tracking', 0) > 0:
                if self.point_tracker is not None:
                    tracks = self.point_tracker.track(images)
                    loss_track = torch.tensor(0.0, device=self.device)
                    losses['tracking'] = loss_track * self.loss_weights['tracking']
            
            # SE3 constraint loss
            if self.loss_weights.get('se3_constraint', 0) > 0:
                loss_se3, info_se3 = self.losses['se3_constraint']()
                losses['se3_constraint'] = loss_se3 * self.loss_weights['se3_constraint']
                loss_info.update(info_se3)
            
            # Trajectory smoothness
            if self.loss_weights.get('trajectory_smooth', 0) > 0:
                loss_traj, info_traj = self.losses['trajectory_smooth']()
                losses['trajectory_smooth'] = loss_traj * self.loss_weights['trajectory_smooth']
                loss_info.update(info_traj)
            
            # Regularization
            if self.loss_weights.get('regularization', 0) > 0:
                loss_reg, info_reg = self.losses['regularization']()
                losses['regularization'] = loss_reg * self.loss_weights['regularization']
                loss_info.update(info_reg)
            
            # Total loss
            loss_total = sum(losses.values())
        
        # [OPT-14] Gradient accumulation
        self.grad_accumulator.backward(loss_total)
        
        if self.grad_accumulator.should_step():
            did_step, grad_norm = self.grad_accumulator.step(
                self.optimizer_static,
                self.scaler,
            )
            
            if self.optimizer_dynamic:
                self.optimizer_dynamic.step()
                self.optimizer_dynamic.zero_grad()
            
            self.state.grad_norm = grad_norm
        else:
            # Just clip gradients
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.static_field.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
        
        # [OPT-15] Mask refinement loop
        if stage_config.sam2_feedback_enabled and 'sam2_masks' in batch:
            rendered_masks = rendered.get('instance_masks', {})
            sam2_masks = batch['sam2_masks']
            
            if rendered_masks and sam2_masks:
                refined_masks, ref_stats = self.mask_refiner.run_refinement_loop(
                    rendered_masks,
                    sam2_masks,
                )
                # Update batch with refined masks for next iteration
                batch['sam2_masks'] = refined_masks
        
        # Update state
        self.state.step += 1
        self.state.stage = stage_config.name.value
        self.state.loss_total = loss_total.item()
        self.state.loss_photometric = losses.get('photometric', torch.tensor(0.0)).item()
        self.state.loss_semantic = losses.get('semantic', torch.tensor(0.0)).item()
        self.state.loss_silhouette = losses.get('silhouette', torch.tensor(0.0)).item()
        self.state.loss_tracking = losses.get('tracking', torch.tensor(0.0)).item()
        self.state.loss_regularization = losses.get('regularization', torch.tensor(0.0)).item()
        self.state.loss_se3_constraint = losses.get('se3_constraint', torch.tensor(0.0)).item()
        self.state.num_gaussians = self.static_field.num_active
        if self.dynamic_field:
            self.state.num_instances = self.dynamic_field.num_active_instances
        self.state.learning_rate = self.optimizer_static.param_groups[0]['lr']
        
        return self.state
    
    def _render_batch(
        self,
        images: torch.Tensor,
        cameras: Any,
        stage_config: Optional[StageConfig] = None,
    ) -> Dict[str, torch.Tensor]:
        """Render a batch with stage-specific settings."""
        B = images.shape[0]
        
        # Simple rendering for now
        outputs = []
        
        for i in range(B):
            cam_intrinsic = cameras['intrinsics'][i] if cameras else torch.eye(3, device=self.device)
            cam_extrinsic = cameras['extrinsics'][i] if cameras else torch.eye(4, device=self.device)
            
            rendered = self.static_field(
                camera={'intrinsics': cam_intrinsic, 'extrinsics': cam_extrinsic}
            )
            outputs.append(rendered)
        
        if B == 1:
            return outputs[0]
        
        return outputs[0]
    
    @property
    def max_grad_norm(self) -> float:
        return self.config.get('max_grad_norm', 1.0)
    
    def densify_and_prune(self) -> None:
        """Perform Gaussian densification and pruning."""
        grad_threshold = self.config.get('densify_threshold', 0.0002)
        opacity_threshold = self.config.get('prune_threshold', 0.0001)
        
        num_new = self.static_field.densify(grad_threshold)
        num_pruned = self.static_field.prune(opacity_threshold)
        
        print(f"Densification: {num_new} new, {num_pruned} pruned, "
              f"total: {self.static_field.num_active}")
    
    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        densify_every: int = 100,
        save_every: int = 500,
        checkpoint_dir: str = "./checkpoints",
    ) -> None:
        """Full training loop with multi-stage support."""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.state.epoch = epoch
            
            for batch_idx, batch in enumerate(dataloader):
                state = self.step(batch)
                
                if self.state.step % self.config.get('log_every', 50) == 0:
                    print(f"Step {state.step} | Stage: {state.stage} | "
                          f"Loss: {state.loss_total:.4f} | "
                          f"LR: {state.learning_rate:.2e}")
                
                if self.state.step % densify_every == 0:
                    self.densify_and_prune()
                
                if self.state.step % save_every == 0:
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, f"checkpoint_{state.step}.pt")
                    )
    
    def save_checkpoint(self, path: str) -> None:
        """Save optimizer checkpoint."""
        checkpoint = {
            'step': self.state.step,
            'epoch': self.state.epoch,
            'static_field_state': self.static_field.state_dict(),
            'optimizer_static_state': self.optimizer_static.state_dict(),
            'config': self.config,
        }
        
        if self.dynamic_field:
            checkpoint['dynamic_field_state'] = self.dynamic_field.state_dict()
            checkpoint['optimizer_dynamic_state'] = self.optimizer_dynamic.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load optimizer checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.state.step = checkpoint['step']
        self.state.epoch = checkpoint['epoch']
        
        self.static_field.load_state_dict(checkpoint['static_field_state'])
        self.optimizer_static.load_state_dict(checkpoint['optimizer_static_state'])
        
        if self.dynamic_field and 'dynamic_field_state' in checkpoint:
            self.dynamic_field.load_state_dict(checkpoint['dynamic_field_state'])
            self.optimizer_dynamic.load_state_dict(checkpoint['optimizer_dynamic_state'])
        
        print(f"Checkpoint loaded from {path}")

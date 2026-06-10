"""
Dynamic Object Field for 4D Gaussian Splatting - Optimized Version.

OPTIMIZATION CHANGELOG (v1.1.0):
  [OPT-8] 基于SAM2语义特征的刚体/非刚体自动分类器
  [OPT-9] 目标级高斯点分组机制（同一车辆共享SE(3)轨迹）
  [OPT-10] 规范空间(Canonical Space)初始化策略
  [OPT-11] 动态高斯点生命周期管理（目标进出画面时的高斯点增删）

Mathematical Foundation:
    Dynamic Object Representation:
        Each dynamic object i has:
        
        .. math::
            \mathcal{G}_i^{dyn} = \{(\mu_i^c, \Sigma_i^c, f_i^c, \xi_i(t))\}
            
        where:
            - μ_i^c is the canonical center
            - Σ_i^c is the canonical covariance
            - f_i^c is the appearance feature
            - ξ_i(t) ∈ se(3) is the pose twist at time t
    
    Instance Classification:
        p_rigid = sigmoid(W @ s_i + b)
        
        If p_rigid > τ_rigid: rigid body (vehicles)
        Else: non-rigid body (pedestrians)

@author Semantic 4DGS Team
@version 1.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import math


# ============================================================================
# [OPT-8] RIGID/NON-RIGID CLASSIFIER BASED ON SAM2 SEMANTIC FEATURES
# ============================================================================

class RigidNonRigidClassifier(nn.Module):
    r"""Classifier for determining if an object is rigid or non-rigid.
    
    OPTIMIZATION [OPT-8]: Uses SAM2 semantic features to automatically
    classify objects as rigid (vehicles) or non-rigid (pedestrians).
    
    The classifier learns to distinguish based on:
    - Shape consistency
    - Motion patterns
    - Semantic class from SAM2
    """
    
    def __init__(
        self,
        semantic_dim: int = 32,
        hidden_dim: int = 64,
        rigid_threshold: float = 0.5,
    ) -> None:
        """Initialize classifier.
        
        Args:
            semantic_dim: Dimension of semantic features
            hidden_dim: Hidden layer dimension
            rigid_threshold: Threshold for rigid classification
        """
        super().__init__()
        
        self.rigid_threshold = rigid_threshold
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, 1)
        
        # Confidence head for classification certainty
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        semantic_features: torch.Tensor,
        motion_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Classify objects as rigid or non-rigid.
        
        Args:
            semantic_features: Semantic features from SAM2 [..., D]
            motion_features: Optional motion features [..., M]
            
        Returns:
            Dictionary with:
                - is_rigid: Boolean mask [..., 1]
                - rigid_probability: Probability [..., 1]
                - confidence: Classification confidence [..., 1]
        """
        # Encode features
        x = self.encoder(semantic_features)
        
        if motion_features is not None:
            x = torch.cat([x, motion_features], dim=-1)
        
        # Classification
        logits = self.classifier(x)
        rigid_probability = torch.sigmoid(logits)
        is_rigid = rigid_probability > self.rigid_threshold
        
        # Confidence
        confidence = self.confidence_head(x)
        
        return {
            'is_rigid': is_rigid,
            'rigid_probability': rigid_probability,
            'confidence': confidence,
            'logits': logits,
        }
    
    def compute_classification_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        target_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        r"""Compute classification loss with optional confidence regularization.
        
        Args:
            predicted: Dict from forward pass
            target: Ground truth rigid labels [..., 1]
            target_confidence: Optional target confidence for uncertainty learning
            
        Returns:
            loss: Classification loss
            info: Loss components
        """
        # BCE loss for classification
        loss_ce = F.binary_cross_entropy_with_logits(
            predicted['logits'].squeeze(-1),
            target.squeeze(-1).float(),
        )
        
        loss = loss_ce
        
        # Confidence regularization (encourage high confidence)
        if target_confidence is not None:
            conf_diff = torch.abs(predicted['confidence'].squeeze(-1) - target_confidence)
            loss_conf = conf_diff.mean()
            loss = loss + 0.1 * loss_conf
        else:
            loss_conf = torch.tensor(0.0)
        
        # Penalize low confidence predictions
        low_conf_penalty = torch.mean(torch.clamp(0.5 - predicted['confidence'], min=0))
        loss = loss + 0.05 * low_conf_penalty
        
        info = {
            'classification_loss': loss_ce.item(),
            'confidence_loss': loss_conf.item() if isinstance(loss_conf, torch.Tensor) else loss_conf,
            'low_conf_penalty': low_conf_penalty.item(),
        }
        
        return loss, info


# ============================================================================
# [OPT-10] CANONICAL SPACE INITIALIZATION STRATEGIES
# ============================================================================

class CanonicalSpaceInitializer:
    r"""Initializer for canonical space of dynamic objects.
    
    OPTIMIZATION [OPT-10]: Provides multiple initialization strategies
    for the canonical space based on available information.
    """
    
    @staticmethod
    def initialize_from_bbox(
        positions: torch.Tensor,
        bbox_3d: Optional[torch.Tensor] = None,
        scale_factor: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Initialize canonical space from 3D bounding box.
        
        Args:
            positions: Observed positions [N, 3]
            bbox_3d: Optional 3D bounding box [2, 3]
            scale_factor: Scale factor for initialization
            
        Returns:
            canonical_positions: [N, 3]
            canonical_scales: [N, 3]
            canonical_rotations: [N, 4] (identity quaternions)
        """
        # Compute center
        center = positions.mean(dim=0, keepdim=True)
        
        # Canonical positions = observed - center
        canonical_positions = positions - center
        
        # Estimate scales from spread
        if bbox_3d is not None:
            # Use provided bounding box
            scales = (bbox_3d[1] - bbox_3d[0]) / 2 * scale_factor
        else:
            # Estimate from position variance
            std = positions.std(dim=0)
            scales = std * scale_factor
        
        # Expand scales to all points
        num_points = positions.shape[0]
        canonical_scales = scales.unsqueeze(0).expand(num_points, -1)
        
        # Identity rotations
        canonical_rotations = torch.zeros(num_points, 4, device=positions.device)
        canonical_rotations[:, 0] = 1.0  # w=1 for identity quaternion
        
        return canonical_positions, canonical_scales, canonical_rotations
    
    @staticmethod
    def initialize_from_pca(
        positions: torch.Tensor,
        n_components: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Initialize canonical space using PCA alignment.
        
        Args:
            positions: Observed positions [N, 3]
            n_components: Number of PCA components
            
        Returns:
            canonical_positions: Aligned positions [N, 3]
            scales: PCA scale factors [3]
            rotations: PCA rotation quaternions [4]
        """
        # Center positions
        center = positions.mean(dim=0)
        positions_centered = positions - center
        
        # PCA
        cov = positions_centered.T @ positions_centered / positions.shape[0]
        
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        except:
            # Fallback to identity
            return positions, torch.ones(3), torch.tensor([1., 0., 0., 0.])
        
        # Sort by eigenvalue descending
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, idx]
        
        # Rotation matrix to quaternion
        R = eigenvectors[:, :n_components]
        
        # Pad to 3x3 if needed
        if R.shape[1] < 3:
            R = torch.nn.functional.pad(R, (0, 3 - R.shape[1]))
        
        # Make proper rotation
        if torch.det(R) < 0:
            R[:, -1] *= -1
        
        # Transform positions
        canonical_positions = (R.T @ positions_centered.T).T
        
        # Scales
        scales = torch.sqrt(eigenvalues[idx[:n_components]])
        
        # Quaternion from rotation matrix
        q = rotation_matrix_to_quaternion(R.unsqueeze(0)).squeeze(0)
        
        return canonical_positions, scales, q
    
    @staticmethod
    def initialize_sphere_packing(
        num_points: int,
        radius: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Initialize with uniform sphere packing.
        
        Args:
            num_points: Number of Gaussians
            radius: Sphere radius
            device: Device
            
        Returns:
            positions: [num_points, 3]
            scales: [num_points, 3]
            rotations: [num_points, 4]
        """
        # Fibonacci sphere for uniform distribution
        phi = torch.arange(0, num_points, device=device) * (2.4 / num_points)
        y = 1 - phi / (num_points - 1) * 2
        radius_at_y = torch.sqrt(1 - y * y)
        
        theta = torch.acos(y)
        phi_angle = torch.arctan2(
            torch.sin(phi) * radius_at_y,
            torch.cos(phi) * radius_at_y
        )
        
        positions = torch.stack([
            torch.cos(phi_angle) * radius_at_y,
            y,
            torch.sin(phi_angle) * radius_at_y,
        ], dim=-1) * radius
        
        # Scale inversely with density
        scale = (4 * torch.pi * radius ** 2 / num_points) ** 0.5
        
        scales = torch.ones(num_points, 3, device=device) * scale * 0.1
        
        # Identity rotations
        rotations = torch.zeros(num_points, 4, device=device)
        rotations[:, 0] = 1.0
        
        return positions, scales, rotations


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    
    q = torch.zeros((*R.shape[:-2], 4), device=R.device, dtype=R.dtype)
    
    # Case 1
    s = torch.sqrt(trace + 1.0) * 2
    q[..., 0] = 0.25 * s
    q[..., 1] = (R[..., 2, 1] - R[..., 1, 2]) / s
    q[..., 2] = (R[..., 0, 2] - R[..., 2, 0]) / s
    q[..., 3] = (R[..., 1, 0] - R[..., 0, 1]) / s
    
    return F.normalize(q, dim=-1)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    norm = torch.norm(q, dim=-1, keepdim=True) + 1e-8
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    
    R = torch.zeros((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
    
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


# ============================================================================
# [OPT-9] OBJECT-LEVEL GAUSSIAN GROUPING MECHANISM
# ============================================================================

@dataclass
class ObjectGaussianGroup:
    r"""Group of Gaussians belonging to the same object.
    
    OPTIMIZATION [OPT-9]: All Gaussians in a group share the same
    SE(3) trajectory, ensuring rigid body consistency.
    
    Attributes:
        group_id: Unique identifier for this group
        gaussian_indices: Indices of Gaussians in this group
        rigid_probability: Probability of being a rigid body
        se3_twist: Current SE(3) twist [6]
        canonical_centers: Canonical positions relative to object center [N, 3]
        is_active: Whether this group is currently visible
        first_observed: First frame this group was observed
        last_observed: Last frame this group was observed
    """
    group_id: int
    gaussian_indices: torch.Tensor  # [N]
    rigid_probability: float
    se3_twist: torch.Tensor  # [6]
    canonical_centers: torch.Tensor  # [N, 3]
    is_active: bool = True
    first_observed: int = 0
    last_observed: int = 0
    appearance_features: Optional[torch.Tensor] = None  # [N, D]
    semantic_features: Optional[torch.Tensor] = None  # [N, D_sam]


class ObjectGaussianGrouper:
    r"""Manages Gaussian grouping by object.
    
    OPTIMIZATION [OPT-9]: Ensures all Gaussians belonging to the same
    object share the same SE(3) transformation trajectory.
    """
    
    def __init__(
        self,
        max_groups: int = 100,
        max_gaussians_per_group: int = 5000,
        iou_threshold: float = 0.3,
        semantic_threshold: float = 0.7,
    ) -> None:
        """Initialize grouper.
        
        Args:
            max_groups: Maximum number of object groups
            max_gaussians_per_group: Gaussians per object group
            iou_threshold: IoU threshold for merging groups
            semantic_threshold: Semantic similarity threshold
        """
        self.max_groups = max_groups
        self.max_gaussians_per_group = max_gaussians_per_group
        self.iou_threshold = iou_threshold
        self.semantic_threshold = semantic_threshold
        
        self._groups: Dict[int, ObjectGaussianGroup] = {}
        self._next_group_id = 0
        self._gaussian_to_group: Dict[int, int] = {}
        
        # SAM2 features buffer
        self._sam2_features_buffer: Optional[torch.Tensor] = None
        self._sam2_mask_buffer: Optional[torch.Tensor] = None
    
    def create_group(
        self,
        gaussian_indices: torch.Tensor,
        semantic_features: torch.Tensor,
        se3_twist: Optional[torch.Tensor] = None,
        rigid_probability: float = 1.0,
    ) -> int:
        r"""Create a new object group.
        
        Args:
            gaussian_indices: Indices of Gaussians in this group
            semantic_features: SAM2 semantic features [N, D]
            se3_twist: Initial SE(3) twist [6]
            rigid_probability: Initial rigid probability
            
        Returns:
            group_id: New group ID
        """
        group_id = self._next_group_id
        self._next_group_id += 1
        
        # Initialize canonical centers
        if se3_twist is not None:
            # Use first Gaussian center as reference
            # Canonical positions will be computed during optimization
            canonical_centers = torch.zeros(len(gaussian_indices), 3)
        else:
            canonical_centers = torch.zeros(len(gaussian_indices), 3)
        
        group = ObjectGaussianGroup(
            group_id=group_id,
            gaussian_indices=gaussian_indices,
            rigid_probability=rigid_probability,
            se3_twist=se3_twist if se3_twist is not None else torch.zeros(6),
            canonical_centers=canonical_centers,
            first_observed=0,
            last_observed=0,
            appearance_features=None,
            semantic_features=semantic_features,
        )
        
        self._groups[group_id] = group
        
        # Update mapping
        for idx in gaussian_indices.tolist():
            self._gaussian_to_group[idx] = group_id
        
        return group_id
    
    def merge_groups(
        self,
        group_id1: int,
        group_id2: int,
    ) -> Optional[int]:
        r"""Merge two groups into one.
        
        Args:
            group_id1: First group
            group_id2: Second group
            
        Returns:
            new_group_id: Merged group ID or None if merge fails
        """
        if group_id1 not in self._groups or group_id2 not in self._groups:
            return None
        
        group1 = self._groups[group_id1]
        group2 = self._groups[group_id2]
        
        # Check if merge is allowed (similar semantic features)
        if self._compute_similarity(group1, group2) < self.semantic_threshold:
            return None
        
        # Merge Gaussians
        all_indices = torch.cat([group1.gaussian_indices, group2.gaussian_indices])
        
        # Check capacity
        if len(all_indices) > self.max_gaussians_per_group:
            return None
        
        # Remove old groups
        self.remove_group(group_id1)
        self.remove_group(group_id2)
        
        # Create merged group
        merged_semantic = torch.cat([group1.semantic_features, group2.semantic_features], dim=0)
        merged_twist = (group1.se3_twist * len(group1.gaussian_indices) + 
                       group2.se3_twist * len(group2.gaussian_indices)) / len(all_indices)
        merged_rigid = max(group1.rigid_probability, group2.rigid_probability)
        
        new_group_id = self.create_group(
            gaussian_indices=all_indices,
            semantic_features=merged_semantic,
            se3_twist=merged_twist,
            rigid_probability=merged_rigid,
        )
        
        return new_group_id
    
    def _compute_similarity(
        self,
        group1: ObjectGaussianGroup,
        group2: ObjectGaussianGroup,
    ) -> float:
        """Compute semantic similarity between two groups."""
        if group1.semantic_features is None or group2.semantic_features is None:
            return 0.0
        
        # Mean feature comparison
        mean1 = group1.semantic_features.mean(dim=0)
        mean2 = group2.semantic_features.mean(dim=0)
        
        similarity = F.cosine_similarity(
            mean1.unsqueeze(0),
            mean2.unsqueeze(0)
        ).item()
        
        return similarity
    
    def assign_gaussians_to_group(
        self,
        gaussian_indices: torch.Tensor,
        group_id: int,
    ) -> None:
        """Assign Gaussians to an existing group."""
        if group_id not in self._groups:
            return
        
        group = self._groups[group_id]
        
        # Check capacity
        if len(group.gaussian_indices) + len(gaussian_indices) > self.max_gaussians_per_group:
            return
        
        # Update
        group.gaussian_indices = torch.cat([group.gaussian_indices, gaussian_indices])
        
        for idx in gaussian_indices.tolist():
            self._gaussian_to_group[idx] = group_id
    
    def remove_group(self, group_id: int) -> None:
        """Remove a group and clear Gaussian assignments."""
        if group_id not in self._groups:
            return
        
        group = self._groups[group_id]
        
        for idx in group.gaussian_indices.tolist():
            if idx in self._gaussian_to_group:
                del self._gaussian_to_group[idx]
        
        del self._groups[group_id]
    
    def get_group_for_gaussian(self, gaussian_idx: int) -> Optional[int]:
        """Get the group ID for a Gaussian."""
        return self._gaussian_to_group.get(gaussian_idx)
    
    def update_group_pose(
        self,
        group_id: int,
        se3_twist: torch.Tensor,
    ) -> None:
        """Update SE(3) pose for a group."""
        if group_id not in self._groups:
            return
        
        self._groups[group_id].se3_twist = se3_twist
    
    def get_active_groups(self) -> List[ObjectGaussianGroup]:
        """Get all active groups."""
        return [g for g in self._groups.values() if g.is_active]


# ============================================================================
# [OPT-11] DYNAMIC GAUSSIAN LIFECYCLE MANAGEMENT
# ============================================================================

class GaussianLifecycleManager:
    r"""Manages the lifecycle of dynamic Gaussians.
    
    OPTIMIZATION [OPT-11]: Handles addition and removal of Gaussians
    when objects enter or leave the scene.
    
    Lifecycle states:
        - APPEARING: New object entering, adding Gaussians
        - ACTIVE: Object visible, Gaussians stable
        - FADING: Object leaving, removing Gaussians
        - INACTIVE: Object hidden but tracked
    """
    
    APPEARING = "appearing"
    ACTIVE = "active"
    FADING = "fading"
    INACTIVE = "inactive"
    
    def __init__(
        self,
        min_appear_frames: int = 3,
        min_active_frames: int = 5,
        fadeout_frames: int = 10,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize lifecycle manager.
        
        Args:
            min_appear_frames: Minimum frames before object is confirmed
            min_active_frames: Minimum active frames before removal allowed
            fadeout_frames: Number of frames for fadeout
            confidence_threshold: Confidence threshold for detection
        """
        self.min_appear_frames = min_appear_frames
        self.min_active_frames = min_active_frames
        self.fadeout_frames = fadeout_frames
        self.confidence_threshold = confidence_threshold
        
        self._lifecycle_state: Dict[int, str] = {}
        self._frame_counts: Dict[int, int] = {}
        self._detection_confidence: Dict[int, float] = {}
    
    def register_object(
        self,
        object_id: int,
        initial_confidence: float = 0.5,
    ) -> str:
        r"""Register a new object in the lifecycle.
        
        Args:
            object_id: Unique object identifier
            initial_confidence: Initial detection confidence
            
        Returns:
            state: Initial lifecycle state
        """
        self._lifecycle_state[object_id] = self.APPEARING
        self._frame_counts[object_id] = 0
        self._detection_confidence[object_id] = initial_confidence
        
        return self.APPEARING
    
    def update_object(
        self,
        object_id: int,
        detection_confidence: float,
        visibility: bool,
    ) -> Tuple[str, bool, bool]:
        r"""Update object state for a new frame.
        
        Args:
            object_id: Object identifier
            detection_confidence: Detection confidence [0, 1]
            visibility: Whether object is currently visible
            
        Returns:
            state: Current lifecycle state
            should_add_gaussians: Whether to add new Gaussians
            should_remove_gaussians: Whether to remove Gaussians
        """
        if object_id not in self._lifecycle_state:
            return self.register_object(object_id, detection_confidence), False, False
        
        self._frame_counts[object_id] += 1
        self._detection_confidence[object_id] = detection_confidence
        
        current_state = self._lifecycle_state[object_id]
        
        # State machine transitions
        should_add = False
        should_remove = False
        
        if current_state == self.APPEARING:
            if self._frame_counts[object_id] >= self.min_appear_frames:
                if detection_confidence >= self.confidence_threshold:
                    self._lifecycle_state[object_id] = self.ACTIVE
                    should_add = True
        
        elif current_state == self.ACTIVE:
            if not visibility or detection_confidence < self.confidence_threshold * 0.5:
                self._lifecycle_state[object_id] = self.FADING
                self._frame_counts[object_id] = 0
        
        elif current_state == self.FADING:
            if visibility and detection_confidence >= self.confidence_threshold:
                # Object reappeared
                self._lifecycle_state[object_id] = self.ACTIVE
                self._frame_counts[object_id] = 0
            elif self._frame_counts[object_id] >= self.fadeout_frames:
                self._lifecycle_state[object_id] = self.INACTIVE
                should_remove = True
        
        elif current_state == self.INACTIVE:
            if visibility and detection_confidence >= self.confidence_threshold:
                # Object reappeared after being lost
                self._lifecycle_state[object_id] = self.APPEARING
                self._frame_counts[object_id] = 0
                should_add = True
        
        return self._lifecycle_state[object_id], should_add, should_remove
    
    def get_fadeout_alpha(
        self,
        object_id: int,
        total_fadeout_frames: int,
    ) -> float:
        r"""Get alpha value for fadeout effect.
        
        Args:
            object_id: Object identifier
            total_fadeout_frames: Total fadeout frames elapsed
            
        Returns:
            alpha: Alpha value [0, 1]
        """
        if self._lifecycle_state.get(object_id) != self.FADING:
            return 1.0
        
        progress = self._frame_counts.get(object_id, 0) / self.fadeout_frames
        return 1.0 - progress
    
    def get_state(self, object_id: int) -> Optional[str]:
        """Get current state of an object."""
        return self._lifecycle_state.get(object_id)
    
    def get_active_objects(self) -> List[int]:
        """Get list of active object IDs."""
        return [oid for oid, state in self._lifecycle_state.items() 
                if state in [self.APPEARING, self.ACTIVE]]
    
    def clear_object(self, object_id: int) -> None:
        """Clear object from lifecycle tracking."""
        if object_id in self._lifecycle_state:
            del self._lifecycle_state[object_id]
        if object_id in self._frame_counts:
            del self._frame_counts[object_id]
        if object_id in self._detection_confidence:
            del self._detection_confidence[object_id]


# ============================================================================
# ENHANCED DYNAMIC FIELD WITH ALL OPTIMIZATIONS
# ============================================================================

@dataclass
class DynamicInstance:
    """Enhanced representation of a dynamic object instance."""
    instance_id: int
    class_id: int
    canonical_positions: torch.Tensor
    canonical_scales: torch.Tensor
    canonical_rotations: torch.Tensor
    canonical_features: torch.Tensor
    pose_twist: torch.Tensor
    temporal_features: torch.Tensor
    tracking_confidence: float = 1.0
    is_visible: bool = True
    num_gaussians: int = 0
    bounding_box: Optional[torch.Tensor] = None
    is_rigid: bool = True  # [OPT-8] Rigid/non-rigid flag
    rigid_probability: float = 1.0  # [OPT-8] Classification confidence
    group_id: Optional[int] = None  # [OPT-9] Gaussian grouping


class DynamicField(nn.Module):
    r"""Enhanced dynamic foreground Gaussian field.
    
    OPTIMIZATION [OPT-8-11]:
        - [OPT-8] Automatic rigid/non-rigid classification
        - [OPT-9] Object-level Gaussian grouping
        - [OPT-10] Canonical space initialization strategies
        - [OPT-11] Lifecycle management for dynamic Gaussians
    """
    
    def __init__(
        self,
        max_instances: int = 100,
        max_gaussians_per_instance: int = 5000,
        feature_dim: int = 32,
        semantic_dim: int = 23,
        use_deformation: bool = False,
        temporal_feature_dim: int = 64,
        se3_config: Optional[Dict] = None,
        deform_config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        
        self.max_instances = max_instances
        self.max_gaussians_per_instance = max_gaussians_per_instance
        self.total_max_gaussians = max_instances * max_gaussians_per_instance
        self.feature_dim = feature_dim
        self.semantic_dim = semantic_dim
        self.use_deformation = use_deformation
        self.temporal_feature_dim = temporal_feature_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instance management
        self._instances: Dict[int, DynamicInstance] = {}
        self._next_instance_id = 0
        self._instance_order: List[int] = []
        
        # Gaussian buffers
        self._positions = nn.Parameter(
            torch.zeros(self.total_max_gaussians, 3, device=self.device),
            requires_grad=True
        )
        self._scales = nn.Parameter(
            torch.log(torch.ones(self.total_max_gaussians, 3, device=self.device) * 0.01),
            requires_grad=True
        )
        self._rotations = nn.Parameter(
            torch.zeros(self.total_max_gaussians, 4, device=self.device),
            requires_grad=True
        )
        self._opacities = nn.Parameter(
            torch.logit(torch.ones(self.total_max_gaussians, 1, device=self.device) * 0.5),
            requires_grad=True
        )
        self._features = nn.Parameter(
            torch.randn(self.total_max_gaussians, feature_dim, device=self.device) * 0.01,
            requires_grad=True
        )
        self._semantic_logits = nn.Parameter(
            torch.zeros(self.total_max_gaussians, semantic_dim, device=self.device),
            requires_grad=True
        )
        
        # Instance mapping
        self._instance_ids = nn.Parameter(
            torch.full((self.total_max_gaussians,), -1, dtype=torch.long, device=self.device),
            requires_grad=False
        )
        
        # SE(3) pose parameters
        self._pose_twists = nn.Parameter(
            torch.zeros(max_instances, 6, device=self.device),
            requires_grad=True
        )
        
        # Temporal features
        self._temporal_features = nn.Parameter(
            torch.zeros(max_instances, temporal_feature_dim, device=self.device),
            requires_grad=True
        )
        
        # Visibility and confidence
        self._visibility = nn.Parameter(
            torch.ones(max_instances, device=self.device),
            requires_grad=False
        )
        self._confidence = nn.Parameter(
            torch.ones(max_instances, device=self.device),
            requires_grad=True
        )
        
        # Class IDs
        self._class_ids = nn.Parameter(
            torch.zeros(max_instances, dtype=torch.long, device=self.device),
            requires_grad=False
        )
        
        # Gaussian ranges
        self._gaussian_ranges = torch.zeros(max_instances, 2, dtype=torch.long, device=self.device)
        
        # Initialize rotation to identity
        with torch.no_grad():
            self._rotations.data[:, 0] = 1.0
        
        # SE(3) transformation module [OPT-6]
        if se3_config is None:
            se3_config = {}
        self.se3_transform = SE3TransformModule(
            num_instances=max_instances,
            **se3_config
        ).to(self.device)
        
        # [OPT-8] Rigid/Non-rigid classifier
        self.rigid_classifier = RigidNonRigidClassifier(
            semantic_dim=semantic_dim,
            hidden_dim=64,
            rigid_threshold=0.5,
        ).to(self.device)
        
        # [OPT-9] Object-level Gaussian grouper
        self.gaussian_grouper = ObjectGaussianGrouper(
            max_groups=max_instances,
            max_gaussians_per_group=max_gaussians_per_instance,
        )
        
        # [OPT-10] Canonical space initializer
        self.canonical_initializer = CanonicalSpaceInitializer()
        
        # [OPT-11] Lifecycle manager
        self.lifecycle_manager = GaussianLifecycleManager(
            min_appear_frames=3,
            min_active_frames=5,
            fadeout_frames=10,
        )
        
        # Non-rigid deformation
        if use_deformation:
            if deform_config is None:
                deform_config = {}
            self.deformation = NonRigidDeformationModule(
                num_instances=max_instances,
                **deform_config
            ).to(self.device)
    
    @property
    def num_active_instances(self) -> int:
        return len(self._instances)
    
    def create_instance(
        self,
        class_id: int,
        positions: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        colors: Optional[torch.Tensor] = None,
        semantic_features: Optional[torch.Tensor] = None,
        bbox_3d: Optional[torch.Tensor] = None,
    ) -> int:
        r"""Create a new dynamic instance with enhanced initialization.
        
        OPTIMIZATION [OPT-10]: Uses canonical space initialization strategies.
        
        Args:
            class_id: Semantic class ID
            positions: Initial positions [N, 3]
            features: Optional appearance features [N, D]
            colors: Optional RGB colors [N, 3]
            semantic_features: SAM2 semantic features [N, D_sam]
            bbox_3d: Optional 3D bounding box [2, 3]
            
        Returns:
            instance_id: New instance ID
        """
        instance_id = self._next_instance_id
        self._next_instance_id += 1
        
        N = positions.shape[0]
        
        start_idx = self._find_available_slots(N)
        if start_idx < 0:
            return -1
        
        end_idx = start_idx + N
        
        # [OPT-10] Initialize canonical space
        canonical_positions, canonical_scales, canonical_rotations = \
            self.canonical_initializer.initialize_from_bbox(
                positions,
                bbox_3d=bbox_3d,
                scale_factor=1.2,
            )
        
        with torch.no_grad():
            # Initialize Gaussian properties
            self._positions[start_idx:end_idx] = positions
            self._scales[start_idx:end_idx] = torch.log(canonical_scales)
            self._rotations[start_idx:end_idx] = canonical_rotations
            
            if features is not None:
                self._features[start_idx:end_idx] = features[:N]
            elif colors is not None:
                self._features[start_idx:end_idx, :3] = colors[:N]
            
            self._instance_ids[start_idx:end_idx] = instance_id
            self._pose_twists.data[instance_id] = 0.0
            self._class_ids.data[instance_id] = class_id
            self._gaussian_ranges[instance_id] = torch.tensor([start_idx, end_idx])
        
        # [OPT-8] Classify rigid/non-rigid
        is_rigid = True
        rigid_probability = 0.8
        
        if semantic_features is not None:
            with torch.no_grad():
                mean_semantic = semantic_features.mean(dim=0, keepdim=True)
                classification = self.rigid_classifier(mean_semantic)
                is_rigid = classification['is_rigid'].item()
                rigid_probability = classification['rigid_probability'].item()
        
        # Create instance
        instance = DynamicInstance(
            instance_id=instance_id,
            class_id=class_id,
            canonical_positions=canonical_positions,
            canonical_scales=canonical_scales,
            canonical_rotations=canonical_rotations,
            canonical_features=self._features[start_idx:end_idx].detach().clone(),
            pose_twist=self._pose_twists.data[instance_id].detach().clone(),
            temporal_features=self._temporal_features.data[instance_id].detach().clone(),
            num_gaussians=N,
            is_rigid=is_rigid,
            rigid_probability=rigid_probability,
        )
        
        self._instances[instance_id] = instance
        self._instance_order.append(instance_id)
        
        # [OPT-9] Create Gaussian group
        gaussian_indices = torch.arange(start_idx, end_idx, device=self.device)
        group_id = self.gaussian_grouper.create_group(
            gaussian_indices=gaussian_indices,
            semantic_features=semantic_features if semantic_features is not None else torch.zeros(N, self.semantic_dim, device=self.device),
            se3_twist=self._pose_twists.data[instance_id],
            rigid_probability=rigid_probability,
        )
        instance.group_id = group_id
        
        # [OPT-11] Register in lifecycle
        self.lifecycle_manager.register_object(instance_id, initial_confidence=rigid_probability)
        
        return instance_id
    
    def _find_available_slots(self, num_needed: int) -> int:
        """Find available slots in Gaussian buffer."""
        for inst_id, inst in self._instances.items():
            start, end = self._gaussian_ranges[inst_id].tolist()
            existing = end - start
            available = self.max_gaussians_per_instance - existing
            if available >= num_needed:
                return end
        
        total_used = sum(inst.num_gaussians for inst in self._instances.values())
        if total_used + num_needed <= self.total_max_gaussians:
            if self._instances:
                max_end = max(
                    self._gaussian_ranges[inst_id][1].item()
                    for inst_id in self._instances
                )
                return max_end
            return 0
        
        return -1
    
    def update_pose(
        self,
        instance_id: int,
        twist: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
    ) -> None:
        """Update SE(3) pose for an instance."""
        if instance_id not in self._instances:
            return
        
        if twist is not None:
            self._pose_twists.data[instance_id] = twist
        elif velocity is not None:
            self._pose_twists.data[instance_id] += velocity * 0.016
        
        # [OPT-9] Update group pose
        if instance_id in self._instances:
            group_id = self._instances[instance_id].group_id
            if group_id is not None:
                self.gaussian_grouper.update_group_pose(
                    group_id,
                    self._pose_twists.data[instance_id]
                )
        
        self._instances[instance_id].pose_twist = \
            self._pose_twists.data[instance_id].detach().clone()
    
    def transform_instance(
        self,
        instance_id: int,
        timestamp: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform canonical Gaussians to observation frame."""
        if instance_id not in self._instances:
            return None, None
        
        instance = self._instances[instance_id]
        start, end = self._gaussian_ranges[instance_id].tolist()
        
        twist = self._pose_twists[instance_id]
        R, t = self._se3_exp(twist)
        
        # Transform positions
        positions_obs = torch.matmul(
            instance.canonical_positions,
            R.T
        ) + t
        
        # Transform rotations
        canonical_rot_mat = self._quaternion_to_matrix(instance.canonical_rotations)
        rotations_obs_mat = torch.matmul(R.unsqueeze(0), canonical_rot_mat)
        rotations_obs = self._matrix_to_quaternion(rotations_obs_mat)
        
        self._positions[start:end] = positions_obs
        self._rotations[start:end] = rotations_obs
        
        return positions_obs, rotations_obs
    
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
    
    def _quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix."""
        return quaternion_to_rotation_matrix(q)
    
    def _matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion."""
        return rotation_matrix_to_quaternion(R)
    
    def get_observations(
        self,
        timestamp: float,
        visible_only: bool = True,
    ) -> List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get observation-ready Gaussians for all instances."""
        observations = []
        
        for instance_id in self._instance_order:
            # [OPT-11] Check lifecycle state
            state = self.lifecycle_manager.get_state(instance_id)
            if visible_only and (state == self.lifecycle_manager.INACTIVE or 
                                  not self._visibility[instance_id]):
                continue
            
            # Apply fadeout if fading
            fade_alpha = 1.0
            if state == self.lifecycle_manager.FADING:
                fade_alpha = self.lifecycle_manager.get_fadeout_alpha(
                    instance_id, self.lifecycle_manager.fadeout_frames
                )
            
            self.transform_instance(instance_id, timestamp)
            
            start, end = self._gaussian_ranges[instance_id].tolist()
            
            observations.append((
                instance_id,
                self._positions[start:end],
                self._features[start:end],
                self._semantic_logits[start:end],
                fade_alpha,
            ))
        
        return observations
    
    def update_lifecycle(
        self,
        instance_id: int,
        detection_confidence: float,
        visibility: bool,
    ) -> Tuple[str, bool, bool]:
        """Update lifecycle state for an instance.
        
        OPTIMIZATION [OPT-11]: Lifecycle management.
        """
        state, should_add, should_remove = self.lifecycle_manager.update_object(
            instance_id,
            detection_confidence,
            visibility,
        )
        
        # Update visibility parameter
        self._visibility.data[instance_id] = float(visibility)
        
        return state, should_add, should_remove
    
    def compute_tracking_loss(
        self,
        predicted_poses: Dict[int, torch.Tensor],
        target_poses: Dict[int, torch.Tensor],
        weight: float = 0.1,
    ) -> torch.Tensor:
        """Compute pose tracking loss."""
        loss = 0.0
        
        for inst_id in predicted_poses.keys():
            if inst_id in target_poses:
                pred = predicted_poses[inst_id]
                target = target_poses[inst_id]
                loss = loss + torch.mean((pred - target) ** 2)
        
        return weight * loss
    
    def forward(
        self,
        timestamp: float,
        camera: "CameraParameters",
    ) -> Dict[str, torch.Tensor]:
        """Render dynamic field."""
        observations = self.get_observations(timestamp)
        
        H, W = camera.height, camera.width
        outputs = {}
        
        outputs['rgb'] = torch.zeros(3, H, W, device=self.device)
        outputs['semantic'] = torch.zeros(self.semantic_dim, H, W, device=self.device)
        outputs['depth'] = torch.zeros(H, W, device=self.device)
        outputs['num_instances'] = len(observations)
        
        return outputs
    
    def remove_instance(self, instance_id: int) -> None:
        """Remove an instance from the field."""
        if instance_id not in self._instances:
            return
        
        start, end = self._gaussian_ranges[instance_id].tolist()
        with torch.no_grad():
            self._positions[start:end] = 0
            self._scales[start:end] = 0
            self._rotations[start:end, 0] = 1
            self._opacities[start:end] = -10
            self._instance_ids[start:end] = -1
        
        # [OPT-9] Remove from grouper
        group_id = self._instances[instance_id].group_id
        if group_id is not None:
            self.gaussian_grouper.remove_group(group_id)
        
        # [OPT-11] Clear lifecycle
        self.lifecycle_manager.clear_object(instance_id)
        
        del self._instances[instance_id]
        if instance_id in self._instance_order:
            self._instance_order.remove(instance_id)
        
        self._visibility.data[instance_id] = 0.0


# Placeholder classes
SE3TransformModule = None
NonRigidDeformationModule = None


def _lazy_import():
    """Lazy import to avoid circular dependency."""
    global SE3TransformModule, NonRigidDeformationModule
    if SE3TransformModule is None:
        try:
            from .se3_transform import SE3Transform
            SE3TransformModule = SE3Transform
        except ImportError:
            SE3TransformModule = type('SE3Transform', (), {})
    if NonRigidDeformationModule is None:
        try:
            from .nonrigid_deform import NonRigidDeformation
            NonRigidDeformationModule = NonRigidDeformation
        except ImportError:
            NonRigidDeformationModule = type('NonRigidDeformation', (), {})

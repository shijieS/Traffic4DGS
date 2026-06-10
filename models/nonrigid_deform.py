"""
Semantic-4DGS-Traffic: Non-Rigid Deformation Module
Implements deformation field for dynamic objects with skeleton priors
Optimization items: 23-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class BoneStructure:
    """Represents skeletal structure for deformation prior"""
    joints: torch.Tensor           # [J, 3] joint positions in world space
    parents: torch.Tensor          # [J] parent joint indices (-1 for root)
    lengths: torch.Tensor          # [J] bone lengths
    rotations: torch.Tensor        # [J, 4] quaternion rotations per joint


class NonRigidDeformation(nn.Module):
    """
    Non-rigid deformation module for dynamic objects.
    
    Uses skeleton-based deformation for articulated objects (pedestrians, vehicles)
    with local stiffness constraints and temporal continuity.
    
    Optimization items:
    - Item 23: Skeleton prior-based pedestrian deformation
    - Item 24: Local stiffness constraint
    - Item 25: Temporal continuity constraint
    - Item 26: Deformation regularization
    """
    
    def __init__(
        self,
        num_gaussians: int = 5000,
        hidden_dim: int = 128,
        num_bones: int = 16,
        device: str = "cuda",
        stiffness_weight: float = 0.5,
        temporal_weight: float = 0.1,
        reg_weight: float = 0.01,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.hidden_dim = hidden_dim
        self.num_bones = num_bones
        self.device = device
        self.stiffness_weight = stiffness_weight
        self.temporal_weight = temporal_weight
        self.reg_weight = reg_weight
        
        # Initialize deformation network
        self._build_network()
        
        # Skeleton structure for articulated objects
        self.skeleton = None
        self.bone_weights = None  # [N, J] Gaussian-to-joint assignment weights
        
        # Temporal state
        self.prev_deformations = []
        self.temporal_window = 5
        
    def _build_network(self):
        """Build MLP network for deformation prediction"""
        # Position-based feature encoding
        self.feature_encoder = nn.Sequential(
            nn.Linear(3, self.hidden_dim),  # 3D position
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # Time encoding (for temporal smoothness)
        self.time_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim),  # 4D time encoding (sin/cos)
            nn.ReLU(),
        )
        
        # Deformation decoder (outputs SE(3) transformation)
        self.deformation_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 6),  # 3 translation + 3 rotation (axis-angle)
        )
        
        # Skeleton modifier (optional, for articulated objects)
        self.skeleton_encoder = nn.Sequential(
            nn.Linear(self.num_bones * 7, self.hidden_dim),  # bone pos + rot
            nn.ReLU(),
        )
        
    def initialize_skeleton(
        self,
        skeleton_type: str = "pedestrian",
        base_pose: Optional[torch.Tensor] = None,
    ):
        """
        Item 23: Initialize skeleton structure for deformation prior.
        
        Args:
            skeleton_type: Type of skeleton ("pedestrian", "vehicle", "custom")
            base_pose: Optional base pose for skeleton joints
        """
        if skeleton_type == "pedestrian":
            # SMPL-like skeleton with 16 joints
            num_joints = 16
            parents = torch.tensor([
                -1, 0, 1, 2, 3, 4,  # spine chain
                0, 0, 0,            # shoulders and head
                5, 6,               # left leg
                7, 8,               # right leg
                9, 10,              # left arm
                11, 12,             # right arm
            ], device=self.device)
            
            # Default joint positions (T-pose)
            if base_pose is None:
                base_pose = self._create_pedestrian_base_pose()
                
        elif skeleton_type == "vehicle":
            # Simple vehicle skeleton (4 wheels + body)
            num_joints = 8
            parents = torch.tensor([
                -1, 0, 0, 0,  # body center, front, rear, top
                1, 2, 3, 3,   # wheels
            ], device=self.device)
            
            if base_pose is None:
                base_pose = self._create_vehicle_base_pose()
                
        else:
            raise ValueError(f"Unknown skeleton type: {skeleton_type}")
            
        # Create skeleton structure
        self.skeleton = BoneStructure(
            joints=base_pose.clone(),
            parents=parents,
            lengths=self._compute_bone_lengths(base_pose, parents),
            rotations=torch.zeros(num_joints, 4, device=self.device),
        )
        
        print(f"Initialized {skeleton_type} skeleton with {num_joints} joints")
        
    def _create_pedestrian_base_pose(self) -> torch.Tensor:
        """Create base T-pose for pedestrian skeleton"""
        joints = torch.zeros(16, 3, device=self.device)
        # Spine
        joints[1] = torch.tensor([0, 0.2, 0], device=self.device)      # hips
        joints[2] = torch.tensor([0, 0.5, 0], device=self.device)      # spine
        joints[3] = torch.tensor([0, 0.8, 0], device=self.device)      # chest
        joints[4] = torch.tensor([0, 1.1, 0], device=self.device)      # neck
        # Head
        joints[5] = torch.tensor([0, 1.3, 0], device=self.device)      # head
        # Shoulders
        joints[6] = torch.tensor([-0.2, 1.0, 0], device=self.device)  # left shoulder
        joints[7] = torch.tensor([0.2, 1.0, 0], device=self.device)    # right shoulder
        # Arms
        joints[8] = torch.tensor([-0.4, 0.9, 0], device=self.device)  # left elbow
        joints[9] = torch.tensor([-0.5, 0.6, 0], device=self.device)  # left wrist
        joints[10] = torch.tensor([0.4, 0.9, 0], device=self.device)  # right elbow
        joints[11] = torch.tensor([0.5, 0.6, 0], device=self.device)  # right wrist
        # Legs
        joints[12] = torch.tensor([-0.1, 0.0, 0], device=self.device)  # left knee
        joints[13] = torch.tensor([-0.1, -0.5, 0], device=self.device)  # left ankle
        joints[14] = torch.tensor([0.1, 0.0, 0], device=self.device)  # right knee
        joints[15] = torch.tensor([0.1, -0.5, 0], device=self.device)  # right ankle
        return joints
        
    def _create_vehicle_base_pose(self) -> torch.Tensor:
        """Create base pose for vehicle skeleton"""
        joints = torch.zeros(8, 3, device=self.device)
        # Body center
        joints[0] = torch.tensor([0, 0.5, 0], device=self.device)
        # Front, rear, top
        joints[1] = torch.tensor([2, 0.5, 0], device=self.device)
        joints[2] = torch.tensor([-2, 0.5, 0], device=self.device)
        joints[3] = torch.tensor([0, 1.0, 0], device=self.device)
        # Wheels
        joints[4] = torch.tensor([1.5, 0.3, 1.0], device=self.device)   # front right
        joints[5] = torch.tensor([1.5, 0.3, -1.0], device=self.device)  # front left
        joints[6] = torch.tensor([-1.5, 0.3, 1.0], device=self.device)  # rear right
        joints[7] = torch.tensor([-1.5, 0.3, -1.0], device=self.device) # rear left
        return joints
        
    def _compute_bone_lengths(
        self, 
        joints: torch.Tensor, 
        parents: torch.Tensor
    ) -> torch.Tensor:
        """Compute bone lengths from joint positions"""
        lengths = torch.zeros(len(joints), device=self.device)
        for i, parent in enumerate(parents):
            if parent >= 0:
                lengths[i] = torch.norm(joints[i] - joints[parent])
        return lengths
        
    def assign_gaussians_to_skeleton(
        self,
        gaussian_positions: torch.Tensor,
        k_neighbors: int = 4,
    ):
        """
        Assign each Gaussian to nearest skeleton joints.
        
        Args:
            gaussian_positions: [N, 3] Gaussian positions
            k_neighbors: Number of nearest joints to consider
        """
        if self.skeleton is None:
            return
            
        joints = self.skeleton.joints  # [J, 3]
        
        # Compute distances to all joints
        distances = torch.cdist(gaussian_positions, joints)  # [N, J]
        
        # Get k nearest joints
        _, nearest_indices = torch.topk(distances, k=k_neighbors, largest=False, dim=-1)
        
        # Compute soft assignment weights (inverse distance)
        topk_distances = torch.gather(distances, 1, nearest_indices)  # [N, k]
        weights = 1.0 / (topk_distances + 1e-6)  # Inverse distance
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize
        
        # Store full assignment matrix [N, J] (for efficiency, store top-k separately)
        self.gaussian_positions = gaussian_positions
        self.nearest_joints = nearest_indices  # [N, k]
        self.joint_weights = weights  # [N, k]
        
    def forward(
        self,
        positions: torch.Tensor,
        timestamp: float,
        gaussian_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for non-rigid deformation.
        
        Args:
            positions: [N, 3] positions to deform
            timestamp: Current timestamp
            gaussian_indices: Optional indices for specific Gaussians
            
        Returns:
            Dictionary containing:
            - deformed_positions: Transformed positions
            - deformation_field: Full deformation vectors
            - se3_transforms: SE(3) transformations per Gaussian
        """
        batch_size = positions.shape[0]
        
        # Time encoding
        time_emb = self._encode_time(timestamp)  # [4]
        
        # Position encoding
        pos_emb = self.feature_encoder(positions)  # [N, hidden_dim]
        
        # Time encoding (broadcast)
        time_emb = time_emb.unsqueeze(0).expand(batch_size, -1)  # [N, 4]
        time_emb = self.time_encoder(time_emb)  # [N, hidden_dim]
        
        # Combine features
        combined = torch.cat([pos_emb, time_emb], dim=-1)  # [N, hidden_dim*2]
        
        # Skeleton modulation (if skeleton prior is used)
        if self.skeleton is not None and self.training:
            skeleton_feat = self._get_skeleton_features(positions, gaussian_indices)
            combined = combined + skeleton_feat
            
        # Predict deformation
        deformation = self.deformation_decoder(combined)  # [N, 6]
        
        # Split into translation and rotation
        translation = deformation[:, :3]  # [N, 3]
        rotation_axis_angle = deformation[:, 3:]  # [N, 3]
        
        # Apply skeleton constraints (if available)
        if self.skeleton is not None and self.nearest_joints is not None:
            translation, rotation_axis_angle = self._apply_skeleton_constraints(
                positions, translation, rotation_axis_angle, 
                gaussian_indices, timestamp
            )
            
        # Apply deformation to positions
        deformed_positions = positions + translation
        
        # Compute rotation matrix from axis-angle
        rotation_matrices = self._axis_angle_to_matrix(rotation_axis_angle)
        
        return {
            "deformed_positions": deformed_positions,
            "deformation_field": translation,
            "rotation_vectors": rotation_axis_angle,
            "rotation_matrices": rotation_matrices,
            "translation": translation,
        }
        
    def _encode_time(self, timestamp: float) -> torch.Tensor:
        """Encode time using sinusoidal features"""
        t = timestamp * torch.ones(4, device=self.device)
        t[[1, 3]] = t[[1, 3]] * 2 * np.pi
        return torch.sin(t)
        
    def _get_skeleton_features(
        self,
        positions: torch.Tensor,
        gaussian_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Get skeleton-based features for Gaussian positions"""
        if self.nearest_joints is None:
            return torch.zeros(positions.shape[0], self.hidden_dim, device=self.device)
            
        # Get nearest joints and weights
        nearest = self.nearest_joints  # [N, k]
        weights = self.joint_weights  # [N, k]
        
        # Get joint positions
        joints = self.skeleton.joints[nearest]  # [N, k, 3]
        
        # Compute weighted joint features
        joint_feat = self.skeleton_encoder(
            joints.view(-1, k * 3)
        )  # [N, hidden_dim]
        
        return joint_feat
        
    def _apply_skeleton_constraints(
        self,
        positions: torch.Tensor,
        translation: torch.Tensor,
        rotation: torch.Tensor,
        gaussian_indices: Optional[torch.Tensor],
        timestamp: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Item 24: Apply local stiffness constraint.
        
        Ensures nearby Gaussians have similar deformations.
        
        Math: L_stiffness = sum_{i,j neighbors} w_{ij} * ||d_i - d_j||^2
        """
        # Build local stiffness based on Gaussian proximity
        k = 4  # Number of neighbors for stiffness
        distances = torch.cdist(positions, positions)  # [N, N]
        
        # Get k nearest (excluding self)
        _, nearest_idx = torch.topk(distances + torch.eye(len(positions), device=self.device) * 1e6, 
                                      k=k, largest=False, dim=-1)
        
        # Compute stiffness loss (applied as gradient penalty)
        translation_diff = translation.unsqueeze(1) - translation[nearest_idx]  # [N, k, 3]
        rotation_diff = rotation.unsqueeze(1) - rotation[nearest_idx]  # [N, k, 3]
        
        # Weight by inverse distance
        weights = 1.0 / (distances.gather(1, nearest_idx) + 1e-6)  # [N, k]
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Store for loss computation
        self._stiffness_data = {
            "translation_diff": translation_diff.detach(),
            "rotation_diff": rotation_diff.detach(),
            "weights": weights.detach(),
        }
        
        return translation, rotation
        
    def stiffness_loss(self) -> torch.Tensor:
        """
        Item 24: Compute local stiffness constraint loss.
        
        Penalizes large differences in deformation between nearby Gaussians.
        """
        if not hasattr(self, "_stiffness_data"):
            return torch.tensor(0.0, device=self.device)
            
        diff = self._stiffness_data
        trans_diff = diff["translation_diff"]  # [N, k, 3]
        rot_diff = diff["rotation_diff"]  # [N, k, 3]
        weights = diff["weights"]  # [N, k]
        
        # Weighted sum of squared differences
        trans_loss = (weights.unsqueeze(-1) * trans_diff.pow(2)).sum() / weights.numel()
        rot_loss = (weights.unsqueeze(-1) * rot_diff.pow(2)).sum() / weights.numel()
        
        return self.stiffness_weight * (trans_loss + rot_loss)
        
    def temporal_continuity_loss(self) -> torch.Tensor:
        """
        Item 25: Temporal continuity constraint for deformation.
        
        Ensures smooth deformation over time.
        
        Math: L_temporal = sum_t ||d_t - d_{t-1}||^2
        """
        if len(self.prev_deformations) == 0:
            return torch.tensor(0.0, device=self.device)
            
        prev_deform = self.prev_deformations[-1]  # Most recent deformation
        
        if not hasattr(self, "_current_deformation"):
            return torch.tensor(0.0, device=self.device)
            
        current = self._current_deformation
        
        # Position smoothness
        pos_diff = F.mse_loss(current["translation"], prev_deform["translation"])
        
        # Rotation smoothness
        rot_diff = F.mse_loss(current["rotation_vectors"], prev_deform["rotation_vectors"])
        
        return self.temporal_weight * (pos_diff + rot_diff)
        
    def deformation_regularization(self) -> torch.Tensor:
        """
        Item 26: Regularization to prevent excessive deformation.
        
        Math: L_reg = ||d||^2 + lambda * ||J(d)||^2
        
        where d is deformation and J is Jacobian.
        """
        if not hasattr(self, "_current_deformation"):
            return torch.tensor(0.0, device=self.device)
            
        current = self._current_deformation
        
        # L2 regularization on deformation magnitude
        trans_reg = current["translation"].pow(2).mean()
        rot_reg = current["rotation_vectors"].pow(2).mean()
        
        return self.reg_weight * (trans_reg + rot_reg)
        
    def _axis_angle_to_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle to rotation matrix using Rodrigues' formula"""
        # Simplified: use torch's built-in rotation conversion
        try:
            from torchGeometry import axis_angle_to_matrix
            return axis_angle_to_matrix(axis_angle)
        except ImportError:
            # Fallback: small angle approximation
            batch_size = axis_angle.shape[0]
            angle = torch.norm(axis_angle, dim=-1, keepdim=True) + 1e-6
            axis = axis_angle / angle
            # Simplified Rodrigues
            I = torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(batch_size, -1, -1)
            skew = self._skew_symmetric(axis)
            angle = angle.squeeze(-1)
            R = I + torch.sin(angle).unsqueeze(-1).unsqueeze(-1) * skew + \
                (1 - torch.cos(angle)).unsqueeze(-1).unsqueeze(-1) * (skew @ skew)
            return R
            
    def _skew_symmetric(self, v: torch.Tensor) -> torch.Tensor:
        """Create skew-symmetric matrix from vector"""
        batch = v.shape[0]
        skew = torch.zeros(batch, 3, 3, device=v.device)
        skew[:, 0, 1] = -v[:, 2]
        skew[:, 0, 2] = v[:, 1]
        skew[:, 1, 0] = v[:, 2]
        skew[:, 1, 2] = -v[:, 0]
        skew[:, 2, 0] = -v[:, 1]
        skew[:, 2, 1] = v[:, 0]
        return skew
        
    def update_temporal_state(self):
        """Store current deformation for temporal continuity"""
        if hasattr(self, "_current_deformation"):
            self.prev_deformations.append({
                "translation": self._current_deformation["translation"].detach().clone(),
                "rotation_vectors": self._current_deformation["rotation_vectors"].detach().clone(),
            })
            
            # Maintain window size
            if len(self.prev_deformations) > self.temporal_window:
                self.prev_deformations.pop(0)
                
    def get_total_loss(self) -> torch.Tensor:
        """Compute all deformation losses"""
        total = torch.tensor(0.0, device=self.device)
        total = total + self.stiffness_loss()
        total = total + self.temporal_continuity_loss()
        total = total + self.deformation_regularization()
        return total
        
    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Return all optimizable parameters"""
        return list(self.parameters())
        
    def state_dict(self) -> Dict:
        """Return state dict for checkpointing"""
        return {
            **super().state_dict(),
            "skeleton_joints": self.skeleton.joints if self.skeleton else None,
            "skeleton_parents": self.skeleton.parents if self.skeleton else None,
            "prev_deformations": self.prev_deformations,
        }
        
    def load_state_dict(self, state_dict: Dict):
        """Load state dict from checkpoint"""
        super().load_state_dict(state_dict)
        # Restore skeleton if available
        if state_dict.get("skeleton_joints") is not None:
            self.skeleton.joints = state_dict["skeleton_joints"]
            self.skeleton.parents = state_dict["skeleton_parents"]
        self.prev_deformations = state_dict.get("prev_deformations", [])


class DeformationOptimizer:
    """Wrapper for optimizing deformation parameters"""
    
    def __init__(
        self,
        deformation_model: NonRigidDeformation,
        lr: float = 1e-4,
    ):
        self.model = deformation_model
        self.optimizer = torch.optim.Adam(deformation_model.parameters(), lr=lr)
        
    def step(self) -> Dict[str, float]:
        """Perform optimization step"""
        self.optimizer.zero_grad()
        
        # Compute loss
        total_loss = self.model.get_total_loss()
        
        # Backward
        total_loss.backward()
        self.optimizer.step()
        
        # Update temporal state
        self.model.update_temporal_state()
        
        return {
            "total_loss": total_loss.item(),
        }

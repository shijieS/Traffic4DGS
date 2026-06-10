"""
4D Gaussian Splatting Field - Base Class.

This module implements the core 4D Gaussian representation used in our
semantic-aware traffic scene reconstruction.

Mathematical Foundation:
    3D Gaussian: G(x) = exp(-0.5 * (x-μ)ᵀΣ⁻¹(x-μ))
    
    Covariance Matrix Decomposition:
        Σ = R @ S @ Sᵀ @ Rᵀ
        
    where:
        - R ∈ SO(3) is a rotation matrix (3×3 orthogonal)
        - S = diag(s) is a diagonal scale matrix
        
    4D Extension with Temporal Dimension:
        For time t, the 4D position is [x, y, z, t]
        The spatio-temporal covariance is:
        
        Σ₄ = | Σₛ   Σₛₜ |
             | Σₜₛ  σₜ² |
             
        where Σₛ ∈ ℝ³ˣ³ is spatial covariance,
        Σₛₜ ∈ ℝ³ˣ¹ is space-time cross-covariance,
        σₜ² is temporal variance.

Gaussian Properties:
    - Position μ ∈ ℝ³: 3D center of Gaussian
    - Scale s ∈ ℝ³⁺: Per-axis scale (positive)
    - Rotation R ∈ SO(3): Orientation as 3×3 rotation matrix
    - Opacity α ∈ [0,1]: Transparency
    - Features f ∈ ℝᴰ: Appearance/semantic features

Rendering Equation:
    C(r) = Σᵢ cᵢ αᵢ ∏ⱼ<ᵢ (1 - αⱼ)
    
    where:
        - C(r) is the rendered color at pixel r
        - cᵢ = sh_i(view_direction) @ f_i for view-dependent color
        - αᵢ = opacity_i × G_2D(Σ_camera) for 2D projection

@author Semantic 4DGS Team
@version 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum


class GaussianProperty(Enum):
    """Enumeration of Gaussian primitive properties."""
    POSITION = "position"           # μ ∈ ℝ³
    SCALE = "scale"                  # s ∈ ℝ³⁺
    ROTATION = "rotation"            # R ∈ SO(3) as quaternion
    OPACITY = "opacity"             # α ∈ [0,1]
    FEATURES = "features"           # f ∈ ℝᴰ
    SEMANTIC_FEATURES = "semantic"  # s_f ∈ ℝᴷ (K classes)
    TEMPORAL_OFFSET = "temporal"    # t_offset ∈ ℝ


@dataclass
class GaussianPrimitive:
    """Single Gaussian primitive representation.
    
    A Gaussian primitive is defined by its mean (position) and covariance,
    along with appearance and semantic properties.
    
    Attributes:
        position: 3D center position μ ∈ ℝ³
        scale: Per-axis scale s ∈ ℝ³⁺ (must be positive)
        rotation: Quaternion (w, x, y, z) representing R ∈ SO(3)
        opacity: Opacity α ∈ [0,1]
        features: Appearance features (RGB or SH coefficients)
        semantic_features: Semantic class scores (K classes)
        temporal_offset: Temporal offset from canonical frame
        instance_id: Instance ID for multi-object tracking (optional)
        class_id: Semantic class ID (optional)
    """
    position: torch.Tensor      # [N, 3]
    scale: torch.Tensor         # [N, 3]
    rotation: torch.Tensor      # [N, 4] quaternion (w, x, y, z)
    opacity: torch.Tensor       # [N, 1]
    features: torch.Tensor      # [N, D] or [N, 3, K] for SH
    semantic_features: Optional[torch.Tensor] = None  # [N, K]
    temporal_offset: Optional[torch.Tensor] = None     # [N, 1]
    instance_id: Optional[torch.Tensor] = None         # [N, 1]
    class_id: Optional[torch.Tensor] = None            # [N, 1]
    
    @property
    def num_gaussians(self) -> int:
        """Return number of Gaussians."""
        return self.position.shape[0]
    
    @property
    def device(self) -> torch.device:
        """Return device of the Gaussian primitives."""
        return self.position.device
    
    def to(self, device: torch.device) -> "GaussianPrimitive":
        """Move all tensors to specified device."""
        return GaussianPrimitive(
            position=self.position.to(device),
            scale=self.scale.to(device),
            rotation=self.rotation.to(device),
            opacity=self.opacity.to(device),
            features=self.features.to(device),
            semantic_features=self.semantic_features.to(device) if self.semantic_features is not None else None,
            temporal_offset=self.temporal_offset.to(device) if self.temporal_offset is not None else None,
            instance_id=self.instance_id.to(device) if self.instance_id is not None else None,
            class_id=self.class_id.to(device) if self.class_id is not None else None,
        )
    
    def detach(self) -> "GaussianPrimitive":
        """Detach all tensors from computation graph."""
        return GaussianPrimitive(
            position=self.position.detach(),
            scale=self.scale.detach(),
            rotation=self.rotation.detach(),
            opacity=self.opacity.detach(),
            features=self.features.detach(),
            semantic_features=self.semantic_features.detach() if self.semantic_features is not None else None,
            temporal_offset=self.temporal_offset.detach() if self.temporal_offset is not None else None,
            instance_id=self.instance_id.detach() if self.instance_id is not None else None,
            class_id=self.class_id.detach() if self.class_id is not None else None,
        )
    
    def clone(self) -> "GaussianPrimitive":
        """Create a deep copy of the Gaussian primitives."""
        return GaussianPrimitive(
            position=self.position.clone(),
            scale=self.scale.clone(),
            rotation=self.rotation.clone(),
            opacity=self.opacity.clone(),
            features=self.features.clone(),
            semantic_features=self.semantic_features.clone() if self.semantic_features is not None else None,
            temporal_offset=self.temporal_offset.clone() if self.temporal_offset is not None else None,
            instance_id=self.instance_id.clone() if self.instance_id is not None else None,
            class_id=self.class_id.clone() if self.class_id is not None else None,
        )


class CameraParameters(NamedTuple):
    """Camera intrinsic and extrinsic parameters.
    
    Attributes:
        intrinsics: Camera intrinsic matrix K [3, 3]
        extrinsics: Camera extrinsic matrix [4, 4] (world to camera)
        width: Image width
        height: Image height
        timestamp: Frame timestamp (for 4D alignment)
    """
    intrinsics: torch.Tensor      # [3, 3]
    extrinsics: torch.Tensor     # [4, 4]
    width: int
    height: int
    timestamp: Optional[float] = None


class GaussianField(nn.Module):
    r"""Base class for 4D Gaussian Splatting field.
    
    This class implements the core 4D Gaussian representation with:
    - Spatio-temporal covariance modeling
    - Differentiable rendering
    - Adaptive density control
    
    Mathematical Formulation:
        
        The 4D Gaussian in world coordinates [x, y, z, t] is defined as:
        
        .. math::
            G_4(\mathbf{x}, t) = \exp\left(-\frac{1}{2}
            \begin{bmatrix} \mathbf{x} - \boldsymbol{\mu}_s \\ 
            t - t_0 \end{bmatrix}^\top
            \boldsymbol{\Sigma}_4^{-1}
            \begin{bmatrix} \mathbf{x} - \boldsymbol{\mu}_s \\ 
            t - t_0 \end{bmatrix}\right)
        
        The covariance is decomposed as:
        
        .. math::
            \boldsymbol{\Sigma}_4 = 
            \begin{bmatrix} \mathbf{R} & \mathbf{0} \\ \mathbf{0} & 1 \end{bmatrix}
            \begin{bmatrix} \mathbf{S}\mathbf{S}^\top & \mathbf{S}\mathbf{c} \\ 
            \mathbf{c}^\top\mathbf{S} & \sigma_t^2 \end{bmatrix}
            \begin{bmatrix} \mathbf{R}^\top & \mathbf{0} \\ \mathbf{0} & 1 \end{bmatrix}
        
        where:
            - :math:`\mathbf{R} \in SO(3)` is the rotation matrix
            - :math:`\mathbf{S} = \text{diag}(s_x, s_y, s_z)` is the scale
            - :math:`\mathbf{c}` is the space-time correlation
            - :math:`\sigma_t^2` is the temporal variance
    
    Args:
        num_gaussians: Initial number of Gaussian primitives
        feature_dim: Dimension of appearance features
        semantic_feature_dim: Dimension of semantic features
        init_scale: Initial scale for Gaussians
        spatial_bound: Spatial bounds for initialization
        device: Device for computation
        
    Example:
        >>> field = GaussianField(num_gaussians=5000, feature_dim=32)
        >>> gaussians = field.initialize_from_pcd(point_cloud)
        >>> rendered = field.render(gaussians, camera)
    """
    
    def __init__(
        self,
        num_gaussians: int = 5000,
        feature_dim: int = 32,
        semantic_feature_dim: int = 256,
        init_scale: float = 0.01,
        spatial_bound: float = 10.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim
        self.semantic_feature_dim = semantic_feature_dim
        self.init_scale = init_scale
        self.spatial_bound = spatial_bound
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Active Gaussians count
        self._num_active_gaussians = nn.Parameter(
            torch.tensor(num_gaussians, dtype=torch.int32),
            requires_grad=False
        )
        
        # Gaussian properties (all optimizable)
        self._positions = nn.Parameter(
            torch.zeros(num_gaussians, 3, device=self.device),
            requires_grad=True
        )
        self._scales = nn.Parameter(
            torch.log(torch.ones(num_gaussians, 3, device=self.device) * init_scale),
            requires_grad=True
        )
        self._rotations = nn.Parameter(
            torch.zeros(num_gaussians, 4, device=self.device),  # quaternion
            requires_grad=True
        )
        self._opacities = nn.Parameter(
            torch.logit(torch.ones(num_gaussians, 1, device=self.device) * 0.5),
            requires_grad=True
        )
        self._features = nn.Parameter(
            torch.randn(num_gaussians, feature_dim, device=self.device) * 0.01,
            requires_grad=True
        )
        self._semantic_features = nn.Parameter(
            torch.randn(num_gaussians, semantic_feature_dim, device=self.device) * 0.01,
            requires_grad=True
        )
        self._temporal_offsets = nn.Parameter(
            torch.zeros(num_gaussians, 1, device=self.device),
            requires_grad=True
        )
        
        # Instance and class IDs (non-optimizable)
        self._instance_ids = nn.Parameter(
            torch.arange(num_gaussians, device=self.device).unsqueeze(1).float(),
            requires_grad=False
        )
        self._class_ids = nn.Parameter(
            torch.zeros(num_gaussians, 1, device=self.device, dtype=torch.long),
            requires_grad=False
        )
        
        # Mask for active Gaussians
        self._active_mask = nn.Parameter(
            torch.zeros(num_gaussians, dtype=torch.bool, device=self.device),
            requires_grad=False
        )
        
        # Initialize rotations to identity quaternions
        with torch.no_grad():
            self._rotations.data[:, 0] = 1.0  # w=1 for identity quaternion
    
    @property
    def positions(self) -> torch.Tensor:
        """Get positions of all Gaussians."""
        return self._positions
    
    @property
    def scales(self) -> torch.Tensor:
        """Get scales (in log space) of all Gaussians."""
        return self._scales
    
    @property
    def rotations(self) -> torch.Tensor:
        """Get rotations (quaternions) of all Gaussians."""
        return self._rotations
    
    @property
    def opacities(self) -> torch.Tensor:
        """Get opacities (in log space) of all Gaussians."""
        return self._opacities
    
    @property
    def features(self) -> torch.Tensor:
        """Get appearance features."""
        return self._features
    
    @property
    def semantic_features(self) -> torch.Tensor:
        """Get semantic features."""
        return self._semantic_features
    
    @property
    def num_active(self) -> int:
        """Get number of active Gaussians."""
        return int(self._active_mask.sum().item())
    
    @property
    def active_mask(self) -> torch.Tensor:
        """Get boolean mask for active Gaussians."""
        return self._active_mask
    
    def get_covariance(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Compute 3D covariance matrices from scale and rotation.
        
        Mathematically:
            Σ = R @ S @ Sᵀ @ Rᵀ
            
        where:
            - R = rotation matrix from quaternion
            - S = diag(exp(scales)) (ensure positive scales)
        
        Args:
            indices: Optional indices to select subset of Gaussians
            
        Returns:
            Covariance matrices [N, 3, 3]
        """
        if indices is not None:
            scales = torch.exp(self._scales[indices])  # [N, 3]
            rotations = self._rotations[indices]         # [N, 4]
        else:
            scales = torch.exp(self._scales)  # [N, 3]
            rotations = self._rotations        # [N, 4]
        
        # Build rotation matrix from quaternion
        # q = [w, x, y, z]
        w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        
        # Rotation matrix
        R = torch.zeros(rotations.shape[0], 3, 3, device=rotations.device)
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - x*w)
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)
        
        # Scale matrix
        S = torch.diag_embed(scales)  # [N, 3, 3]
        
        # Covariance: Σ = R @ S @ Sᵀ @ Rᵀ = R @ diag(s^2) @ Rᵀ
        SS = S @ S.transpose(-2, -1)  # [N, 3, 3] = diag(s^2)
        Sigma = R @ SS @ R.transpose(-2, -1)
        
        return Sigma
    
    def get_gaussians(self) -> GaussianPrimitive:
        """Get current Gaussian primitives.
        
        Returns:
            GaussianPrimitive containing all properties
        """
        return GaussianPrimitive(
            position=self._positions.detach(),
            scale=torch.exp(self._scales.detach()),
            rotation=self._rotations.detach(),
            opacity=torch.sigmoid(self._opacities.detach()),
            features=self._features.detach(),
            semantic_features=self._semantic_features.detach(),
            temporal_offset=self._temporal_offsets.detach(),
            instance_id=self._instance_ids.detach(),
            class_id=self._class_ids.detach(),
        )
    
    def initialize_from_pcd(
        self,
        points: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        colors: Optional[torch.Tensor] = None,
    ) -> None:
        r"""Initialize Gaussians from point cloud.
        
        Uses K-nearest neighbors to initialize Gaussian primitives from a point cloud.
        
        Args:
            points: Point cloud [N, 3] in world coordinates
            features: Optional point features [N, D]
            colors: Optional RGB colors [N, 3]
        """
        N = points.shape[0]
        
        # Initialize positions from points
        if N > self.num_gaussians:
            # Subsample
            indices = torch.randperm(N)[:self.num_gaussians]
            self._positions.data = points[indices].to(self.device)
            N = self.num_gaussians
        else:
            self._positions.data[:N] = points.to(self.device)
        
        # Initialize scales based on nearest neighbor distances
        with torch.no_grad():
            from scipy.spatial import cKDTree
            import numpy as np
            
            points_np = points.cpu().numpy()
            if N > 3:
                tree = cKDTree(points_np)
                distances, _ = tree.query(points_np, k=4)
                median_dist = np.median(distances[:, 1:])  # Exclude self
                init_scales = np.ones((N, 3)) * (median_dist * 0.5)
            else:
                init_scales = np.ones((N, 3)) * self.init_scale
            
            self._scales.data[:N] = torch.log(
                torch.from_numpy(init_scales).float().to(self.device)
            )
        
        # Initialize rotations to identity
        self._rotations.data[:N, 0] = 1.0
        
        # Initialize features
        if features is not None:
            self._features.data[:N] = features[:N].to(self.device)
        if colors is not None:
            # Use colors as first 3 dimensions of features
            self._features.data[:N, :3] = colors[:N].to(self.device)
        
        # Set active mask
        self._active_mask[:N] = True
        self._num_active_gaussians.data.fill_(N)
    
    def densify(
        self,
        grad_threshold: float = 0.0002,
        max_gaussians: Optional[int] = None,
    ) -> None:
        r"""Densify Gaussians by splitting large Gaussians.
        
        Gaussians with large gradients (indicating poor approximation)
        are split into smaller Gaussians.
        
        Mathematically:
            For a Gaussian with high gradient, clone it and offset
            by the gradient direction scaled by the Gaussian's scale.
            
            μ_new = μ ± grad / ||grad|| * scale
            
        Args:
            grad_threshold: Gradient threshold for densification
            max_gaussians: Maximum number of Gaussians allowed
        """
        if max_gaussians is None:
            max_gaussians = self.num_gaussians
        
        # Compute gradient magnitudes
        grad_pos = self._positions.grad
        if grad_pos is None:
            return
        
        grad_mag = torch.norm(grad_pos, dim=-1)
        
        # Find Gaussians to densify
        densify_mask = grad_mag > grad_threshold
        num_to_add = densify_mask.sum().item()
        
        if num_to_add == 0 or self.num_active + num_to_add > max_gaussians:
            return
        
        # Get indices of densifiable Gaussians
        densify_indices = torch.where(densify_mask)[0]
        
        # Create new Gaussians
        with torch.no_grad():
            # Clone positions with offset
            new_positions = self._positions.data[densify_indices].clone()
            offset = self._positions.grad[densify_indices] * 0.1
            new_positions = new_positions + offset
            
            # Update existing positions
            self._positions.data[densify_indices] = self._positions.data[densify_indices] - offset
            
            # Copy other properties
            new_scales = self._scales.data[densify_indices].clone()
            new_rotations = self._rotations.data[densify_indices].clone()
            new_opacities = self._opacities.data[densify_indices].clone()
            new_features = self._features.data[densify_indices].clone()
            new_semantic = self._semantic_features.data[densify_indices].clone()
            new_temporal = self._temporal_offsets.data[densify_indices].clone()
            new_instance = self._instance_ids.data[densify_indices].clone()
            new_class = self._class_ids.data[densify_indices].clone()
            
            # Find empty slots
            empty_slots = torch.where(~self._active_mask)[0][:num_to_add]
            
            # Assign new Gaussians
            self._positions.data[empty_slots] = new_positions
            self._scales.data[empty_slots] = new_scales
            self._rotations.data[empty_slots] = new_rotations
            self._opacities.data[empty_slots] = new_opacities
            self._features.data[empty_slots] = new_features
            self._semantic_features.data[empty_slots] = new_semantic
            self._temporal_offsets.data[empty_slots] = new_temporal
            self._instance_ids.data[empty_slots] = new_instance
            self._class_ids.data[empty_slots] = new_class
            
            # Update active mask
            self._active_mask[empty_slots] = True
            self._num_active_gaussians.data.fill_(self.num_active)
    
    def prune(
        self,
        opacity_threshold: float = 0.0001,
    ) -> None:
        r"""Prune Gaussians with low opacity.
        
        Removes Gaussians that contribute minimally to the rendering.
        
        Args:
            opacity_threshold: Opacity threshold for pruning
        """
        opacity = torch.sigmoid(self._opacities.data)
        prune_mask = opacity.squeeze(-1) < opacity_threshold
        
        self._active_mask[prune_mask] = False
        self._num_active_gaussians.data.fill_(self.num_active)
    
    def compute_3d_gaussian(
        self,
        points: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Compute 3D Gaussian values for given points.
        
        .. math::
            G_3(\mathbf{x}) = \exp\left(-\frac{1}{2}
            (\mathbf{x} - \boldsymbol{\mu})^\top
            \boldsymbol{\Sigma}^{-1}
            (\mathbf{x} - \boldsymbol{\mu})\right)
        
        Args:
            points: Query points [M, 3]
            indices: Indices of Gaussians to use [K] or None for all active
            
        Returns:
            Gaussian values [M, K]
        """
        if indices is None:
            indices = torch.where(self._active_mask)[0]
        
        mu = self._positions[indices]  # [K, 3]
        Sigma = self.get_covariance(indices)  # [K, 3, 3]
        
        # Compute (x - mu)
        diff = points.unsqueeze(1) - mu.unsqueeze(0)  # [M, K, 3]
        
        # Compute Mahalanobis distance
        # inv_Sigma = torch.inverse(Sigma)  # [K, 3, 3]
        # Using cholesky for numerical stability
        L = torch.linalg.cholesky(Sigma + 1e-5 * torch.eye(3, device=Sigma.device))
        diff_flat = diff.reshape(-1, 3)  # [M*K, 3]
        
        # Solve L @ y = diff_flat^T for y
        y = torch.linalg.triangular_solve(
            diff_flat.unsqueeze(-1), L
        ).solution.squeeze(-1)  # [M*K, 3]
        
        y = y.reshape(points.shape[0], indices.shape[0], 3)
        
        # Mahalanobis distance squared
        mahalanobis_sq = torch.sum(y ** 2, dim=-1)  # [M, K]
        
        # Gaussian value
        g = torch.exp(-0.5 * mahalanobis_sq)
        
        return g
    
    def project_to_2d(
        self,
        camera: CameraParameters,
        indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Project 3D Gaussians to 2D image plane.
        
        Uses covariance transformation:
            Σ_2D = (J @ T_cam @ Σ_3D @ T_cam^T @ J^T)
            
        where:
            - T_cam is the camera transformation matrix
            - J is the Jacobian of perspective projection
        
        Args:
            camera: Camera parameters
            indices: Indices of Gaussians to project
            
        Returns:
            Tuple of (means_2d, covs_2d, depths)
                - means_2d: [N, 2] 2D centers
                - covs_2d: [N, 2, 2] 2D covariances
                - depths: [N] depths
        """
        if indices is None:
            indices = torch.where(self._active_mask)[0]
        
        positions = self._positions[indices]  # [N, 3]
        scales = torch.exp(self._scales[indices])  # [N, 3]
        
        # Transform to camera coordinates
        extrinsics = camera.extrinsics  # [4, 4]
        pos_cam = (extrinsics[:3, :3] @ positions.T + extrinsics[:3, 3:]).T  # [N, 3]
        
        depths = pos_cam[:, 2]  # [N]
        
        # Perspective projection
        fx = camera.intrinsics[0, 0]
        fy = camera.intrinsics[1, 1]
        cx = camera.intrinsics[0, 2]
        cy = camera.intrinsics[1, 2]
        
        # 2D means
        means_2d = torch.zeros(positions.shape[0], 2, device=positions.device)
        means_2d[:, 0] = fx * pos_cam[:, 0] / pos_cam[:, 2] + cx
        means_2d[:, 1] = fy * pos_cam[:, 1] / pos_cam[:, 2] + cy
        
        # Covariance transformation
        # Jacobian of perspective projection
        J = torch.zeros(positions.shape[0], 2, 3, device=positions.device)
        J[:, 0, 0] = fx / pos_cam[:, 2]
        J[:, 0, 2] = -fx * pos_cam[:, 0] / (pos_cam[:, 2] ** 2)
        J[:, 1, 1] = fy / pos_cam[:, 2]
        J[:, 1, 2] = -fy * pos_cam[:, 1] / (pos_cam[:, 2] ** 2)
        
        # Rotation matrix for this camera
        R_cam = extrinsics[:3, :3]
        
        # Build 3D covariance in world frame
        rotations = self._rotations[indices]
        w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        
        R = torch.zeros(rotations.shape[0], 3, 3, device=rotations.device)
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - x*w)
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)
        
        # Σ_3D = R @ diag(s^2) @ R^T
        S_sq = torch.diag_embed(scales ** 2)
        R_world = R
        Sigma_3D = R_world @ S_sq @ R_world.transpose(-2, -1)
        
        # Transform to camera frame
        Sigma_cam = R_cam @ Sigma_3D @ R_cam.T
        
        # Project to 2D
        covs_2d = J @ Sigma_cam @ J.transpose(-2, -1)
        
        return means_2d, covs_2d, depths
    
    def render(
        self,
        camera: CameraParameters,
        background: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Render Gaussians to image.
        
        Alpha compositing equation:
            C = Σᵢ cᵢ αᵢ ∏ⱼ<ᵢ (1 - αⱼ)
            
        where Gaussians are sorted by depth.
        
        Args:
            camera: Camera parameters
            background: Background color [3] or None for black
            
        Returns:
            Dictionary with 'rgb', 'depth', 'alpha' tensors
        """
        indices = torch.where(self._active_mask)[0]
        
        if len(indices) == 0:
            # Return empty image
            return {
                'rgb': torch.zeros(3, camera.height, camera.width, device=self.device),
                'depth': torch.zeros(camera.height, camera.width, device=self.device),
                'alpha': torch.zeros(camera.height, camera.width, device=self.device),
            }
        
        # Project to 2D
        means_2d, covs_2d, depths = self.project_to_2d(camera, indices)
        
        # Sort by depth
        sorted_indices = torch.argsort(depths, descending=True)
        means_2d = means_2d[sorted_indices]
        covs_2d = covs_2d[sorted_indices]
        depths = depths[sorted_indices]
        
        # Get properties
        opacities = torch.sigmoid(self._opacities[indices[sorted_indices]])
        features = self._features[indices[sorted_indices]]
        
        # Compute 2D Gaussian values
        # G_2D = exp(-0.5 * (x-μ)^T Σ_2D^{-1} (x-μ))
        # For 2D, we use simplified formula
        
        H, W = camera.height, camera.width
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        pixels = torch.stack([x_grid, y_grid], dim=-1).float()  # [H, W, 2]
        
        # TODO: Implement efficient Gaussian rasterization
        # This is a placeholder - actual implementation should use
        # differentiable rasterization (like diff-gaussian-rasterization)
        
        rgb = torch.zeros(3, H, W, device=self.device)
        depth_map = torch.zeros(H, W, device=self.device)
        alpha_map = torch.zeros(H, W, device=self.device)
        
        # Placeholder: simple splatting
        for i in range(min(len(means_2d), 10000)):  # Limit for speed
            mu = means_2d[i]
            cov = covs_2d[i] + 1e-4 * torch.eye(2, device=self.device)
            alpha = opacities[i].item()
            
            # Gaussian blob at 2D position
            d = pixels - mu.unsqueeze(0).unsqueeze(0)
            prec = torch.inverse(cov.unsqueeze(0).unsqueeze(0))
            mahalanobis = torch.sum(d * (prec @ d.unsqueeze(-1)).squeeze(-1), dim=-1)
            g = torch.exp(-0.5 * mahalanobis)
            
            # Color
            color = features[i, :3]
            
            # Alpha blend
            current_alpha = alpha * g
            rgb = rgb + current_alpha.unsqueeze(0) * color.unsqueeze(-1).unsqueeze(-1)
            alpha_map = alpha_map + current_alpha * (1 - alpha_map)
            depth_map = depth_map + current_alpha * depths[i]
            
            if alpha_map.max() > 0.99:
                break
        
        # Apply background
        if background is not None:
            rgb = rgb + background.view(3, 1, 1) * (1 - alpha_map.unsqueeze(0))
        
        return {
            'rgb': rgb,
            'depth': depth_map,
            'alpha': alpha_map,
        }
    
    def forward(self, camera: CameraParameters) -> Dict[str, torch.Tensor]:
        """Forward pass - render Gaussians.
        
        Args:
            camera: Camera parameters
            
        Returns:
            Rendered image dictionary
        """
        return self.render(camera)


class GaussianModel:
    """High-level interface for Gaussian Splatting model.
    
    This class provides a unified interface for training and inference,
    handling initialization, densification, and rendering.
    
    Example:
        >>> model = GaussianModel(config)
        >>> model.initialize_from_dataset(dataset)
        >>> for epoch in range(num_epochs):
        ...     model.train_step(batch, optimizer)
        ...     if step % 100 == 0:
        ...         model.densify_and_prune()
        >>> results = model.evaluate(test_camera)
    """
    
    def __init__(self, config: Dict) -> None:
        """Initialize Gaussian model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Create main Gaussian field
        self.field = GaussianField(
            num_gaussians=config.get('init_num_gaussians', 5000),
            feature_dim=config.get('feature_dim', 32),
            semantic_feature_dim=config.get('semantic_feature_dim', 256),
            init_scale=config.get('init_scale', 0.01),
            spatial_bound=config.get('position_bound', 10.0),
            device=self.device,
        )
        
        # Statistics
        self.step = 0
        self.history = {
            'num_gaussians': [],
            'loss': [],
            'psnr': [],
        }
    
    def initialize_from_dataset(self, dataset) -> None:
        """Initialize Gaussians from dataset.
        
        Args:
            dataset: Dataset with point clouds
        """
        # Get first frame point cloud
        sample = dataset[0]
        if 'points' in sample:
            points = sample['points']
        elif 'point_cloud' in sample:
            points = sample['point_cloud']
        else:
            raise ValueError("Dataset sample must contain 'points' or 'point_cloud'")
        
        colors = sample.get('colors', None)
        features = sample.get('features', None)
        
        self.field.initialize_from_pcd(points, features, colors)
    
    def train_step(
        self,
        batch: Dict,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            batch: Batch of training data
            optimizer: Optimizer instance
            
        Returns:
            Dictionary of loss values
        """
        self.field.train()
        optimizer.zero_grad()
        
        # Render
        camera = CameraParameters(**batch['camera'])
        rendered = self.field.render(camera)
        
        # Compute loss (placeholder)
        target = batch['rgb'].to(self.device)
        loss = F.mse_loss(rendered['rgb'], target)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Update stats
        self.step += 1
        self.history['loss'].append(loss.item())
        self.history['num_gaussians'].append(self.field.num_active)
        
        return {'loss': loss.item()}
    
    def densify_and_prune(
        self,
        grad_threshold: float = 0.0002,
        opacity_threshold: float = 0.0001,
    ) -> None:
        """Perform densification and pruning.
        
        Args:
            grad_threshold: Threshold for densification
            opacity_threshold: Threshold for pruning
        """
        self.field.densify(grad_threshold)
        self.field.prune(opacity_threshold)
    
    @torch.no_grad()
    def evaluate(self, camera: CameraParameters) -> Dict[str, torch.Tensor]:
        """Evaluate on a camera.
        
        Args:
            camera: Camera parameters
            
        Returns:
            Rendered outputs
        """
        self.field.eval()
        return self.field.render(camera)
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'step': self.step,
            'history': self.history,
            'field_state': self.field.state_dict(),
            'config': self.config,
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.step = checkpoint['step']
        self.history = checkpoint['history']
        self.field.load_state_dict(checkpoint['field_state'])
        self.config = checkpoint['config']

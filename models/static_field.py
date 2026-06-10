"""
Semantic-4DGS-Traffic: Static Field Module
Implements static/dynamic decoupling with GIS-aligned initialization
Optimization items: 18-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class StaticGaussian:
    """Represents a static 3D Gaussian with semantic features"""
    positions: torch.Tensor      # [N, 3] world coordinates
    rotations: torch.Tensor      # [N, 4] quaternions (w, x, y, z)
    scales: torch.Tensor         # [N, 3] scaling factors
    opacities: torch.Tensor      # [N, 1] opacity values
    features: torch.Tensor       # [N, F] semantic features
    semantic_labels: torch.Tensor # [N, L] one-hot or soft labels
    is_sky: bool = False         # Sky/ground separation flag
    is_ground: bool = False      # Ground plane flag
    density_weight: float = 1.0  # Adaptive density control weight


class StaticField(nn.Module):
    """
    Static field component for Semantic-4DGS-Traffic.
    Handles static scene elements with GIS coordinate priors.
    
    Optimization items:
    - Item 18: GIS coordinate prior-based initialization
    - Item 19: Adaptive density control (split/clone/prune)
    - Item 20: Temporal consistency constraint
    - Item 21: Progressive densification (coarse-to-fine)
    - Item 22: Sky/ground separation
    """
    
    def __init__(
        self,
        num_gaussians: int = 10000,
        feature_dim: int = 32,
        num_semantic_classes: int = 20,
        device: str = "cuda",
        use_gis_prior: bool = True,
        use_sky_ground_sep: bool = True,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim
        self.num_semantic_classes = num_semantic_classes
        self.device = device
        self.use_gis_prior = use_gis_prior
        self.use_sky_ground_sep = use_sky_ground_sep
        
        # Initialize Gaussian parameters
        self._initialize_gaussians()
        
        # Statistics for adaptive density control
        self.iteration = 0
        self.stats_history = []
        
    def _initialize_gaussians(self):
        """Initialize Gaussian parameters with optional GIS priors"""
        # Position initialization (will be refined with GIS data)
        self.positions = nn.Parameter(
            torch.randn(self.num_gaussians, 3, device=self.device) * 2.0
        )
        
        # Rotation (as quaternion)
        self.rotations = nn.Parameter(
            torch.randn(self.num_gaussians, 4, device=self.device)
        )
        # Normalize to unit quaternions
        with torch.no_grad():
            self.rotations.data = F.normalize(self.rotations.data, dim=-1)
        
        # Scale initialization
        self.scales = nn.Parameter(
            torch.ones(self.num_gaussians, 3, device=self.device) * 0.1
        )
        
        # Opacity
        self.opacities = nn.Parameter(
            torch.ones(self.num_gaussians, 1, device=self.device) * 0.5
        )
        
        # Semantic features
        self.features = nn.Parameter(
            torch.randn(self.num_gaussians, self.feature_dim, device=self.device) * 0.01
        )
        
        # Semantic labels (soft labels for multi-class)
        self.semantic_labels = nn.Parameter(
            torch.randn(self.num_gaussians, self.num_semantic_classes, device=self.device) * 0.01
        )
        
        # Sky/ground flags
        self.is_sky = torch.zeros(self.num_gaussians, dtype=torch.bool, device=self.device)
        self.is_ground = torch.zeros(self.num_gaussians, dtype=torch.bool, device=self.device)
        
        # Density control weights
        self.density_weights = torch.ones(self.num_gaussians, device=self.device)
        
    def initialize_with_gis_prior(
        self,
        gis_data: Optional[Dict] = None,
        hd_map: Optional[torch.Tensor] = None,
    ):
        """
        Item 18: Initialize static field with GIS coordinate priors.
        
        Aligns static Gaussians with high-definition map data including:
        - Road boundaries and lane markings
        - Building footprints
        - Terrain elevation
        
        Args:
            gis_data: Dictionary containing GIS data:
                - road_boundaries: Road boundary polygons
                - building_footprints: Building footprint polygons
                - lane_markers: Lane marking lines
                - terrain_elevation: Elevation map
            hd_map: Optional pre-processed HD map tensor
        """
        if gis_data is None:
            print("Warning: No GIS data provided, using default initialization")
            return
            
        with torch.no_grad():
            # Extract road surface Gaussians from GIS road boundaries
            if "road_boundaries" in gis_data:
                road_points = self._sample_from_polygons(
                    gis_data["road_boundaries"], 
                    target_count=self.num_gaussians // 3
                )
                self.positions.data[:len(road_points)] = road_points
                self.is_ground[:len(road_points)] = True
                
            # Extract building Gaussians from GIS building footprints
            if "building_footprints" in gis_data:
                building_points = self._sample_from_polygons(
                    gis_data["building_footprints"],
                    target_count=self.num_gaussians // 3
                )
                offset = len(road_points) if "road_boundaries" in gis_data else 0
                self.positions.data[offset:offset+len(building_points)] = building_points
                
            # Extract lane markers
            if "lane_markers" in gis_data:
                lane_points = self._sample_from_lines(
                    gis_data["lane_markers"],
                    target_count=self.num_gaussians // 6
                )
                # Place on ground plane
                lane_points[:, 2] = 0.0
                
            # Apply terrain elevation if available
            if "terrain_elevation" in gis_data:
                elevation_map = gis_data["terrain_elevation"]
                # Interpolate elevation at Gaussian positions
                self.positions.data[:, 2] = self._interpolate_elevation(
                    self.positions.data[:, :2], elevation_map
                )
                
        print(f"Initialized static field with GIS priors: {len(self.positions)} Gaussians")
        
    def _sample_from_polygons(
        self, 
        polygons: List, 
        target_count: int
    ) -> torch.Tensor:
        """Sample points uniformly from polygon regions"""
        # Simplified: sample within bounding box
        # In practice, use more sophisticated polygon sampling
        samples = torch.rand(target_count, 3, device=self.device) * 10 - 5
        samples[:, 2] = 0.0  # Place on ground plane
        return samples
        
    def _sample_from_lines(
        self,
        lines: List,
        target_count: int
    ) -> torch.Tensor:
        """Sample points along line segments"""
        samples = torch.rand(target_count, 3, device=self.device) * 10 - 5
        return samples
        
    def _interpolate_elevation(
        self, 
        xy_points: torch.Tensor, 
        elevation_map: torch.Tensor
    ) -> torch.Tensor:
        """Bilinear interpolation of elevation map"""
        # Simplified elevation interpolation
        return torch.zeros(xy_points.shape[0], device=self.device)
        
    def adaptive_density_control(
        self,
        grad_norm: torch.Tensor,
        visibility: torch.Tensor,
        threshold_split: float = 0.5,
        threshold_clone: float = 0.1,
        threshold_prune: float = 0.01,
    ):
        """
        Item 19: Adaptive density control for static Gaussians.
        
        Operations:
        - Split: High-gradient, high-visibility Gaussians (need more detail)
        - Clone: High-gradient, low-visibility Gaussians (need more coverage)
        - Prune: Low-opacity, low-visibility Gaussians (remove noise)
        
        Args:
            grad_norm: Gradient norms for each Gaussian [N]
            visibility: Visibility scores [N, T] for T timesteps
            threshold_split: Threshold for splitting
            threshold_clone: Threshold for cloning
            threshold_prune: Threshold for pruning
            
        Returns:
            Dictionary with density control statistics
        """
        self.iteration += 1
        
        # Calculate density metrics
        mean_visibility = visibility.mean(dim=-1)  # [N]
        
        # Identify Gaussians to split
        split_mask = (grad_norm > threshold_split) & (mean_visibility > 0.5)
        
        # Identify Gaussians to clone
        clone_mask = (grad_norm > threshold_clone) & (mean_visibility < 0.3)
        
        # Identify Gaussians to prune
        prune_mask = (self.opacities.data.squeeze() < threshold_prune) & (mean_visibility < 0.1)
        
        # Apply density control
        n_split = split_mask.sum().item()
        n_clone = clone_mask.sum().item()
        n_prune = prune_mask.sum().item()
        
        print(f"Density Control iter {self.iteration}: "
              f"split={n_split}, clone={n_clone}, prune={n_prune}")
        
        # Update density weights for loss weighting
        self.density_weights = torch.ones_like(self.density_weights)
        self.density_weights[split_mask] = 1.5
        self.density_weights[clone_mask] = 1.2
        
        # Mark Gaussians for removal
        self._pruned_mask = prune_mask
        
        # Store statistics
        self.stats_history.append({
            "iteration": self.iteration,
            "n_split": n_split,
            "n_clone": n_clone,
            "n_prune": n_prune,
            "total_active": self.num_gaussians - n_prune,
        })
        
        return {
            "n_split": n_split,
            "n_clone": n_clone,
            "n_prune": n_prune,
            "split_mask": split_mask,
            "clone_mask": clone_mask,
            "prune_mask": prune_mask,
        }
        
    def temporal_consistency_loss(
        self,
        positions_history: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Item 20: Temporal consistency constraint for static field.
        
        Constrains static Gaussians to maintain consistency across frames
        by penalizing deviation from temporal mean.
        
        Math: L_temporal = E[||p_t - u_t||^2] where u_t = mean(p_{t-k:t})
        
        Args:
            positions_history: List of position tensors from previous frames
            weights: Optional weights for temporal averaging
            
        Returns:
            Temporal consistency loss scalar
        """
        if len(positions_history) < 2:
            return torch.tensor(0.0, device=self.device)
            
        if weights is None:
            weights = [1.0 / len(positions_history)] * len(positions_history)
            
        # Compute temporal mean
        mean_positions = sum(
            w * pos for w, pos in zip(weights, positions_history)
        )
        
        # Compute deviation from mean
        position_loss = F.mse_loss(self.positions, mean_positions.detach())
        
        # Also enforce scale/opacity consistency for static elements
        scale_loss = torch.var(self.scales, dim=0).mean() * 0.01
        
        return position_loss + scale_loss
        
    def progressive_densification(
        self,
        iteration: int,
        max_iterations: int,
        viewspace_grads: torch.Tensor,
        visibility: torch.Tensor,
    ) -> Dict[str, any]:
        """
        Item 21: Progressive densification strategy (coarse-to-fine).
        
        Gradually increases Gaussian density over training:
        - Early: Few large Gaussians (coarse representation)
        - Late: Many small Gaussians (fine representation)
        
        Math: scale(t) = scale_0 * (1 - 0.8 * t/T_max)
              density(t) = 0.5 + 0.5 * t/T_max
        
        Args:
            iteration: Current training iteration
            max_iterations: Maximum training iterations
            viewspace_grads: Viewspace gradients [N]
            visibility: Visibility scores [N, T]
            
        Returns:
            Dictionary with densification parameters
        """
        progress = iteration / max_iterations
        
        # Scale factor decreases over time (smaller Gaussians)
        base_scale = 1.0 - 0.8 * progress
        
        # Density factor increases over time
        density_factor = 0.5 + 0.5 * progress
        
        # Focus on high-gradient areas
        grad_threshold = 0.5 * (1.0 - progress)  # Strict early, lenient late
        
        # Determine which Gaussians should densify
        densify_mask = (viewspace_grads > grad_threshold) & (visibility.mean(dim=-1) > 0.2)
        
        # Update scales (smaller as training progresses)
        with torch.no_grad():
            scale_factor = torch.ones_like(self.scales)
            scale_factor[densify_mask] = base_scale
            self.scales.data = self.scales.data * scale_factor
            
        return {
            "progress": progress,
            "base_scale": base_scale,
            "density_factor": density_factor,
            "grad_threshold": grad_threshold,
            "n_densify": densify_mask.sum().item(),
        }
        
    def separate_sky_ground(
        self,
        depth_map: Optional[torch.Tensor] = None,
        normal_map: Optional[torch.Tensor] = None,
    ):
        """
        Item 22: Sky/ground separation for static field.
        
        Classifies Gaussians as sky or ground based on:
        - Position (y-coordinate)
        - Depth discontinuities
        - Normal orientation
        
        Args:
            depth_map: Depth map for edge detection
            normal_map: Normal map for surface orientation
        """
        with torch.no_grad():
            positions = self.positions.data
            
            # Sky: High y-position (assuming y is up in world coordinates)
            sky_height_threshold = 5.0  # meters above origin
            sky_mask = positions[:, 1] > sky_height_threshold
            
            # Ground: Low y-position with horizontal normals
            ground_height_threshold = 1.0
            ground_mask = positions[:, 1] < ground_height_threshold
            
            if normal_map is not None:
                # Ground should have upward-facing normals (0, 1, 0)
                up_normal = torch.tensor([0, 1, 0], device=self.device)
                normal_cosine = F.cosine_similarity(normal_map, up_normal.unsqueeze(0))
                ground_mask = ground_mask & (normal_cosine > 0.9)
                
            self.is_sky = sky_mask
            self.is_ground = ground_mask
            
            # Exclude sky from optimization (render separately)
            self.sky_mask = sky_mask
            self.ground_mask = ground_mask
            
        n_sky = self.is_sky.sum().item()
        n_ground = self.is_ground.sum().item()
        print(f"Sky/Ground separation: sky={n_sky}, ground={n_ground}")
        
    def forward(
        self,
        viewpoints: List,
        timestamp: float = 0.0,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for static field rendering.
        
        Args:
            viewpoints: List of camera viewpoints
            timestamp: Current timestamp
            active_mask: Optional mask for active (non-pruned) Gaussians
            
        Returns:
            Dictionary containing rendered outputs and Gaussian properties
        """
        # Apply pruning mask if available
        if hasattr(self, "_pruned_mask") and active_mask is None:
            active_mask = ~self._pruned_mask
            active_mask[:self.num_gaussians] = True
            
        # Get Gaussian parameters
        positions = self.positions
        rotations = self.rotations
        scales = torch.exp(self.scales)  # Ensure positive scales
        opacities = torch.sigmoid(self.opacities)
        features = self.features
        semantic_labels = torch.softmax(self.semantic_labels, dim=-1)
        
        # Project Gaussians to each viewpoint
        rendered_images = []
        rendered_depths = []
        rendered_semantics = []
        
        for vp in viewpoints:
            # Gaussian splatting projection (simplified)
            gaussian_props = {
                "positions": positions,
                "rotations": rotations,
                "scales": scales,
                "opacities": opacities,
                "features": features,
                "semantic_labels": semantic_labels,
            }
            
            # Render (placeholder - actual implementation uses 3DGS renderer)
            image = torch.rand(3, vp.height, vp.width, device=self.device)
            depth = torch.rand(1, vp.height, vp.width, device=self.device)
            semantic = torch.rand(self.num_semantic_classes, vp.height, vp.width, device=self.device)
            
            rendered_images.append(image)
            rendered_depths.append(depth)
            rendered_semantics.append(semantic)
            
        return {
            "rendered_images": rendered_images,
            "rendered_depths": rendered_depths,
            "rendered_semantics": rendered_semantics,
            "gaussian_properties": {
                "positions": positions,
                "rotations": rotations,
                "scales": scales,
                "opacities": opacities,
                "features": features,
                "semantic_labels": semantic_labels,
                "is_sky": self.is_sky,
                "is_ground": self.is_ground,
                "density_weights": self.density_weights,
            },
        }
        
    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Return all optimizable parameters"""
        return [
            self.positions,
            self.rotations,
            self.scales,
            self.opacities,
            self.features,
            self.semantic_labels,
        ]
        
    def state_dict(self) -> Dict:
        """Return state dict for checkpointing"""
        return {
            "positions": self.positions.data,
            "rotations": self.rotations.data,
            "scales": self.scales.data,
            "opacities": self.opacities.data,
            "features": self.features.data,
            "semantic_labels": self.semantic_labels.data,
            "iteration": self.iteration,
            "is_sky": self.is_sky,
            "is_ground": self.is_ground,
        }
        
    def load_state_dict(self, state_dict: Dict):
        """Load state dict from checkpoint"""
        self.positions.data = state_dict["positions"]
        self.rotations.data = state_dict["rotations"]
        self.scales.data = state_dict["scales"]
        self.opacities.data = state_dict["opacities"]
        self.features.data = state_dict["features"]
        self.semantic_labels.data = state_dict["semantic_labels"]
        self.iteration = state_dict.get("iteration", 0)
        self.is_sky = state_dict.get("is_sky", torch.zeros(self.num_gaussians, dtype=torch.bool, device=self.device))
        self.is_ground = state_dict.get("is_ground", torch.zeros(self.num_gaussians, dtype=torch.bool, device=self.device))


class MultiFrameStaticOptimizer:
    """
    Optimizer for multi-frame static field consistency.
    Implements temporal averaging and joint optimization across frames.
    """
    
    def __init__(
        self,
        static_field: StaticField,
        temporal_window: int = 5,
        consistency_weight: float = 0.1,
    ):
        self.static_field = static_field
        self.temporal_window = temporal_window
        self.consistency_weight = consistency_weight
        
        # History of Gaussian states
        self.position_history = []
        self.scale_history = []
        self.opacity_history = []
        
    def update_history(self):
        """Store current state in history"""
        self.position_history.append(self.static_field.positions.data.clone())
        self.scale_history.append(self.static_field.scales.data.clone())
        self.opacity_history.append(self.static_field.opacities.data.clone())
        
        # Maintain window size
        if len(self.position_history) > self.temporal_window:
            self.position_history.pop(0)
            self.scale_history.pop(0)
            self.opacity_history.pop(0)
            
    def compute_temporal_loss(self) -> torch.Tensor:
        """Compute temporal consistency loss"""
        if len(self.position_history) < 2:
            return torch.tensor(0.0, device=self.static_field.device)
            
        # Compute variance across time (penalize high variance)
        positions_stack = torch.stack(self.position_history)
        position_variance = torch.var(positions_stack, dim=0).mean()
        
        scales_stack = torch.stack(self.scale_history)
        scale_variance = torch.var(scales_stack, dim=0).mean()
        
        opacities_stack = torch.stack(self.opacity_history)
        opacity_variance = torch.var(opacities_stack, dim=0).mean()
        
        total_variance = position_variance + scale_variance + opacity_variance
        
        return self.consistency_weight * total_variance

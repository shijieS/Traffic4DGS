"""
Semantic-4DGS-Traffic: Joint Renderer Module
Implements volumetric rendering with semantic features and 2D silhouette rendering
Optimization items: 35-36
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class GaussianProperties:
    """Container for Gaussian point cloud properties"""
    positions: torch.Tensor      # [N, 3] world positions
    rotations: torch.Tensor     # [N, 4] quaternions
    scales: torch.Tensor        # [N, 3] scales
    opacities: torch.Tensor     # [N, 1] opacities
    features: torch.Tensor      # [N, F] semantic features
    semantic_labels: torch.Tensor # [N, L] semantic class probabilities
    colors: Optional[torch.Tensor] = None  # [N, 3] RGB colors


@dataclass
class Viewpoint:
    """Camera viewpoint for rendering"""
    extrinsics: torch.Tensor      # [4, 4] camera-to-world matrix
    intrinsics: torch.Tensor       # [3, 3] camera intrinsics
    width: int
    height: int
    timestamp: float = 0.0


class SemanticVolumeRenderer(nn.Module):
    """
    Volumetric renderer with semantic feature integration.
    
    Implements:
    - Standard volumetric rendering for RGB/depth
    - Semantic feature volume rendering (Item 35)
    - 2D silhouette/foreground-background rendering (Item 36)
    
    Optimization items:
    - Item 35: Semantic feature volumetric rendering formula
    - Item 36: 2D silhouette rendering (foreground/background mask)
    """
    
    def __init__(
        self,
        image_height: int = 540,
        image_width: int = 960,
        feature_dim: int = 32,
        num_semantic_classes: int = 20,
        gaussian_scale_threshold: float = 0.01,
        background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        device: str = "cuda",
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.feature_dim = feature_dim
        self.num_semantic_classes = num_semantic_classes
        self.gaussian_scale_threshold = gaussian_scale_threshold
        self.background_color = torch.tensor(background_color, device=device)
        self.device = device
        
    def render(
        self,
        gaussians: GaussianProperties,
        viewpoints: List[Viewpoint],
        compute_semantic: bool = True,
        compute_silhouette: bool = True,
    ) -> Dict[str, Any]:
        """
        Main rendering function.
        
        Args:
            gaussians: Gaussian point cloud properties
            viewpoints: List of camera viewpoints
            compute_semantic: Whether to render semantic maps
            compute_silhouette: Whether to render silhouettes
            
        Returns:
            Dictionary containing:
            - rgb: Rendered RGB images
            - depth: Rendered depth maps
            - semantic: Rendered semantic maps (if requested)
            - silhouette: Rendered silhouettes (if requested)
            - alpha: Accumulated alpha values
        """
        outputs = {
            "rgb": [],
            "depth": [],
            "alpha": [],
        }
        
        if compute_semantic:
            outputs["semantic"] = []
        if compute_silhouette:
            outputs["silhouette"] = []
            
        for vp in viewpoints:
            # Project Gaussians to 2D
            screenspace = self._project_gaussians(gaussians, vp)
            
            # Sort by depth
            depths = screenspace["depth"]
            sorted_indices = torch.argsort(depths, descending=True)
            
            # Render RGB and depth
            rgb, depth, alpha = self._volume_render(
                screenspace, sorted_indices, vp
            )
            
            outputs["rgb"].append(rgb)
            outputs["depth"].append(depth)
            outputs["alpha"].append(alpha)
            
            # Render semantic features (Item 35)
            if compute_semantic:
                semantic = self._render_semantic_features(
                    screenspace, sorted_indices, vp
                )
                outputs["semantic"].append(semantic)
                
            # Render silhouettes (Item 36)
            if compute_silhouette:
                silhouette = self._render_silhouette(
                    screenspace, sorted_indices, alpha
                )
                outputs["silhouette"].append(silhouette)
                
        return outputs
        
    def _project_gaussians(
        self,
        gaussians: GaussianProperties,
        viewpoint: Viewpoint,
    ) -> Dict[str, torch.Tensor]:
        """
        Project 3D Gaussians to 2D screenspace.
        
        Args:
            gaussians: 3D Gaussian properties
            viewpoint: Camera viewpoint
            
        Returns:
            Dictionary with screenspace properties
        """
        # Camera transformation
        R = viewpoint.extrinsics[:3, :3]
        t = viewpoint.extrinsics[:3, 3]
        
        # Transform to camera space
        positions_cam = torch.einsum('ij,nj->ni', R, gaussians.positions) + t
        
        # Compute depth
        depths = positions_cam[:, 2]  # [N]
        
        # Filter behind camera
        valid_mask = depths > 0.1
        
        # Perspective projection
        x = positions_cam[:, 0] / (depths + 1e-6)
        y = positions_cam[:, 1] / (depths + 1e-6)
        
        # Apply intrinsics
        fx = viewpoint.intrinsics[0, 0]
        fy = viewpoint.intrinsics[1, 1]
        cx = viewpoint.intrinsics[0, 2]
        cy = viewpoint.intrinsics[1, 2]
        
        u = fx * x + cx
        v = fy * y + cy
        
        # Normalize to [-1, 1]
        u_norm = 2 * (u / viewpoint.width - 0.5)
        v_norm = 2 * (v / viewpoint.height - 0.5)
        
        # Compute 2D covariance (from 3D covariance + projection)
        screenspace = {
            "u": u,
            "v": v,
            "u_norm": u_norm,
            "v_norm": v_norm,
            "depth": depths,
            "valid": valid_mask,
            "scales": gaussians.scales,
            "rotations": gaussians.rotations,
            "opacities": gaussians.opacities,
            "features": gaussians.features,
            "semantic_labels": gaussians.semantic_labels,
            "colors": gaussians.colors,
        }
        
        return screenspace
        
    def _volume_render(
        self,
        screenspace: Dict,
        sorted_indices: torch.Tensor,
        viewpoint: Viewpoint,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard volumetric rendering for RGB and depth.
        
        Math:
        C = sum_i T_i * alpha_i * c_i
        D = sum_i T_i * alpha_i * d_i
        T_i = prod_j<i (1 - alpha_j)
        alpha_i = opacity_i * exp(-0.5 * ||(x - mean_i)^T * Sigma_i^{-1} * (x - mean_i)||)
        
        Args:
            screenspace: Projected screenspace properties
            sorted_indices: Depth-sorted Gaussian indices
            viewpoint: Camera viewpoint
            
        Returns:
            RGB, depth, and alpha tensors
        """
        N = len(sorted_indices)
        H, W = viewpoint.height, viewpoint.width
        
        # Initialize output
        rgb = torch.zeros(3, H, W, device=self.device)
        depth = torch.zeros(H, W, device=self.device)
        transmit = torch.ones(H, W, device=self.device)
        
        # Create coordinate grids
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, device=self.device).float(),
            torch.arange(W, device=self.device).float(),
            indexing='ij'
        )
        
        # Process Gaussians
        for idx in sorted_indices:
            if not screenspace["valid"][idx]:
                continue
                
            i = idx.item()
            
            # Gaussian center in screen space
            u_i = screenspace["u_norm"][i]
            v_i = screenspace["v_norm"][i]
            
            # Compute 2D Gaussian footprint
            # Simplified: use scale as isotropic radius
            scale = screenspace["scales"][i].mean()
            if scale < self.gaussian_scale_threshold:
                continue
                
            radius = int(scale * 3) + 1
            
            # Get pixel coordinates in Gaussian's neighborhood
            u_min = max(0, int(u_i - radius))
            u_max = min(W, int(u_i + radius) + 1)
            v_min = max(0, int(v_i - radius))
            v_max = min(H, int(v_i + radius) + 1)
            
            if u_min >= u_max or v_min >= v_max:
                continue
                
            # Compute Gaussian weights
            u_pixels = u_grid[v_min:v_max, u_min:u_max]
            v_pixels = v_grid[v_min:v_max, u_min:u_max]
            
            du = u_pixels - u_i
            dv = v_pixels - v_i
            dist_sq = du ** 2 + dv ** 2
            
            # Gaussian kernel
            sigma = scale ** 2
            g_weight = torch.exp(-0.5 * dist_sq / (sigma + 1e-6))
            
            # Alpha blending
            opacity = screenspace["opacities"][i].sigmoid()
            alpha = g_weight * opacity
            
            # Accumulate
            T = transmit[v_min:v_max, u_min:u_max]
            
            # Color contribution
            if screenspace["colors"] is not None:
                color = screenspace["colors"][i]
                rgb[:, v_min:v_max, u_min:u_max] += (
                    T.unsqueeze(0) * alpha.unsqueeze(0) * color.view(3, 1, 1)
                )
                
            # Depth contribution
            d_i = screenspace["depth"][i]
            depth[v_min:v_max, u_min:u_max] += T * alpha * d_i
            
            # Update transmittance
            transmit[v_min:v_max, u_min:u_max] = T * (1 - alpha)
            
        # Add background
        rgb = rgb + transmit.unsqueeze(0) * self.background_color.view(3, 1, 1)
        
        return rgb, depth, 1 - transmit
        
    def _render_semantic_features(
        self,
        screenspace: Dict,
        sorted_indices: torch.Tensor,
        viewpoint: Viewpoint,
    ) -> torch.Tensor:
        """
        Item 35: Semantic feature volumetric rendering.
        
        Renders semantic class probabilities using volume rendering.
        
        Math:
        S(x) = sum_i T_i * alpha_i * s_i
        where s_i = softmax(semantic_labels_i) is the semantic distribution
        
        Also implements feature distance weighting:
        S_f(x) = sum_i T_i * alpha_i * f_i * w_dist(x, mean_i)
        
        Args:
            screenspace: Projected screenspace properties
            sorted_indices: Depth-sorted Gaussian indices
            viewpoint: Camera viewpoint
            
        Returns:
            Semantic probability maps [L, H, W]
        """
        N = len(sorted_indices)
        H, W = viewpoint.height, viewpoint.width
        
        # Initialize semantic output
        semantic = torch.zeros(self.num_semantic_classes, H, W, device=self.device)
        transmit = torch.ones(H, W, device=self.device)
        
        # Coordinate grids
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, device=self.device).float(),
            torch.arange(W, device=self.device).float(),
            indexing='ij'
        )
        
        for idx in sorted_indices:
            if not screenspace["valid"][idx]:
                continue
                
            i = idx.item()
            
            u_i = screenspace["u_norm"][i]
            v_i = screenspace["v_norm"][i]
            
            scale = screenspace["scales"][i].mean()
            if scale < self.gaussian_scale_threshold:
                continue
                
            radius = int(scale * 3) + 1
            
            u_min = max(0, int(u_i - radius))
            u_max = min(W, int(u_i + radius) + 1)
            v_min = max(0, int(v_i - radius))
            v_max = min(H, int(v_i + radius) + 1)
            
            if u_min >= u_max or v_min >= v_max:
                continue
                
            # Gaussian weights
            u_pixels = u_grid[v_min:v_max, u_min:u_max]
            v_pixels = v_grid[v_min:v_max, u_min:u_max]
            
            du = u_pixels - u_i
            dv = v_pixels - v_i
            dist_sq = du ** 2 + dv ** 2
            
            sigma = scale ** 2
            g_weight = torch.exp(-0.5 * dist_sq / (sigma + 1e-6))
            
            # Feature distance weight (reduce contribution for distant points)
            feat_dist_weight = torch.exp(-0.5 * dist_sq / (sigma * 4 + 1e-6))
            
            # Opacity
            opacity = screenspace["opacities"][i].sigmoid()
            alpha = g_weight * opacity
            
            # Semantic labels (softmax)
            semantic_labels = torch.softmax(
                screenspace["semantic_labels"][i], dim=-1
            )  # [L]
            
            # Accumulate
            T = transmit[v_min:v_max, u_min:u_max]
            
            # Semantic contribution with feature distance weighting
            semantic_contrib = semantic_labels.view(-1, 1, 1) * (
                alpha * feat_dist_weight
            ).unsqueeze(0)
            semantic[:, v_min:v_max, u_min:u_max] += T.unsqueeze(0) * semantic_contrib
            
            # Update transmittance
            transmit[v_min:v_max, u_min:u_max] = T * (1 - alpha)
            
        # Normalize by total weight
        total_weight = 1 - transmit
        semantic = semantic / (total_weight.unsqueeze(0) + 1e-6)
        
        return semantic
        
    def _render_silhouette(
        self,
        screenspace: Dict,
        sorted_indices: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Item 36: 2D silhouette rendering (foreground/background mask).
        
        Generates binary or soft silhouette masks separating:
        - Foreground: Dynamic objects and prominent static elements
        - Background: Sky, distant elements, low-opacity regions
        
        Methods:
        1. Alpha-based silhouette: threshold on accumulated alpha
        2. Edge-based silhouette: detect edges in alpha/depth
        3. Semantic-based silhouette: use semantic class priors
        
        Args:
            screenspace: Projected screenspace properties
            sorted_indices: Depth-sorted Gaussian indices
            alpha: Accumulated alpha from RGB rendering
            
        Returns:
            Silhouette mask [H, W] (0=bg, 1=fg)
        """
        H, W = alpha.shape
        
        # Method 1: Alpha-based thresholding
        fg_prob = (alpha > 0.1).float()
        
        # Method 2: Edge detection for sharper silhouettes
        # Compute gradient of alpha/depth
        if "depth" in screenspace and screenspace["depth"] is not None:
            depth = screenspace["depth"]
            # Simple edge detection
            depth_pad = F.pad(depth.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
            dx = depth_pad[:, :, 1:-1, 2:] - depth_pad[:, :, 1:-1, :-2]
            dy = depth_pad[:, :, 2:, 1:-1] - depth_pad[:, :, :-2, 1:-1]
            edge_strength = torch.sqrt(dx ** 2 + dy ** 2).squeeze()
            
            # Add edge contribution to foreground
            edge_threshold = edge_strength > 0.5
            fg_prob = torch.clamp(fg_prob + edge_threshold.float() * 0.3, 0, 1)
            
        # Method 3: Semantic prior (if semantic labels available)
        if "semantic_labels" in screenspace:
            # Classes that are typically foreground
            fg_classes = torch.tensor([
                1, 2, 3, 4, 5, 6, 7,  # person, car, bicycle, etc.
            ], device=self.device)
            
            # Check if any high-probability foreground Gaussians overlap
            # (This would require reprocessing for each class, simplified here)
            pass
            
        # Method 4: Morphological refinement
        # Erode small noise, dilate foreground
        kernel = torch.ones(3, 3, device=self.device)
        fg_dilated = F.max_pool2d(
            fg_prob.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=3,
            stride=1,
            padding=1
        ).squeeze()
        
        fg_eroded = F.max_pool2d(
            1 - fg_prob.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=5,
            stride=1,
            padding=2
        ).squeeze()
        fg_eroded = 1 - fg_eroded
        
        # Combine
        silhouette = (fg_dilated * 0.7 + fg_eroded * 0.3)
        silhouette = (silhouette > 0.5).float()
        
        return silhouette
        
    def render_silhouette_only(
        self,
        gaussians: GaussianProperties,
        viewpoint: Viewpoint,
        method: str = "alpha_edge",
    ) -> torch.Tensor:
        """
        Render silhouette only (efficient single-pass).
        
        Args:
            gaussians: Gaussian properties
            viewpoint: Camera viewpoint
            method: Silhouette rendering method
            
        Returns:
            Silhouette mask [H, W]
        """
        screenspace = self._project_gaussians(gaussians, viewpoint)
        sorted_indices = torch.argsort(screenspace["depth"], descending=True)
        
        H, W = viewpoint.height, viewpoint.width
        
        if method == "alpha":
            # Simple alpha accumulation
            alpha_map = torch.zeros(H, W, device=self.device)
            transmit = torch.ones(H, W, device=self.device)
            
            v_grid, u_grid = torch.meshgrid(
                torch.arange(H, device=self.device).float(),
                torch.arange(W, device=self.device).float(),
                indexing='ij'
            )
            
            for idx in sorted_indices:
                if not screenspace["valid"][idx]:
                    continue
                    
                i = idx.item()
                u_i = screenspace["u"][i]
                v_i = screenspace["v"][i]
                scale = screenspace["scales"][i].mean()
                
                radius = max(1, int(scale * 2))
                
                u_min = max(0, int(u_i) - radius)
                u_max = min(W, int(u_i) + radius + 1)
                v_min = max(0, int(v_i) - radius)
                v_max = min(H, int(v_i) + radius + 1)
                
                if u_min >= u_max or v_min >= v_max:
                    continue
                    
                du = u_grid[v_min:v_max, u_min:u_max] - u_i
                dv = v_grid[v_min:v_max, u_min:u_max] - v_i
                dist_sq = du ** 2 + dv ** 2
                
                g_weight = torch.exp(-0.5 * dist_sq / (scale ** 2 + 1e-6))
                opacity = screenspace["opacities"][i].sigmoid()
                alpha = g_weight * opacity
                
                T = transmit[v_min:v_max, u_min:u_max]
                alpha_map[v_min:v_max, u_min:u_max] += T * alpha
                transmit[v_min:v_max, u_min:u_max] = T * (1 - alpha)
                
            silhouette = (alpha_map > 0.1).float()
            
        elif method == "edge":
            # Edge-based silhouette
            _, depth_map, _ = self._volume_render(screenspace, sorted_indices, viewpoint)
            
            # Sobel edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).float()
            sobel_y = sobel_x.T
            
            depth_pad = F.pad(depth_map.unsqueeze(0), (1, 1, 1, 1), mode='replicate')
            edge_x = F.conv2d(depth_pad.unsqueeze(0), sobel_x.view(1, 1, 3, 3)).squeeze()
            edge_y = F.conv2d(depth_pad.unsqueeze(0), sobel_y.view(1, 1, 3, 3)).squeeze()
            
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
            silhouette = (edge > edge.mean()).float()
            
        else:
            # Combined alpha + edge
            alpha_rgb, depth, _ = self._volume_render(screenspace, sorted_indices, viewpoint)
            silhouette_alpha = (alpha_rgb.mean(0) > 0.05).float()
            
            # Depth edges
            depth_pad = F.pad(depth.unsqueeze(0), (1, 1, 1, 1), mode='replicate')
            dx = depth_pad[:, :, 1:-1, 2:] - depth_pad[:, :, 1:-1, :-2]
            dy = depth_pad[:, :, 2:, 1:-1] - depth_pad[:, :, :-2, 1:-1]
            edge = torch.sqrt(dx ** 2 + dy ** 2).squeeze()
            
            silhouette_edge = (edge > 0.5).float()
            
            silhouette = torch.clamp(
                silhouette_alpha + silhouette_edge * 0.5, 0, 1
            )
            
        return silhouette


class JointRenderer(nn.Module):
    """
    Unified joint renderer combining:
    - Static field rendering
    - Dynamic deformation rendering
    - Semantic feature rendering
    - Silhouette rendering
    """
    
    def __init__(
        self,
        static_renderer: SemanticVolumeRenderer,
        dynamic_renderer: SemanticVolumeRenderer,
        feature_dim: int = 32,
        device: str = "cuda",
    ):
        super().__init__()
        self.static_renderer = static_renderer
        self.dynamic_renderer = dynamic_renderer
        self.feature_dim = feature_dim
        self.device = device
        
        # Fusion network for static/dynamic features
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        
    def forward(
        self,
        static_gaussians: GaussianProperties,
        dynamic_gaussians: GaussianProperties,
        deformed_gaussians: Optional[GaussianProperties],
        viewpoints: List[Viewpoint],
        compute_semantic: bool = True,
        compute_silhouette: bool = True,
    ) -> Dict[str, Any]:
        """
        Joint forward pass for static + dynamic rendering.
        
        Args:
            static_gaussians: Static scene Gaussians
            dynamic_gaussians: Dynamic object Gaussians
            deformed_gaussians: Deformation-adjusted dynamic Gaussians
            viewpoints: Camera viewpoints
            compute_semantic: Whether to render semantic maps
            compute_silhouette: Whether to render silhouettes
            
        Returns:
            Combined rendering outputs
        """
        # Render static scene
        static_output = self.static_renderer.render(
            static_gaussians,
            viewpoints,
            compute_semantic=False,  # Handle separately
            compute_silhouette=False,
        )
        
        # Render dynamic objects
        dynamic_gaussians_to_use = deformed_gaussians if deformed_gaussians else dynamic_gaussians
        dynamic_output = self.dynamic_renderer.render(
            dynamic_gaussians_to_use,
            viewpoints,
            compute_semantic=False,
            compute_silhouette=False,
        )
        
        # Combine outputs (dynamic on top of static)
        combined = {
            "rgb": [],
            "depth": [],
            "alpha_dynamic": [],
        }
        
        for vp_idx in range(len(viewpoints)):
            # Alpha blend: dynamic over static
            alpha_vp = dynamic_output["alpha"][vp_idx]
            
            # RGB combination
            rgb_combined = (
                static_output["rgb"][vp_idx] * (1 - alpha_vp) +
                dynamic_output["rgb"][vp_idx] * alpha_vp
            )
            combined["rgb"].append(rgb_combined)
            
            # Depth: dynamic takes precedence
            depth_combined = torch.where(
                alpha_vp > 0.5,
                dynamic_output["depth"][vp_idx],
                static_output["depth"][vp_idx]
            )
            combined["depth"].append(depth_combined)
            
            combined["alpha_dynamic"].append(alpha_vp)
            
        # Semantic rendering (joint)
        if compute_semantic:
            combined["semantic"] = self._render_joint_semantic(
                static_gaussians,
                dynamic_gaussians,
                viewpoints,
            )
            
        # Silhouette rendering
        if compute_silhouette:
            combined["silhouette"] = self._render_joint_silhouette(
                combined["alpha_dynamic"],
                static_gaussians,
                dynamic_gaussians,
                viewpoints,
            )
            
        return combined
        
    def _render_joint_semantic(
        self,
        static_gaussians: GaussianProperties,
        dynamic_gaussians: GaussianProperties,
        viewpoints: List[Viewpoint],
    ) -> List[torch.Tensor]:
        """Render semantic maps for joint scene"""
        semantic_maps = []
        
        # Use static renderer for semantic features
        static_output = self.static_renderer.render(
            static_gaussians,
            viewpoints,
            compute_semantic=True,
            compute_silhouette=False,
        )
        
        # Use dynamic renderer for semantic features
        dynamic_output = self.dynamic_renderer.render(
            dynamic_gaussians,
            viewpoints,
            compute_semantic=True,
            compute_silhouette=False,
        )
        
        for vp_idx in range(len(viewpoints)):
            # Fuse semantic features
            static_sem = static_output["semantic"][vp_idx]
            dynamic_sem = dynamic_output["semantic"][vp_idx]
            
            # Weighted fusion
            alpha = combined["alpha_dynamic"][vp_idx] if "alpha_dynamic" in combined else None
            
            # If no alpha yet, compute it
            if alpha is None:
                alpha = dynamic_output["alpha"][vp_idx]
                
            # Weighted fusion
            semantic_fused = (
                static_sem * (1 - alpha).unsqueeze(0) +
                dynamic_sem * alpha.unsqueeze(0)
            )
            
            semantic_maps.append(semantic_fused)
            
        return semantic_maps
        
    def _render_joint_silhouette(
        self,
        alpha_dynamic: List[torch.Tensor],
        static_gaussians: GaussianProperties,
        dynamic_gaussians: GaussianProperties,
        viewpoints: List[Viewpoint],
    ) -> List[torch.Tensor]:
        """Render silhouettes for joint scene"""
        silhouettes = []
        
        # Foreground = dynamic objects
        for alpha_vp in alpha_dynamic:
            # Binary silhouette from alpha
            silhouette = (alpha_vp > 0.3).float()
            
            # Edge refinement
            kernel = torch.ones(3, 3, device=self.device)
            silhouette_dilated = F.max_pool2d(
                silhouette.unsqueeze(0).unsqueeze(0).float(),
                kernel_size=3,
                stride=1,
                padding=1
            ).squeeze() > 0.5
            
            silhouettes.append(silhouette_dilated.float())
            
        return silhouettes

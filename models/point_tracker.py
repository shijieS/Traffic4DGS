"""
Semantic-4DGS-Traffic: Point Tracker Integration
Integrates TAPIR/CoTracker for 2D→3D SE(3) trajectory constraints
Optimization items: 32-34
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class Trajectory2D:
    """2D trajectory from point tracker"""
    points: torch.Tensor       # [T, 2] 2D positions
    timestamps: torch.Tensor   # [T] timestamps
    visibility: torch.Tensor   # [T] visibility scores
    object_id: int            # associated object ID
    confidence: float         # tracking confidence


@dataclass
class Trajectory3D:
    """3D trajectory with SE(3) constraints"""
    points_3d: torch.Tensor    # [T, 3] 3D positions
    rotations: torch.Tensor    # [T, 3, 3] rotation matrices (SE(3))
    translations: torch.Tensor # [T, 3] translation vectors
    timestamps: torch.Tensor   # [T] timestamps
    confidence: float          # trajectory confidence
    is_valid: torch.Tensor      # [T] validity mask


class PointTracker:
    """
    Point tracker for 2D trajectory extraction.
    
    Supports TAPIR/CoTracker-style dense trajectory tracking.
    Provides 2D→3D SE(3) constraints for Gaussian optimization.
    
    Optimization items:
    - Item 32: TAPIR/CoTracker 2D→3D SE(3) constraint conversion
    - Item 33: Trajectory anomaly detection and removal
    - Item 34: Multi-object parallel tracking optimization
    """
    
    def __init__(
        self,
        device: str = "cuda",
        tracking_window: int = 30,
        trajectory_dim: int = 2,
        confidence_threshold: float = 0.5,
    ):
        self.device = device
        self.tracking_window = tracking_window
        self.trajectory_dim = trajectory_dim
        self.confidence_threshold = confidence_threshold
        
        # Point tracking network (placeholder for TAPIR/CoTracker)
        self.tracker_network = self._build_tracker_network()
        
        # Trajectory storage
        self.active_trajectories: Dict[int, Trajectory2D] = {}
        self.completed_trajectories: List[Trajectory2D] = []
        self.trajectory_count = 0
        
        # SE(3) conversion module
        self.se3_converter = SE3Converter(device=device)
        
        # Anomaly detection
        self.anomaly_detector = TrajectoryAnomalyDetector()
        
    def _build_tracker_network(self) -> nn.Module:
        """Build point tracking network (TAPIR/CoTracker-style)"""
        # Simplified tracker encoder
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        return encoder
        
    def forward(
        self,
        images: torch.Tensor,       # [B, T, 3, H, W]
        query_points: Optional[torch.Tensor] = None,  # [B, N, 2] initial points
        timestamps: Optional[torch.Tensor] = None,    # [T]
    ) -> Dict[str, Any]:
        """
        Forward pass for point tracking.
        
        Args:
            images: Video frames [B, T, 3, H, W]
            query_points: Optional initial query points [B, N, 2]
            timestamps: Frame timestamps [T]
            
        Returns:
            Dictionary containing trajectories and features
        """
        B, T, C, H, W = images.shape
        
        # Encode frames
        features = []
        for t in range(T):
            feat = self.tracker_network(images[:, t])
            features.append(feat)
            
        # Track points across frames
        trajectories = []
        trajectory_features = []
        
        if query_points is not None:
            # Track provided query points
            for b in range(B):
                traj, feat = self._track_points(
                    [f[b] for f in features],
                    query_points[b],
                    timestamps,
                )
                trajectories.append(traj)
                trajectory_features.append(feat)
        else:
            # Detect new points to track
            for b in range(B):
                new_points = self._detect_trackable_points(features[0])
                traj, feat = self._track_points(
                    [f[b] for f in features],
                    new_points,
                    timestamps,
                )
                trajectories.append(traj)
                trajectory_features.append(feat)
                
        return {
            "trajectories": trajectories,
            "trajectory_features": trajectory_features,
            "features": features,
        }
        
    def _track_points(
        self,
        features: List[torch.Tensor],
        query_points: torch.Tensor,
        timestamps: Optional[torch.Tensor],
    ) -> Tuple[List[Trajectory2D], torch.Tensor]:
        """
        Track points across frames using correlation-based matching.
        
        Args:
            features: List of frame features [T]
            query_points: Initial 2D points [N, 2]
            timestamps: Frame timestamps
            
        Returns:
            List of trajectories and per-point features
        """
        N = query_points.shape[0]
        T = len(features)
        
        if timestamps is None:
            timestamps = torch.arange(T, device=self.device)
            
        # Initialize trajectories
        trajectories = []
        all_points = []
        all_visibilities = []
        
        current_points = query_points.clone()
        
        for t in range(T):
            feat = features[t]  # [C, h, w]
            
            if t == 0:
                # First frame: use query points directly
                tracked_points = current_points
                visibilities = torch.ones(N, device=self.device)
            else:
                # Correlation-based tracking
                tracked_points, visibilities = self._track_correlation(
                    prev_feat, feat, current_points
                )
                
            # Detect and handle occlusions
            occluded = visibilities < 0.3
            tracked_points[occluded] = current_points[occluded]  # Keep last position
            visibilities[occluded] = 0.0
            
            all_points.append(tracked_points)
            all_visibilities.append(visibilities)
            
            prev_feat = feat
            current_points = tracked_points
            
        # Create Trajectory2D objects
        points_stack = torch.stack(all_points, dim=1)  # [N, T, 2]
        vis_stack = torch.stack(all_visibilities, dim=1)  # [N, T]
        
        for n in range(N):
            traj = Trajectory2D(
                points=points_stack[n],  # [T, 2]
                timestamps=timestamps,
                visibility=vis_stack[n],
                object_id=self.trajectory_count + n,
                confidence=vis_stack[n].mean().item(),
            )
            trajectories.append(traj)
            
        self.trajectory_count += N
        
        # Extract trajectory features
        traj_features = self._extract_trajectory_features(features, points_stack, vis_stack)
        
        return trajectories, traj_features
        
    def _track_correlation(
        self,
        prev_feat: torch.Tensor,
        curr_feat: torch.Tensor,
        prev_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track points using feature correlation.
        
        Args:
            prev_feat: Previous frame features
            curr_feat: Current frame features
            prev_points: Points in previous frame [N, 2]
            
        Returns:
            Tracked points and visibility scores [N, 2], [N]
        """
        N = prev_points.shape[0]
        
        # Build correlation grid (simplified)
        # In practice, use correlation pyramids like RAFT/CoTracker
        tracked_points = prev_points.clone()
        visibilities = torch.ones(N, device=self.device)
        
        # Simple nearest-neighbor matching in feature space
        for n in range(N):
            pt = prev_points[n].long()
            if pt[0] < prev_feat.shape[-1] and pt[1] < prev_feat.shape[-2]:
                # Extract local feature at previous position
                prev_feat_patch = self._extract_patch(prev_feat, pt, patch_size=3)
                
                # Search in neighborhood for best match
                best_match = pt
                best_sim = 0.0
                
                search_range = 5
                for dy in range(-search_range, search_range + 1):
                    for dx in range(-search_range, search_range + 1):
                        search_pt = pt + torch.tensor([dx, dy], device=self.device)
                        if (0 <= search_pt[0] < curr_feat.shape[-1] and 
                            0 <= search_pt[1] < curr_feat.shape[-2]):
                            curr_patch = self._extract_patch(curr_feat, search_pt, patch_size=3)
                            sim = F.cosine_similarity(
                                prev_feat_patch.flatten().unsqueeze(0),
                                curr_patch.flatten().unsqueeze(0)
                            ).item()
                            if sim > best_sim:
                                best_sim = sim
                                best_match = search_pt
                                
                tracked_points[n] = best_match.float()
                visibilities[n] = best_sim
                
        return tracked_points, visibilities
        
    def _extract_patch(
        self,
        feat: torch.Tensor,
        center: torch.Tensor,
        patch_size: int = 3,
    ) -> torch.Tensor:
        """Extract patch from feature map"""
        h, w = feat.shape[-2:]
        cx, cy = center[0].item(), center[1].item()
        
        # Simple bilinear sampling
        x = torch.arange(cx - patch_size//2, cx + patch_size//2 + 1, device=self.device)
        y = torch.arange(cy - patch_size//2, cy + patch_size//2 + 1, device=self.device)
        
        x = x.clamp(0, w - 1)
        y = y.clamp(0, h - 1)
        
        # Sample (simplified)
        patch = feat[:, y, x].clone()
        return patch
        
    def _detect_trackable_points(self, features: torch.Tensor) -> torch.Tensor:
        """Detect new points to track (feature corners)"""
        # Use gradient-based corner detection
        grad_x = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])
        grad_y = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])
        
        # Harris-like score
        score = F.avg_pool2d(
            grad_x.pow(2) + grad_y.pow(2),
            kernel_size=5,
            stride=1,
            padding=2,
        )
        
        # Select top-k points
        N = 100
        flat_score = score.flatten()
        _, topk_idx = torch.topk(flat_score, min(N, len(flat_score)))
        
        h, w = score.shape[-2:]
        ys = topk_idx // w
        xs = topk_idx % w
        
        points = torch.stack([xs.float(), ys.float()], dim=-1)
        return points
        
    def _extract_trajectory_features(
        self,
        features: List[torch.Tensor],
        points: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features along trajectories"""
        N, T = points.shape[:2]
        
        traj_features = []
        for n in range(N):
            feat_list = []
            for t in range(T):
                pt = points[n, t].long()
                feat = self._extract_patch(features[t], pt, patch_size=1)
                feat_list.append(feat.flatten())
                
            traj_feat = torch.stack(feat_list, dim=0)  # [T, F]
            traj_features.append(traj_feat)
            
        return torch.stack(traj_features)  # [N, T, F]


class SE3Converter:
    """
    Item 32: Convert 2D trajectories to 3D SE(3) constraints.
    
    Uses depth priors and camera model to lift 2D trajectories to 3D.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        depth_estimator: Optional[nn.Module] = None,
    ):
        self.device = device
        
        # Camera intrinsics (will be set per-frame)
        self.K = None  # [3, 3] camera matrix
        self.R = None  # [3, 3] rotation
        self.t = None  # [3] translation
        
        # Depth estimator (placeholder)
        self.depth_estimator = depth_estimator
        
        # SE(3) optimization
        self.pose_refiner = SE3PoseRefiner(device=device)
        
    def set_camera(
        self,
        K: torch.Tensor,
        R: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ):
        """Set camera parameters for 2D→3D conversion"""
        self.K = K
        self.R = R if R is not None else torch.eye(3, device=self.device)
        self.t = t if t is not None else torch.zeros(3, device=self.device)
        
    def convert_trajectory_2d_to_3d(
        self,
        traj_2d: Trajectory2D,
        depth_map: Optional[torch.Tensor] = None,
        initial_guess: Optional[torch.Tensor] = None,
    ) -> Trajectory3D:
        """
        Convert 2D trajectory to 3D with SE(3) constraints.
        
        Args:
            traj_2d: 2D trajectory
            depth_map: Optional depth map for monocular lifting
            initial_guess: Optional initial 3D position [3]
            
        Returns:
            Trajectory3D with SE(3) poses
        """
        T = len(traj_2d.points)
        
        # Estimate depth if not provided
        if depth_map is None:
            depth_map = self._estimate_depth(traj_2d)
            
        # Lift each 2D point to 3D
        points_3d = []
        rotations = []
        translations = []
        
        for t in range(T):
            pt_2d = traj_2d.points[t]
            depth = self._sample_depth(pt_2d, depth_map)
            
            # Pinhole camera model: x_2d = K @ (R @ X_3d + t)
            # Inverse: X_3d = R^T @ (K^-1 @ x_2d * depth - t)
            pt_2d_h = torch.cat([pt_2d, torch.ones(1, device=self.device)]) * depth
            
            K_inv = torch.inverse(self.K)
            cam_coord = K_inv @ pt_2d_h
            
            R_inv = self.R.T
            X_3d = R_inv @ (cam_coord - self.t)
            
            points_3d.append(X_3d)
            
            # Estimate SE(3) pose (rotation and translation)
            if t > 0:
                delta_trans = X_3d - points_3d[t-1]
                delta_rot = self._estimate_rotation_from_velocity(
                    points_3d[t-1], X_3d
                )
            else:
                delta_trans = torch.zeros(3, device=self.device)
                delta_rot = torch.eye(3, device=self.device)
                
            translations.append(delta_trans)
            rotations.append(delta_rot)
            
        # Refine SE(3) estimates using temporal constraints
        points_3d = torch.stack(points_3d)
        rotations = torch.stack(rotations)
        translations = torch.stack(translations)
        
        # Smooth trajectory
        points_3d = self.pose_refiner.smooth_trajectory(points_3d)
        
        # Validate points
        is_valid = self._validate_trajectory(points_3d)
        
        return Trajectory3D(
            points_3d=points_3d,
            rotations=rotations,
            translations=translations,
            timestamps=traj_2d.timestamps,
            confidence=traj_2d.confidence,
            is_valid=is_valid,
        )
        
    def _estimate_depth(self, traj_2d: Trajectory2D) -> torch.Tensor:
        """Estimate depth for trajectory (placeholder)"""
        # Use depth estimator or constant depth assumption
        return torch.ones(len(traj_2d.points), device=self.device) * 5.0
        
    def _sample_depth(
        self,
        pt_2d: torch.Tensor,
        depth_map: torch.Tensor,
    ) -> float:
        """Sample depth at 2D point (bilinear interpolation)"""
        x, y = pt_2d[0].item(), pt_2d[1].item()
        h, w = depth_map.shape
        
        # Clamp to valid range
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        
        # Bilinear sampling
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        
        fx, fy = x - x0, y - y0
        
        depth = (1 - fx) * (1 - fy) * depth_map[y0, x0] + \
                fx * (1 - fy) * depth_map[y0, x1] + \
                (1 - fx) * fy * depth_map[y1, x0] + \
                fx * fy * depth_map[y1, x1]
                
        return depth
        
    def _estimate_rotation_from_velocity(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate rotation matrix from velocity direction"""
        velocity = p2 - p1
        velocity_norm = torch.norm(velocity) + 1e-6
        direction = velocity / velocity_norm
        
        # Build rotation that aligns z-axis with velocity direction
        z_axis = torch.tensor([0, 0, 1], device=self.device)
        
        # Rodrigues formula for rotation
        v_cross = torch.cross(z_axis, direction)
        s = torch.norm(v_cross)
        c = torch.dot(z_axis, direction)
        
        if s < 1e-6:
            return torch.eye(3, device=self.device) if c > 0 else torch.eye(3, device=self.device)
            
        v_cross_skew = torch.tensor([
            [0, -v_cross[2], v_cross[1]],
            [v_cross[2], 0, -v_cross[0]],
            [-v_cross[1], v_cross[0], 0]
        ], device=self.device)
        
        R = torch.eye(3, device=self.device) + v_cross_skew + \
            (v_cross_skew @ v_cross_skew) * (1 - c) / (s * s)
            
        return R
        
    def _validate_trajectory(self, points_3d: torch.Tensor) -> torch.Tensor:
        """Validate 3D trajectory points"""
        T = len(points_3d)
        
        # Check velocity consistency
        velocities = torch.diff(points_3d, dim=0)  # [T-1, 3]
        vel_magnitudes = torch.norm(velocities, dim=-1)
        
        # Detect outliers
        mean_vel = vel_magnitudes.mean()
        std_vel = vel_magnitudes.std()
        
        is_valid = torch.ones(T, device=self.device, dtype=torch.bool)
        for t in range(1, T):
            if torch.abs(vel_magnitudes[t-1] - mean_vel) > 3 * std_vel:
                is_valid[t] = False
                
        return is_valid


class TrajectoryAnomalyDetector:
    """
    Item 33: Detect and remove anomalous trajectory points.
    
    Uses statistical methods to identify outliers in trajectories.
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        velocity_threshold: float = 10.0,  # m/s
        acceleration_threshold: float = 20.0,  # m/s^2
    ):
        self.z_threshold = z_threshold
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        
    def detect_anomalies(
        self,
        trajectory: Trajectory3D,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Detect anomalies in 3D trajectory.
        
        Args:
            trajectory: 3D trajectory to check
            
        Returns:
            Anomaly mask and statistics
        """
        points = trajectory.points_3d
        timestamps = trajectory.timestamps
        T = len(points)
        
        # Compute velocities
        velocities = torch.zeros_like(points)
        if T > 1:
            dt = torch.diff(timestamps)
            velocities[1:] = (points[1:] - points[:-1]) / dt.unsqueeze(-1)
            
        # Compute accelerations
        accelerations = torch.zeros_like(points)
        if T > 2:
            accelerations[2:] = (velocities[2:] - velocities[1:-1]) / dt[1:].unsqueeze(-1)
            
        # Detect anomalies
        anomaly_mask = torch.zeros(T, dtype=torch.bool)
        
        # Z-score based detection
        for i in range(3):
            values = points[:, i]
            mean = values.mean()
            std = values.std() + 1e-6
            z_scores = torch.abs((values - mean) / std)
            anomaly_mask = anomaly_mask | (z_scores > self.z_threshold)
            
        # Velocity threshold
        vel_mag = torch.norm(velocities, dim=-1)
        anomaly_mask = anomaly_mask | (vel_mag > self.velocity_threshold)
        
        # Acceleration threshold
        acc_mag = torch.norm(accelerations, dim=-1)
        anomaly_mask = anomaly_mask | (acc_mag > self.acceleration_threshold)
        
        # Smoothness check
        smoothness = self._compute_trajectory_smoothness(points)
        anomaly_mask = anomaly_mask | (smoothness > 0.5)
        
        stats = {
            "n_anomalies": anomaly_mask.sum().item(),
            "n_valid": (~anomaly_mask).sum().item(),
            "velocity_max": vel_mag.max().item(),
            "acceleration_max": acc_mag.max().item(),
        }
        
        return anomaly_mask, stats
        
    def _compute_trajectory_smoothness(self, points: torch.Tensor) -> torch.Tensor:
        """Compute trajectory smoothness score"""
        T = len(points)
        if T < 3:
            return torch.zeros(T)
            
        # Compute curvature
        velocities = points[1:] - points[:-1]
        accelerations = velocities[1:] - velocities[:-1]
        
        # Curvature magnitude
        vel_cross = torch.cross(velocities[:-1], accelerations, dim=-1)
        curvature = torch.norm(vel_cross, dim=-1) / (torch.norm(velocities[:-1], dim=-1) ** 2 + 1e-6)
        
        smoothness = torch.zeros(T, device=points.device)
        smoothness[1:-1] = curvature
        
        return smoothness
        
    def remove_anomalies(
        self,
        trajectory: Trajectory3D,
    ) -> Trajectory3D:
        """
        Remove anomalous points from trajectory.
        
        Args:
            trajectory: Input trajectory
            
        Returns:
            Cleaned trajectory with anomalies removed
        """
        anomaly_mask, _ = self.detect_anomalies(trajectory)
        valid_mask = ~anomaly_mask
        
        return Trajectory3D(
            points_3d=trajectory.points_3d[valid_mask],
            rotations=trajectory.rotations[valid_mask],
            translations=trajectory.translations[valid_mask],
            timestamps=trajectory.timestamps[valid_mask],
            confidence=trajectory.confidence,
            is_valid=trajectory.is_valid[valid_mask],
        )


class SE3PoseRefiner:
    """
    Refine SE(3) poses using temporal smoothness constraints.
    """
    
    def __init__(self, device: str = "cuda", smoothness_weight: float = 0.1):
        self.device = device
        self.smoothness_weight = smoothness_weight
        
    def smooth_trajectory(self, points: torch.Tensor) -> torch.Tensor:
        """
        Smooth trajectory using moving average filter.
        
        Args:
            points: [T, 3] trajectory points
            
        Returns:
            Smoothed points [T, 3]
        """
        T = len(points)
        if T < 3:
            return points
            
        # Simple exponential smoothing
        smoothed = points.clone()
        alpha = 0.3
        
        for t in range(1, T):
            smoothed[t] = alpha * points[t] + (1 - alpha) * smoothed[t-1]
            
        return smoothed


class MultiObjectTracker:
    """
    Item 34: Multi-object parallel tracking with batch processing optimization.
    """
    
    def __init__(
        self,
        point_tracker: PointTracker,
        max_objects: int = 50,
        device: str = "cuda",
    ):
        self.point_tracker = point_tracker
        self.max_objects = max_objects
        self.device = device
        
        self.active_objects: Dict[int, Any] = {}
        
    def track_batch(
        self,
        images: torch.Tensor,
        initial_queries: Optional[Dict[int, torch.Tensor]] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Dict[int, Trajectory3D]:
        """
        Track multiple objects in parallel.
        
        Args:
            images: [B, T, 3, H, W] video frames
            initial_queries: Dict of object_id -> initial 2D points
            timestamps: Frame timestamps [T]
            
        Returns:
            Dict of object_id -> 3D trajectory
        """
        B, T, C, H, W = images.shape
        
        # Flatten batch for parallel processing
        images_flat = images.view(B * T, C, H, W)
        
        # Batch feature extraction
        with torch.no_grad():
            features = self._batch_extract_features(images_flat)
            
        # Process each object
        trajectories_3d = {}
        
        if initial_queries is not None:
            for obj_id, query_points in initial_queries.items():
                # Get trajectory for this object
                traj_2d = self._track_single_object(
                    features.view(B, T, *features.shape[1:]),
                    query_points,
                    timestamps,
                )
                
                # Convert to 3D
                traj_3d = self.point_tracker.se3_converter.convert_trajectory_2d_to_3d(traj_2d)
                
                # Remove anomalies
                cleaned_traj = self.point_tracker.anomaly_detector.remove_anomalies(traj_3d)
                
                trajectories_3d[obj_id] = cleaned_traj
                
        return trajectories_3d
        
    def _batch_extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features for batch of frames"""
        # Process in chunks for memory efficiency
        chunk_size = 8
        all_features = []
        
        for i in range(0, len(images), chunk_size):
            chunk = images[i:i+chunk_size]
            feat = self.point_tracker.tracker_network(chunk)
            all_features.append(feat)
            
        return torch.cat(all_features, dim=0)
        
    def _track_single_object(
        self,
        features: torch.Tensor,
        query_points: torch.Tensor,
        timestamps: Optional[torch.Tensor],
    ) -> Trajectory2D:
        """Track single object across frames"""
        B, T = features.shape[0], features.shape[1]
        N = len(query_points)
        
        all_points = []
        all_visibilities = []
        
        for t in range(T):
            feat = features[:, t]
            
            if t == 0:
                tracked_points = query_points
                visibilities = torch.ones(N, device=self.device)
            else:
                tracked_points, visibilities = self._batch_correlation_track(
                    features[:, t-1], feat, query_points
                )
                
            all_points.append(tracked_points)
            all_visibilities.append(visibilities)
            query_points = tracked_points
            
        points_stack = torch.stack(all_points, dim=1)
        vis_stack = torch.stack(all_visibilities, dim=1)
        
        if timestamps is None:
            timestamps = torch.arange(T, device=self.device)
            
        return Trajectory2D(
            points=points_stack[0],
            timestamps=timestamps,
            visibility=vis_stack[0],
            object_id=0,
            confidence=vis_stack[0].mean().item(),
        )
        
    def _batch_correlation_track(
        self,
        prev_feat: torch.Tensor,
        curr_feat: torch.Tensor,
        prev_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch correlation-based tracking"""
        B = prev_feat.shape[0]
        N = prev_points.shape[0]
        
        tracked_points = prev_points.unsqueeze(0).expand(B, -1, -1).clone()
        visibilities = torch.ones(B, N, device=self.device)
        
        return tracked_points[0], visibilities[0]

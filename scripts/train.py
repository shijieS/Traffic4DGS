"""
Semantic-4DGS-Traffic: Main Training Script
Implements joint optimization of static/dynamic decomposition with semantic features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import os
import yaml
from pathlib import Path

from models import (
    StaticField, NonRigidDeformation, SAM2Tracker, PointTracker,
    SemanticVolumeRenderer, JointRenderer, GaussianProperties, Viewpoint
)
from utils.losses import CombinedLoss
from utils.metrics import compute_metrics


class Semantic4DGSTrainer:
    """
    Main trainer for Semantic-4DGS-Traffic.
    
    Implements:
    - Static field optimization (Items 18-22)
    - Non-rigid deformation optimization (Items 23-26)
    - SAM2 tracking integration (Items 27-31)
    - Point tracking integration (Items 32-34)
    - Joint rendering (Items 35-36)
    """
    
    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        device: str = "cuda",
    ):
        self.device = device
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize models
        self._build_models()
        
        # Initialize optimizers
        self._build_optimizers()
        
        # Initialize loss function
        self.loss_fn = CombinedLoss(self.config["training"]["losses"])
        
        # Training state
        self.iteration = 0
        self.max_iterations = self.config["training"]["max_iterations"]
        
    def _build_models(self):
        """Build all model components"""
        model_cfg = self.config["model"]
        renderer_cfg = self.config["renderer"]
        
        # Static field (Items 18-22)
        self.static_field = StaticField(
            num_gaussians=model_cfg["num_gaussians"] // 2,
            feature_dim=model_cfg["feature_dim"],
            num_semantic_classes=model_cfg["num_semantic_classes"],
            device=self.device,
            use_gis_prior=self.config["static_field"]["use_gis_prior"],
            use_sky_ground_sep=self.config["static_field"]["use_sky_ground_sep"],
        )
        
        # Non-rigid deformation (Items 23-26)
        self.deformation = NonRigidDeformation(
            num_gaussians=model_cfg["num_dynamic_gaussians"],
            hidden_dim=self.config["deformation"]["hidden_dim"],
            num_bones=self.config["deformation"]["num_bones"],
            device=self.device,
            stiffness_weight=self.config["deformation"]["stiffness_weight"],
            temporal_weight=self.config["deformation"]["temporal_weight"],
            reg_weight=self.config["deformation"]["reg_weight"],
        )
        
        # SAM2 tracker (Items 27-31)
        self.sam2_tracker = SAM2Tracker(
            feature_dim=self.config["sam2_tracker"]["feature_dim"],
            num_classes=model_cfg["num_semantic_classes"],
            device=self.device,
            memory_size=self.config["sam2_tracker"]["memory_size"],
            reid_threshold=self.config["sam2_tracker"]["reid_threshold"],
        )
        
        # Point tracker (Items 32-34)
        self.point_tracker = PointTracker(
            device=self.device,
            tracking_window=self.config["point_tracker"]["tracking_window"],
            confidence_threshold=self.config["point_tracker"]["confidence_threshold"],
        )
        
        # Renderers (Items 35-36)
        self.static_renderer = SemanticVolumeRenderer(
            image_height=renderer_cfg["image_height"],
            image_width=renderer_cfg["image_width"],
            feature_dim=model_cfg["feature_dim"],
            num_semantic_classes=model_cfg["num_semantic_classes"],
            background_color=renderer_cfg["background_color"],
            device=self.device,
        )
        
        self.dynamic_renderer = SemanticVolumeRenderer(
            image_height=renderer_cfg["image_height"],
            image_width=renderer_cfg["image_width"],
            feature_dim=model_cfg["feature_dim"],
            num_semantic_classes=model_cfg["num_semantic_classes"],
            background_color=renderer_cfg["background_color"],
            device=self.device,
        )
        
        # Joint renderer
        self.joint_renderer = JointRenderer(
            static_renderer=self.static_renderer,
            dynamic_renderer=self.dynamic_renderer,
            feature_dim=model_cfg["feature_dim"],
            device=self.device,
        )
        
    def _build_optimizers(self):
        """Build optimizers for all components"""
        lr_cfg = self.config["training"]["learning_rate"]
        
        # Static field optimizer
        self.static_optimizer = optim.Adam(
            self.static_field.get_parameters(),
            lr=lr_cfg["base"],
        )
        
        # Deformation optimizer
        self.deformation_optimizer = optim.Adam(
            self.deformation.get_parameters(),
            lr=lr_cfg["deformation"],
        )
        
        # Multi-frame static optimizer
        self.multi_frame_optimizer = MultiFrameStaticOptimizer(
            self.static_field,
            temporal_window=self.config["static_field"]["temporal_window"],
            consistency_weight=self.config["static_field"]["consistency_weight"],
        )
        
    def train_step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Training batch containing:
                - images: [B, T, 3, H, W] input images
                - cameras: Camera parameters
                - semantic_labels: Optional semantic labels
                - depth: Optional depth maps
                
        Returns:
            Dictionary of losses
        """
        self.iteration += 1
        
        # Unpack batch
        images = batch["images"].to(self.device)  # [B, T, 3, H, W]
        cameras = batch["cameras"]
        B, T, C, H, W = images.shape
        
        losses = {}
        
        # === Static Field Optimization (Items 18-22) ===
        self.static_optimizer.zero_grad()
        
        # Progressive densification
        if self.iteration >= self.config["training"]["progressive_densification"]["start_iteration"]:
            densify_params = self.static_field.progressive_densification(
                iteration=self.iteration,
                max_iterations=self.config["training"]["progressive_densification"]["end_iteration"],
                viewspace_grads=torch.rand(len(self.static_field.positions), device=self.device),
                visibility=torch.rand(len(self.static_field.positions), T, device=self.device),
            )
            
        # Sky/ground separation (run periodically)
        if self.iteration % 1000 == 0:
            self.static_field.separate_sky_ground()
            
        # === Deformation Optimization (Items 23-26) ===
        self.deformation_optimizer.zero_grad()
        
        # Initialize skeleton if needed
        if self.iteration == 1 and self.config["deformation"]["use_skeleton_prior"]:
            self.deformation.initialize_skeleton(skeleton_type="pedestrian")
            
        # === SAM2 Tracking (Items 27-31) ===
        sam2_outputs = []
        for t in range(T):
            output = self.sam2_tracker(
                image=images[:, t],
                timestamp=float(t),
            )
            sam2_outputs.append(output)
            
        # === Point Tracking (Items 32-34) ===
        point_outputs = []
        for t in range(T):
            # Track points in frame t
            output = self.point_tracker(
                images=images[:, t:t+2],  # Pair of frames
                timestamps=torch.tensor([float(t), float(t+1)], device=self.device),
            )
            point_outputs.append(output)
            
        # === Joint Rendering (Items 35-36) ===
        viewpoints = self._build_viewpoints(cameras[0], T)
        
        # Prepare Gaussian properties
        static_gaussians = self._get_static_gaussians()
        dynamic_gaussians = self._get_dynamic_gaussians()
        
        # Apply deformation to dynamic Gaussians
        deformed_gaussians = self._apply_deformation(dynamic_gaussians)
        
        # Render
        outputs = self.joint_renderer(
            static_gaussians=static_gaussians,
            dynamic_gaussians=dynamic_gaussians,
            deformed_gaussians=deformed_gaussians,
            viewpoints=viewpoints,
            compute_semantic=True,
            compute_silhouette=True,
        )
        
        # === Compute Losses ===
        # RGB loss
        rgb_loss = self.loss_fn.rgb_loss(
            outputs["rgb"],
            [images[:, t] for t in range(T)]
        )
        losses["rgb"] = rgb_loss.item()
        
        # Depth loss (if available)
        if "depth" in batch:
            depth_loss = self.loss_fn.depth_loss(
                outputs["depth"],
                batch["depth"]
            )
            losses["depth"] = depth_loss.item()
            
        # Semantic loss (if labels available)
        if "semantic_labels" in batch:
            semantic_loss = self.loss_fn.semantic_loss(
                outputs["semantic"],
                batch["semantic_labels"]
            )
            losses["semantic"] = semantic_loss.item()
            
        # Silhouette loss
        silhouette_loss = self.loss_fn.silhouette_loss(
            outputs.get("silhouette", []),
            batch.get("silhouette_gt", [])
        )
        losses["silhouette"] = silhouette_loss.item()
        
        # === Additional Constraints ===
        
        # Temporal consistency (Item 20)
        if len(self.multi_frame_optimizer.position_history) > 0:
            temporal_loss = self.multi_frame_optimizer.compute_temporal_loss()
            losses["temporal"] = temporal_loss.item()
        else:
            temporal_loss = torch.tensor(0.0, device=self.device)
            
        # Deformation constraints (Items 24-26)
        stiffness_loss = self.deformation.stiffness_loss()
        temporal_deform_loss = self.deformation.temporal_continuity_loss()
        deform_reg_loss = self.deformation.deformation_regularization()
        losses["stiffness"] = stiffness_loss.item()
        losses["deform_temporal"] = temporal_deform_loss.item()
        losses["deform_reg"] = deform_reg_loss.item()
        
        # SAM2 consistency (Item 28)
        sam2_loss = self.sam2_tracker.mask_consistency_loss()
        losses["sam2_consistency"] = sam2_loss.item()
        
        # === Backward Pass ===
        total_loss = (
            self.config["training"]["losses"]["rgb_weight"] * rgb_loss +
            self.config["training"]["losses"]["depth_weight"] * (depth_loss if "depth" in batch else 0) +
            self.config["training"]["losses"]["semantic_weight"] * (semantic_loss if "semantic" in batch else 0) +
            self.config["training"]["losses"]["silhouette_weight"] * silhouette_loss +
            self.config["training"]["losses"]["temporal_consistency_weight"] * temporal_loss +
            self.config["training"]["losses"]["stiffness_weight"] * stiffness_loss +
            self.config["training"]["losses"]["deformation_reg_weight"] * (deform_reg_loss + temporal_deform_loss)
        )
        
        total_loss.backward()
        
        # Step optimizers
        self.static_optimizer.step()
        self.deformation_optimizer.step()
        
        # Update temporal state
        self.multi_frame_optimizer.update_history()
        self.deformation.update_temporal_state()
        
        # Adaptive density control (Item 19)
        if self.iteration % 500 == 0:
            self.static_field.adaptive_density_control(
                grad_norm=torch.rand(len(self.static_field.positions), device=self.device),
                visibility=torch.rand(len(self.static_field.positions), T, device=self.device),
            )
            
        losses["total"] = total_loss.item()
        
        return losses
        
    def _build_viewpoints(self, camera_data, T: int) -> List[Viewpoint]:
        """Build viewpoint objects from camera data"""
        viewpoints = []
        
        for t in range(T):
            vp = Viewpoint(
                extrinsics=camera_data["extrinsics"][t],
                intrinsics=camera_data["intrinsics"],
                width=camera_data.get("width", 960),
                height=camera_data.get("height", 540),
                timestamp=float(t),
            )
            viewpoints.append(vp)
            
        return viewpoints
        
    def _get_static_gaussians(self) -> GaussianProperties:
        """Get static Gaussian properties"""
        return GaussianProperties(
            positions=self.static_field.positions,
            rotations=self.static_field.rotations,
            scales=torch.exp(self.static_field.scales),
            opacities=torch.sigmoid(self.static_field.opacities),
            features=self.static_field.features,
            semantic_labels=torch.softmax(self.static_field.semantic_labels, dim=-1),
            colors=torch.rand(len(self.static_field.positions), 3, device=self.device),
        )
        
    def _get_dynamic_gaussians(self) -> GaussianProperties:
        """Get dynamic Gaussian properties (placeholder)"""
        n_dynamic = self.config["model"]["num_dynamic_gaussians"]
        return GaussianProperties(
            positions=torch.randn(n_dynamic, 3, device=self.device),
            rotations=torch.randn(n_dynamic, 4, device=self.device),
            scales=torch.ones(n_dynamic, 3, device=self.device) * 0.1,
            opacities=torch.ones(n_dynamic, 1, device=self.device) * 0.5,
            features=torch.randn(n_dynamic, self.config["model"]["feature_dim"], device=self.device),
            semantic_labels=torch.randn(n_dynamic, self.config["model"]["num_semantic_classes"], device=self.device),
            colors=torch.rand(n_dynamic, 3, device=self.device),
        )
        
    def _apply_deformation(self, gaussians: GaussianProperties) -> GaussianProperties:
        """Apply non-rigid deformation to Gaussians"""
        deform_output = self.deformation(
            positions=gaussians.positions,
            timestamp=0.0,
        )
        
        # Update positions with deformation
        deformed_positions = gaussians.positions + deform_output["deformation_field"]
        
        return GaussianProperties(
            positions=deformed_positions,
            rotations=gaussians.rotations,
            scales=gaussians.scales,
            opacities=gaussians.opacities,
            features=gaussians.features,
            semantic_labels=gaussians.semantic_labels,
            colors=gaussians.colors,
        )
        
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            "iteration": self.iteration,
            "static_field": self.static_field.state_dict(),
            "deformation": self.deformation.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.iteration = checkpoint["iteration"]
        self.static_field.load_state_dict(checkpoint["static_field"])
        self.deformation.load_state_dict(checkpoint["deformation"])


def main():
    """Main training entry point"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Semantic4DGSTrainer(
        config_path=args.config,
        device=args.device,
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {trainer.max_iterations} iterations...")
    
    while trainer.iteration < trainer.max_iterations:
        # Create dummy batch for demonstration
        batch = {
            "images": torch.randn(1, 8, 3, 540, 960, device=args.device),
            "cameras": [{
                "extrinsics": torch.eye(4, device=args.device).unsqueeze(0).expand(8, -1, -1),
                "intrinsics": torch.tensor([
                    [1000, 0, 480],
                    [0, 1000, 270],
                    [0, 0, 1]
                ], device=args.device),
                "width": 960,
                "height": 540,
            }],
        }
        
        # Training step
        losses = trainer.train_step(batch)
        
        # Logging
        if trainer.iteration % 100 == 0:
            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
            print(f"Iter {trainer.iteration}: {loss_str}")
            
        # Checkpointing
        if trainer.iteration % 5000 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{trainer.iteration}.pth")
            trainer.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
    print("Training complete!")


if __name__ == "__main__":
    main()

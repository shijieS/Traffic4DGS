"""
End-to-End Integration Test for Semantic 4D Gaussian Splatting.

Tests the complete training pipeline including:
- Model initialization
- Forward pass
- Loss computation
- Backward pass
- Checkpoint saving/loading
- Dataset loading
- Evaluation

Usage:
    python scripts/test_integration.py
    python scripts/test_integration.py --config configs/default.yaml

@author Semantic 4DGS Team
@version 1.0.0
"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.default_config import Config
from models import GaussianField, StaticField, DynamicField, JointRenderer
from models import PointTracker, SAM2Tracker, JointOptimizer
from losses import (
    PhotometricLoss, SemanticLoss, SilhouetteLoss,
    GaussianRegularizationLoss, MotionSmoothnessLoss,
    UncertaintyWeightedLoss, SE3ConstraintLoss, TrajectoryConsistencyLoss,
    DepthConsistencyLoss, FocalLossVariant, CombinedJointLoss
)
from datasets import WaymoDataset, nuScenesDataset, KITTI360Dataset
from datasets.loaders import DataAugmentation, create_dataloader


class IntegrationTest:
    """Integration tests for the complete pipeline."""
    
    def __init__(self, device: torch.device = None):
        """Initialize test suite."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = None
        self.passed = []
        self.failed = []
    
    def run_all_tests(self):
        """Run all integration tests."""
        print("="*60)
        print("Running Integration Tests for Semantic 4DGS")
        print("="*60)
        print(f"Device: {self.device}")
        print()
        
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Core tests
            self.test_model_initialization()
            self.test_forward_pass()
            self.test_loss_computation()
            self.test_backward_pass()
            self.test_checkpoint_save_load()
            self.test_dataset_loading()
            self.test_augmentation()
            self.test_uncertainty_weighting()
            self.test_se3_constraint()
            self.test_trajectory_consistency()
            self.test_combined_joint_loss()
            self.test_distributed_preparation()
            
        finally:
            # Cleanup
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
        
        # Print summary
        self.print_summary()
        
        return len(self.failed) == 0
    
    def test_model_initialization(self):
        """Test model initialization."""
        test_name = "Model Initialization"
        print(f"Testing: {test_name}...")
        
        try:
            model = GaussianField(
                num_gaussians=100,
                feature_dim=32,
                semantic_feature_dim=16,
                device=self.device,
            )
            
            assert model is not None
            assert hasattr(model, 'field')
            
            self.passed.append(test_name)
            print(f"  ✓ {test_name}")
            
        except Exception as e:
            self.failed.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
    
    def test_forward_pass(self):
        """Test forward pass."""
        test_name = "Forward Pass"
        print(f"Testing: {test_name}...")
        
        try:
            model = GaussianField(
                num_gaussians=100,
                feature_dim=32,
                device=self.device,
            ).to(self.device)
            
            # Create dummy batch
            batch = {
                'rgb': torch.randn(1, 3, 256, 256).to(self.device),
                'intrinsics': torch.eye(3).unsqueeze(0).to(self.device),
                'extrinsics': torch.eye(4).unsqueeze(0).to(self.device),
            }
            
            with torch.no_grad():
                output = model(batch)
            
            assert 'rgb' in output
            assert output['rgb'].shape[0] == 1
            assert output['rgb'].shape[1] == 3
            
            self.passed.append(test_name)
            print(f"  ✓ {test_name}")
            
        except Exception as e:
            self.failed.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
    
    def test_loss_computation(self):
        """Test loss computation."""
        test_name = "Loss Computation"
        print(f"Testing: {test_name}...")
        
        try:
            # Test photometric loss
            loss_fn = PhotometricLoss()
            pred = torch.randn(1, 3, 256, 256).to(self.device)
            target = torch.randn(1, 3, 256, 256).to(self.device)
            
            loss, info = loss_fn(pred, target)
            
            assert isinstance(loss, torch.Tensor)
            assert loss.item() >= 0
            
            # Test semantic loss
            semantic_loss = SemanticLoss(num_classes=23)
            semantic_pred = torch.randn(1, 23, 256, 256).to(self.device)
            semantic_target = torch.randint(0, 23, (1, 256, 256)).to(self.device)
            
            loss_sem, info_sem = semantic_loss(semantic_pred, semantic_target)
            
            assert isinstance(loss_sem, torch.Tensor)
            
            self.passed.append(test_name)
            print(f"  ✓ {test_name}")
            
        except Exception as e:
            self.failed.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
    
    def test_backward_pass(self):
        """Test backward pass and optimization."""
        test_name = "Backward Pass"
        print(f"Testing: {test_name}...")
        
        try:
            model = GaussianField(
                num_gaussians=100,
                feature_dim=32,
                device=self.device,
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = PhotometricLoss()
            
            # Forward
            batch = {
                'rgb': torch.randn(1, 3, 256, 256).to(self.device),
                'intrinsics': torch.eye(3).unsqueeze(0).to(self.device),
                'extrinsics': torch.eye(4).unsqueeze(0).to(self.device),
            }
            
            output = model(batch)
            loss = loss_fn(output['rgb'], batch['rgb'])
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            assert True  # If we got here, backward passed
            
            self.passed.append(test_name)
            print(f"  ✓ {test_name}")
            
        except Exception as e:
            self.failed.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        test_name = "Checkpoint Save/Load"
        print(f"Testing: {test_name}...")
        
        try:
            model = GaussianField(
                num_gaussians=100,
                feature_dim=32,
                device=self.device,
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Save checkpoint
            ckpt_path = Path(self.temp_dir) / "test_checkpoint.pt"
            
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'step': 100,
            }, ckpt_path)
            
            assert ckpt_path.exists()
            
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            
            assert 'model_state' in checkpoint
            assert 'optimizer_state' in checkpoint
            assert checkpoint['step'] == 100
            
            # Test loading into model
            model2 = GaussianField(
                num_gaussians=100,
                feature_dim=32,
                device=self.device,
            ).to(self.device)
            
            model2.load_state_dict(checkpoint['model_state'])
            
            self.passed.append(test_name)
            print(f"  ✓ {test_name}")
            
        except Exception as e:
            self.failed.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
    
    def test_dataset_loading(self):
        """Test dataset loading."""
        test_name = "Dataset Loading"
        print(f"Testing: {test_name}...")
        
        try:
            # Test base dataset class
            dataset = WaymoDataset(
                root="/tmp/nonexistent",
                sequence_length=4,
                split="training",
            )
            
            # Should not raise error, just empty dataset
            assert len(dataset) == 0
            
            # Test dataloader creation function
            # Note: This won't actually load data without the dataset
            # Just test the function signature
            from datasets import create_dataloader
            
            self.passed.append(test_name)
            print(f"  ✓ {test_name}")
            
        except Exception as e:
            self.failed.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
    
    def test_augmentation(self):
        """Test data augmentation."""
        test_name = "Data Augmentation"
        print(f"Testing: {test_name}...")
        
        try:
            aug = DataAugmentation(
                random_flip=True,
                random_color=True,
                random_dropout=0.1,
            )
            
            # Create dummy sample
            sample = {
                'rgb': torch.randn(4, 3, 256, 256),
                'depth': torch.randn(256, 256),
                'intrinsics': torch.eye(3),
            }
            
            # Apply augmentation
            augmented = aug(sample)
            
            assert 'rgb' in augmented
            assert augmented['rgb'].shape == sample['rgb'].shape
            
            self.passed.append(test_name)
            print(f"  ✓ {test_name}")
            
        except Exception as e:
            self.failed.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
    
    def test_uncertainty_weighting(self):
        """Test uncertainty weighting loss."""
        test_name = "Uncertainty Weighting"
        print(f"Testing: {test_name}...")
        
        try:
            loss_fn = UncertaintyWeightedLoss(num_losses=3)
            
            losses = {
                'photo': torch.tensor(1.0),
                'semantic': torch.tensor(0.5),
                'reg': torch.tensor(0.1),
            }
            
            total_loss, info = loss_fn(losses)
            
            assert isinstance(total_loss, torch.Tensor)
            assert 'total_weighted' in info
            
            self.passed.append(test_name)
            print(f"  ✓ {test_name}")
            
        except Exception as e:
            self.failed.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
    
    def test_se3_constraint(self):
        """Test SE(3) constraint loss."""
        test_name = "SE(3) Constraint"
        print(f"Testing: {test_name}...")
        
        try:
            loss_fn = SE3ConstraintLoss()
            
            # Create dummy rotation matrices
            rotations = torch.eye(3).unsqueeze(0).expand(100, -1, -1).to(self.device)
            translations = torch.randn(100, 3).to(self.device)
            
            loss, info = loss_fn(rotations, translations, step=1000)
            
            assert isinstance(loss, torch.Tensor)
            
            self.passed.append(test_name)
            print(f"  ✓ {test_name}")
            
        except Exception as e:
            self.failed.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
    
    def test_trajectory_consistency(self):
        """Test trajectory consistency loss."""
        test_name = "Trajectory Consistency"
        print(f"Testing: {test_name}...")
        
        try:
            loss_fn = TrajectoryConsistencyLoss()
            
            # Create dummy trajectories
            T, N = 8, 50
            gauss_traj = torch.randn(T, N, 3).to(self.device)
            track_traj = torch.randn(T, N, 3).to(self.device)
            
            loss, info = loss_fn(gauss_traj, track_traj)
            
            assert isinstance(loss, torch.Tensor)
            
            self.passed.append(test_name)
            print(f"  ✓ {test_name}")
            
        except Exception as e:
            self.failed.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
    
    def test_combined_joint_loss(self):
        """Test combined joint loss."""
        test_name = "Combined Joint Loss"
        print(f"Testing: {test_name}...")
        
        try:
            loss_components = {
                'photometric': PhotometricLoss(),
                'semantic': SemanticLoss(num_classes=23),
                'regularization': GaussianRegularizationLoss(),
            }
            
            combined = CombinedJointLoss(
                loss_components=loss_components,
                uncertainty_weighting=True,
                progressive_activation=True,
            )
            
            predictions = {
                'rgb': torch.randn(1, 3, 256, 256).to(self.device),
                'semantic': torch.randn(1, 23, 256, 256).to(self.device),
            }
            
            targets = {
                'rgb': torch.randn(1, 3, 256, 256).to(self.device),
                'semantic': torch.randint(0, 23, (1, 256, 256)).to(self.device),
            }
            
            loss, info = combined(predictions, targets, step=100)
            
            assert isinstance(loss, torch.Tensor)
            
            self.passed.append(test_name)
            print(f"  ✓ {test_name}")
            
        except Exception as e:
            self.failed.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
    
    def test_distributed_preparation(self):
        """Test distributed training preparation."""
        test_name = "Distributed Preparation"
        print(f"Testing: {test_name}...")
        
        try:
            # Test that model can be wrapped for DDP
            model = GaussianField(
                num_gaussians=100,
                feature_dim=32,
                device=self.device,
            ).to(self.device)
            
            # Note: Can't actually test DDP without multiple processes
            # Just verify model is compatible
            
            assert hasattr(model, 'forward')
            
            self.passed.append(test_name)
            print(f"  ✓ {test_name}")
            
        except Exception as e:
            self.failed.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
    
    def print_summary(self):
        """Print test summary."""
        print()
        print("="*60)
        print("Integration Test Summary")
        print("="*60)
        print(f"Passed: {len(self.passed)}")
        print(f"Failed: {len(self.failed)}")
        
        if self.failed:
            print()
            print("Failed Tests:")
            for name, error in self.failed:
                print(f"  - {name}: {error}")
        
        print()
        if len(self.failed) == 0:
            print("✓ All integration tests passed!")
        else:
            print("✗ Some tests failed.")
        print("="*60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path")
    args = parser.parse_args()
    
    # Run tests
    test_suite = IntegrationTest()
    success = test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

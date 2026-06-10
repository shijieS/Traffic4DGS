"""
Models module for Semantic 4D Gaussian Splatting.

This module contains the core model components:
- GaussianField: Base class for 4D Gaussian representation
- StaticField: Static background Gaussian field
- DynamicField: Dynamic object Gaussian field with SE(3) + deformation
- SE3Transform: SE(3) rigid body transformation
- NonRigidDeform: Non-rigid deformation network
- SAM2Tracker: SAM2 video object segmentation integration
- PointTracker: TAPIR/CoTracker point tracking integration
- JointRenderer: Multi-modal Gaussian renderer
- JointOptimizer: Tracking-reconstruction joint optimizer

@author Semantic 4DGS Team
@version 1.0.0
"""

from .gaussian_field import GaussianField, GaussianModel
from .static_field import StaticField
from .dynamic_field import DynamicField
from .se3_transform import SE3Transform, SE3Parameter
from .nonrigid_deform import NonRigidDeformation, DeformationMLP
from .sam2_tracker import SAM2Tracker
from .point_tracker import PointTracker, TAPIRTracker, CoTracker
from .renderer import JointRenderer, GaussianRenderer
from .joint_optimizer import JointOptimizer

__all__ = [
    "GaussianField",
    "GaussianModel",
    "StaticField",
    "DynamicField",
    "SE3Transform",
    "SE3Parameter",
    "NonRigidDeformation",
    "DeformationMLP",
    "SAM2Tracker",
    "PointTracker",
    "TAPIRTracker",
    "CoTracker",
    "JointRenderer",
    "GaussianRenderer",
    "JointOptimizer",
]

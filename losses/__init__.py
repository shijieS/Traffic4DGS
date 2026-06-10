"""
Losses module for Semantic 4D Gaussian Splatting.
"""

from .photometric import PhotometricLoss, SSIMLoss, LPIPSLoss, DepthPhotometricLoss
from .semantic import SemanticLoss, InstanceAwareSemanticLoss, SemanticFeatureContrastiveLoss
from .silhouette import SilhouetteLoss, MultiScaleSilhouetteLoss, DepthSilhouetteConsistencyLoss
from .regularization import (
    GaussianRegularizationLoss,
    MotionSmoothnessLoss,
    DiffeomorphismRegularizationLoss,
)

__all__ = [
    "PhotometricLoss",
    "SSIMLoss",
    "LPIPSLoss",
    "DepthPhotometricLoss",
    "SemanticLoss",
    "InstanceAwareSemanticLoss",
    "SemanticFeatureContrastiveLoss",
    "SilhouetteLoss",
    "MultiScaleSilhouetteLoss",
    "DepthSilhouetteConsistencyLoss",
    "GaussianRegularizationLoss",
    "MotionSmoothnessLoss",
    "DiffeomorphismRegularizationLoss",
]

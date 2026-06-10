"""
Scripts module for Semantic 4D Gaussian Splatting.

Contains:
- train.py: Training script with DDP, logging, checkpoints
- eval.py: Evaluation script with comprehensive metrics
- test_integration.py: End-to-end integration tests
- preprocess.py: Data preprocessing utilities

@author Semantic 4DGS Team
@version 1.0.0
"""

from .train import main as train
from .eval import main as evaluate
from .test_integration import main as test_integration

__all__ = [
    "train",
    "evaluate",
    "test_integration",
]

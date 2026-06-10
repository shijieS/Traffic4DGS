# Semantic-aware Spatiotemporal Disentangled 4DGS for Traffic Scene Reconstruction

A novel framework for traffic scene 4D reconstruction with tracking-reconstruction joint optimization.

## Core Contributions
1. **Semantic-aware Spatiotemporal Disentangled 4DGS**: Separate static/dynamic components with semantic consistency
2. **Hybrid Deformation Priors with SE(3) Rigidity**: Rigid body constraints for vehicles + non-rigid for pedestrians
3. **Joint Mask-Geometry Rendering**: Self-supervised optimization via semantic + geometric rendering

## Project Structure
```
├── models/          # Core model implementations
├── losses/          # Loss functions
├── scripts/         # Training & evaluation scripts
├── configs/         # Configuration files
├── utils/           # Utility functions
└── datasets/        # Data loading
```

## Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+

## Quick Start
```bash
python scripts/train.py --config configs/default.yaml
```

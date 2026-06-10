"""
IO utilities for Semantic 4D Gaussian Splatting.
"""

import torch
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np


def save_checkpoint(
    path: Union[str, Path],
    model_state: Dict,
    optimizer_state: Optional[Dict] = None,
    scheduler_state: Optional[Dict] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    config: Optional[Any] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """Save training checkpoint.
    
    Args:
        path: Checkpoint path
        model_state: Model state dict
        optimizer_state: Optimizer state dict
        scheduler_state: Scheduler state dict
        epoch: Current epoch
        step: Current step
        config: Configuration object
        metadata: Additional metadata
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state': model_state,
    }
    
    if optimizer_state is not None:
        checkpoint['optimizer_state'] = optimizer_state
    if scheduler_state is not None:
        checkpoint['scheduler_state'] = scheduler_state
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if step is not None:
        checkpoint['step'] = step
    if config is not None:
        checkpoint['config'] = config
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Dict:
    """Load training checkpoint.
    
    Args:
        path: Checkpoint path
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    print(f"Checkpoint loaded from {path}")
    
    return checkpoint


def save_json(
    path: Union[str, Path],
    data: Dict,
    indent: int = 2,
) -> None:
    """Save dictionary to JSON file.
    
    Args:
        path: Output path
        data: Dictionary to save
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(path: Union[str, Path]) -> Dict:
    """Load dictionary from JSON file.
    
    Args:
        path: JSON file path
        
    Returns:
        Loaded dictionary
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_pickle(
    path: Union[str, Path],
    data: Any,
) -> None:
    """Save data to pickle file.
    
    Args:
        path: Output path
        data: Data to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load data from pickle file.
    
    Args:
        path: Pickle file path
        
    Returns:
        Loaded data
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_point_cloud(
    path: Union[str, Path],
    points: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    format: str = "ply",
) -> None:
    """Save point cloud to file.
    
    Args:
        path: Output path
        points: Point coordinates [N, 3]
        colors: Optional RGB colors [N, 3]
        format: File format ('ply' or 'xyz')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if colors is not None and isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()
    
    if format == "ply":
        try:
            from plyfile import PlyData, PlyElement
            
            vertices = points
            if colors is not None:
                vertices = np.hstack([points, colors * 255])
            
            vertices = vertices.astype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')] + 
                                        [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')] 
                                        if colors is not None else [])
            
            if colors is None:
                vertices = points.astype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            
            el = PlyElement.describe(vertices, 'vertex')
            PlyData([el]).write(path)
            
        except ImportError:
            # Fallback: save as XYZ
            np.savetxt(path.with_suffix('.xyz'), points)
    
    elif format == "xyz":
        np.savetxt(path, points)


def load_point_cloud(
    path: Union[str, Path],
    format: str = "auto",
) -> tuple:
    """Load point cloud from file.
    
    Args:
        path: Point cloud file path
        format: File format ('ply', 'xyz', 'auto')
        
    Returns:
        Tuple of (points, colors)
    """
    path = Path(path)
    
    if format == "auto":
        format = path.suffix.lstrip('.')
    
    if format == "ply":
        try:
            from plyfile import PlyData
            
            plydata = PlyData.read(path)
            vertex = plydata['vertex']
            
            points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
            
            if 'red' in vertex.data.dtype.names:
                colors = np.vstack([
                    vertex['red'], vertex['green'], vertex['blue']
                ]).T / 255.0
            else:
                colors = None
            
        except ImportError:
            # Fallback to numpy
            data = np.loadtxt(path)
            points = data[:, :3]
            colors = data[:, 3:6] if data.shape[1] >= 6 else None
    
    elif format == "xyz":
        data = np.loadtxt(path)
        points = data[:, :3]
        colors = data[:, 3:6] if data.shape[1] >= 6 else None
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return torch.from_numpy(points), torch.from_numpy(colors) if colors is not None else None


class CheckpointManager:
    """Manager for checkpoint saving and loading.
    
    Keeps track of checkpoints and manages storage.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
    ) -> None:
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
    
    def save(
        self,
        model_state: Dict,
        step: int,
        optimizer_state: Optional[Dict] = None,
        **kwargs,
    ) -> Path:
        """Save checkpoint and manage storage.
        
        Args:
            model_state: Model state dict
            step: Current step
            optimizer_state: Optimizer state dict
            **kwargs: Additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:06d}.pt"
        
        save_checkpoint(
            path=checkpoint_path,
            model_state=model_state,
            optimizer_state=optimizer_state,
            step=step,
            **kwargs,
        )
        
        self.checkpoints.append(checkpoint_path)
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
                print(f"Removed old checkpoint: {old_path}")
        
        return checkpoint_path
    
    def load_latest(self, device: Optional[torch.device] = None) -> Optional[Dict]:
        """Load latest checkpoint.
        
        Args:
            device: Device to load to
            
        Returns:
            Checkpoint dict or None
        """
        if not self.checkpoints:
            # Scan directory
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
            if checkpoints:
                self.checkpoints = checkpoints
            else:
                return None
        
        if self.checkpoints:
            return load_checkpoint(self.checkpoints[-1], device)
        
        return None

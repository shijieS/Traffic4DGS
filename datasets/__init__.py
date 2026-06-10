"""
Dataset Loaders for Traffic Scene Reconstruction.

Supports Waymo Open Dataset, nuScenes, and KITTI-360.

Example usage:
    >>> from datasets import WaymoDataset, nuScenesDataset, KITT360Dataset
    >>> dataset = WaymoDataset(root="/data/waymo", sequence_length=16)
    >>> sample = dataset[0]
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import json


class BaseTrafficDataset(Dataset):
    """Base class for traffic scene datasets."""
    
    def __init__(
        self,
        root: str,
        sequence_length: int = 16,
        stride: int = 1,
        image_size: Tuple[int, int] = (1920, 1080),
        scale_factor: float = 1.0,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize dataset.
        
        Args:
            root: Dataset root directory
            sequence_length: Number of frames per sequence
            stride: Frame sampling stride
            image_size: Target image size (H, W)
            scale_factor: Downsampling factor
            transform: Optional transform
        """
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.transform = transform
        
        self.frames = []
        self._load_frames()
    
    def _load_frames(self) -> None:
        """Load frame metadata. Override in subclass."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample.
        
        Returns:
            Dictionary with:
                - rgb: RGB images [T, 3, H, W]
                - camera: Camera parameters
                - semantic: Semantic labels (if available)
                - depth: Depth maps (if available)
                - intrinsics: Camera intrinsics [3, 3]
                - extrinsics: Camera extrinsics [4, 4]
        """
        raise NotImplementedError
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess image."""
        from PIL import Image
        import torchvision.transforms as transforms
        
        img = Image.open(path).convert('RGB')
        
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(transform_list)
        
        return transform(img)
    
    def _load_depth(self, path: str) -> torch.Tensor:
        """Load depth map."""
        depth = np.load(path).astype(np.float32)
        return torch.from_numpy(depth)
    
    def _load_semantic(self, path: str) -> torch.Tensor:
        """Load semantic segmentation."""
        semantic = np.load(path).astype(np.int64)
        return torch.from_numpy(semantic)


class WaymoDataset(BaseTrafficDataset):
    """Waymo Open Dataset loader.
    
    Structure:
        waymo/
            training/
                scene_0001/
                    front_camera/
                        0001.jpg
                        ...
                    lidar/
                        0001.npy
                    annotations/
                        0001.json
            validation/
                ...
    """
    
    def __init__(
        self,
        root: str,
        split: str = "training",
        cameras: List[str] = None,
        sequence_length: int = 16,
        stride: int = 1,
        image_size: Tuple[int, int] = (1920, 1080),
        scale_factor: float = 0.25,
    ) -> None:
        """Initialize Waymo dataset.
        
        Args:
            root: Waymo dataset root
            split: 'training' or 'validation'
            cameras: List of cameras to load
            sequence_length: Frames per sequence
            stride: Frame stride
            image_size: Target image size
            scale_factor: Downsample factor
        """
        self.split = split
        self.cameras = cameras or ["front_camera"]
        
        super().__init__(
            root=root,
            sequence_length=sequence_length,
            stride=stride,
            image_size=image_size,
            scale_factor=scale_factor,
        )
    
    def _load_frames(self) -> None:
        """Load Waymo frames."""
        split_dir = self.root / self.split
        
        if not split_dir.exists():
            print(f"Warning: Waymo split {self.split} not found at {split_dir}")
            return
        
        # Scan scene directories
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            for camera in self.cameras:
                camera_dir = scene_dir / camera
                if not camera_dir.exists():
                    continue
                
                # Get frame files
                frames = sorted(camera_dir.glob("*.jpg")) + \
                        sorted(camera_dir.glob("*.png"))
                
                # Create sequences
                for i in range(0, len(frames) - self.sequence_length + 1, self.stride):
                    seq_frames = frames[i:i + self.sequence_length]
                    self.frames.append({
                        'scene': scene_dir.name,
                        'camera': camera,
                        'paths': [str(f) for f in seq_frames],
                        'lidar_dir': str(scene_dir / "lidar"),
                        'annotation_dir': str(scene_dir / "annotations"),
                    })
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get Waymo sample."""
        frame_data = self.frames[idx]
        
        # Load images
        images = []
        for path in frame_data['paths']:
            img = self._load_image(path)
            images.append(img)
        
        rgb = torch.stack(images, dim=0)  # [T, 3, H, W]
        
        # Camera parameters (placeholder)
        H, W = self.image_size
        fx = W * 0.7  # Approximate focal length
        fy = H * 0.7
        cx = W / 2
        cy = H / 2
        
        intrinsics = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        extrinsics = torch.eye(4, dtype=torch.float32)
        
        # Load lidar points if available
        lidar_path = frame_data['paths'][0].replace('.jpg', '.npy').replace(
            frame_data['camera'], 'lidar'
        )
        if Path(lidar_path).exists():
            points = np.load(lidar_path)
            points = torch.from_numpy(points).float()
        else:
            points = torch.zeros(0, 3)
        
        sample = {
            'rgb': rgb,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'points': points,
            'scene': frame_data['scene'],
            'camera': frame_data['camera'],
            'paths': frame_data['paths'],
        }
        
        # Load annotations if available
        annotation_path = frame_data['paths'][0].replace('.jpg', '.json').replace(
            frame_data['camera'], 'annotations'
        )
        if Path(annotation_path).exists():
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            sample['annotations'] = annotations
        
        return sample


class nuScenesDataset(BaseTrafficDataset):
    """nuScenes dataset loader.
    
    Structure:
        nuscenes/
            v1.0-trainval/
                samples/
                    CAM_FRONT/
                        n015-2018-07-22-11-15-16+0800__CAM_FRONT__1531282764047954.jpg
                        ...
                sweeps/
                lidarseg/
                maps/
    """
    
    def __init__(
        self,
        root: str,
        version: str = "v1.0-trainval",
        cameras: List[str] = None,
        sequence_length: int = 16,
        stride: int = 1,
        image_size: Tuple[int, int] = (1600, 900),
        scale_factor: float = 0.5,
        use_lidarseg: bool = True,
    ) -> None:
        """Initialize nuScenes dataset."""
        self.version = version
        self.cameras = cameras or ["CAM_FRONT"]
        self.use_lidarseg = use_lidarseg
        
        super().__init__(
            root=root,
            sequence_length=sequence_length,
            stride=stride,
            image_size=image_size,
            scale_factor=scale_factor,
        )
    
    def _load_frames(self) -> None:
        """Load nuScenes frames."""
        # Implementation similar to WaymoDataset
        # Would need nuScenes API for actual loading
        print("nuScenes dataset loader - placeholder")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get nuScenes sample."""
        # Placeholder
        return {
            'rgb': torch.zeros(self.sequence_length, 3, *self.image_size),
            'intrinsics': torch.eye(3),
            'extrinsics': torch.eye(4),
            'points': torch.zeros(0, 3),
        }


class KITTI360Dataset(BaseTrafficDataset):
    """KITTI-360 dataset loader.
    
    Structure:
        kitti360/
            data_3d_raw/
                2013_05_28_drive_0000_sync/
                    velodyne_points/
                        data/
                    image_02/
                        data/
                    image_03/
                        data/
    """
    
    def __init__(
        self,
        root: str,
        sequence_name: str = "2013_05_28_drive_0000_sync",
        cameras: List[str] = None,
        sequence_length: int = 16,
        stride: int = 1,
        image_size: Tuple[int, int] = (1920, 1080),
        scale_factor: float = 0.5,
    ) -> None:
        """Initialize KITTI-360 dataset."""
        self.sequence_name = sequence_name
        self.cameras = cameras or ["image_02"]
        
        super().__init__(
            root=root,
            sequence_length=sequence_length,
            stride=stride,
            image_size=image_size,
            scale_factor=scale_factor,
        )
    
    def _load_frames(self) -> None:
        """Load KITTI-360 frames."""
        seq_dir = self.root / "data_3d_raw" / self.sequence_name
        
        if not seq_dir.exists():
            print(f"Warning: KITTI-360 sequence not found at {seq_dir}")
            return
        
        for camera in self.cameras:
            image_dir = seq_dir / camera / "data"
            if not image_dir.exists():
                continue
            
            frames = sorted(image_dir.glob("*.png"))
            
            for i in range(0, len(frames) - self.sequence_length + 1, self.stride):
                seq_frames = frames[i:i + self.sequence_length]
                self.frames.append({
                    'sequence': self.sequence_name,
                    'camera': camera,
                    'paths': [str(f) for f in seq_frames],
                    'calibration': str(seq_dir / "calibration.txt"),
                })
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get KITTI-360 sample."""
        frame_data = self.frames[idx]
        
        images = []
        for path in frame_data['paths']:
            img = self._load_image(path)
            images.append(img)
        
        rgb = torch.stack(images, dim=0)
        
        # Load calibration
        intrinsics, extrinsics = self._load_calibration(
            frame_data['calibration'],
            frame_data['camera']
        )
        
        # Load velodyne points
        velodyne_path = frame_data['paths'][0].replace(
            frame_data['camera'], 'velodyne_points/data'
        ).replace('.png', '.bin')
        
        if Path(velodyne_path).exists():
            points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
            points = torch.from_numpy(points[:, :3]).float()
        else:
            points = torch.zeros(0, 3)
        
        return {
            'rgb': rgb,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'points': points,
            'sequence': frame_data['sequence'],
            'camera': frame_data['camera'],
        }
    
    def _load_calibration(
        self,
        calib_path: str,
        camera: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load camera calibration."""
        # Simplified calibration loading
        # Actual implementation would parse calibration files
        H, W = self.image_size
        
        intrinsics = torch.tensor([
            [W * 0.5, 0, W / 2],
            [0, H * 0.5, H / 2],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        extrinsics = torch.eye(4, dtype=torch.float32)
        
        return intrinsics, extrinsics


def create_dataloader(
    dataset_name: str,
    root: str,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    """Create dataloader for specified dataset.
    
    Args:
        dataset_name: 'waymo', 'nuscenes', or 'kitti360'
        root: Dataset root directory
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Shuffle data
        **kwargs: Additional dataset arguments
        
    Returns:
        DataLoader instance
    """
    if dataset_name.lower() == "waymo":
        dataset = WaymoDataset(root=root, **kwargs)
    elif dataset_name.lower() == "nuscenes":
        dataset = nuScenesDataset(root=root, **kwargs)
    elif dataset_name.lower() == "kitti360":
        dataset = KITTI360Dataset(root=root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )

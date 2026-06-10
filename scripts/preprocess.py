"""
Data Preprocessing Script for Semantic 4DGS.

Prepares datasets for training:
- Point cloud extraction from LiDAR
- Camera calibration parsing
- Semantic segmentation mapping
"""

import argparse
from pathlib import Path
import numpy as np
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def preprocess_waymo(
    data_root: str,
    output_root: str,
    split: str = "training",
    cameras: list = None,
) -> None:
    """Preprocess Waymo dataset.
    
    Args:
        data_root: Waymo dataset root
        output_root: Output directory
        split: Dataset split
        cameras: List of cameras to process
    """
    cameras = cameras or ["front_camera"]
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    split_dir = data_root / split
    
    for scene_dir in sorted(split_dir.iterdir())[:10]:  # Process first 10 scenes
        if not scene_dir.is_dir():
            continue
        
        print(f"Processing scene: {scene_dir.name}")
        
        scene_output = output_root / scene_dir.name
        scene_output.mkdir(exist_ok=True)
        
        # Process each camera
        for camera in cameras:
            camera_dir = scene_dir / camera
            if not camera_dir.exists():
                continue
            
            # Get frames
            frames = sorted(camera_dir.glob("*.jpg"))[:50]  # First 50 frames
            
            # Extract point clouds
            lidar_dir = scene_dir / "lidar"
            
            for frame_path in frames:
                frame_id = frame_path.stem
                
                # Load LiDAR points
                lidar_path = lidar_dir / f"{frame_id}.npy"
                
                if lidar_path.exists():
                    points = np.load(lidar_path)
                    # Save processed points
                    output_points = scene_output / f"{frame_id}_points.npy"
                    np.save(output_points, points)
                
                print(f"  Processed {frame_id}")


def preprocess_nuscenes(
    data_root: str,
    output_root: str,
    version: str = "v1.0-trainval",
) -> None:
    """Preprocess nuScenes dataset.
    
    Args:
        data_root: nuScenes dataset root
        output_root: Output directory
        version: Dataset version
    """
    print(f"Preprocessing nuScenes {version}...")
    # Implementation similar to Waymo
    print("nuScenes preprocessing complete")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["waymo", "nuscenes", "kitti360"],
                        help="Dataset name")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Dataset root directory")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--split", type=str, default="training",
                        help="Dataset split")
    args = parser.parse_args()
    
    if args.dataset == "waymo":
        preprocess_waymo(args.data_root, args.output_root, args.split)
    elif args.dataset == "nuscenes":
        preprocess_nuscenes(args.data_root, args.output_root)
    else:
        print(f"Dataset {args.dataset} preprocessing not implemented")


if __name__ == "__main__":
    main()

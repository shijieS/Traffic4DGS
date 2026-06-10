"""
Visualization utilities for Semantic 4D Gaussian Splatting.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple
from pathlib import Path


def visualize_gaussian_3d(
    positions: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    output_path: Optional[str] = None,
) -> None:
    """Visualize 3D Gaussians as point cloud.
    
    Args:
        positions: Gaussian positions [N, 3]
        colors: RGB colors [N, 3]
        scales: Gaussian scales [N, 3]
        output_path: Optional path to save visualization
    """
    try:
        import open3d as o3d
        
        points = positions.cpu().numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            colors_np = colors.cpu().numpy()
            colors_np = np.clip(colors_np, 0, 1)
            pcd.colors = o3d.utility.Vector3dVector(colors_np)
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        
    except ImportError:
        # Fallback to matplotlib scatter
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        points = positions.cpu().numpy()
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
        plt.close()


def visualize_rendering(
    rgb: torch.Tensor,
    depth: Optional[torch.Tensor] = None,
    semantic: Optional[torch.Tensor] = None,
    silhouette: Optional[torch.Tensor] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> np.ndarray:
    """Visualize rendering outputs.
    
    Args:
        rgb: RGB image [3, H, W] or [H, W, 3]
        depth: Depth map [H, W]
        semantic: Semantic map [H, W] or [C, H, W]
        silhouette: Silhouette [H, W]
        output_path: Path to save visualization
        figsize: Figure size
        
    Returns:
        Figure as numpy array
    """
    # Convert to numpy
    if rgb.dim() == 3 and rgb.shape[0] == 3:
        rgb = rgb.permute(1, 2, 0)
    rgb_np = rgb.cpu().numpy()
    
    num_plots = 1 + (depth is not None) + (semantic is not None) + (silhouette is not None)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(figsize[0] * num_plots // 4, figsize[1]))
    if num_plots == 1:
        axes = [axes]
    
    ax_idx = 0
    
    # RGB
    axes[ax_idx].imshow(rgb_np)
    axes[ax_idx].set_title("RGB")
    axes[ax_idx].axis('off')
    ax_idx += 1
    
    # Depth
    if depth is not None:
        depth_np = depth.cpu().numpy()
        im = axes[ax_idx].imshow(depth_np, cmap='turbo')
        axes[ax_idx].set_title("Depth")
        axes[ax_idx].axis('off')
        plt.colorbar(im, ax=axes[ax_idx])
        ax_idx += 1
    
    # Semantic
    if semantic is not None:
        if semantic.dim() == 3:
            semantic = torch.argmax(semantic, dim=0)
        semantic_np = semantic.cpu().numpy()
        axes[ax_idx].imshow(semantic_np, cmap='tab20')
        axes[ax_idx].set_title("Semantic")
        axes[ax_idx].axis('off')
        ax_idx += 1
    
    # Silhouette
    if silhouette is not None:
        sil_np = silhouette.cpu().numpy()
        axes[ax_idx].imshow(sil_np, cmap='gray')
        axes[ax_idx].set_title("Silhouette")
        axes[ax_idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # Convert to array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return img


def visualize_trajectories(
    trajectories: Dict[int, List[Tuple[float, float, float]]],
    output_path: Optional[str] = None,
) -> None:
    """Visualize object trajectories in top-down view.
    
    Args:
        trajectories: Dict mapping instance_id to list of (x, y, z) positions
        output_path: Path to save visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))
    
    for idx, (inst_id, traj) in enumerate(trajectories.items()):
        traj_arr = np.array(traj)
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], '-o', 
                color=colors[idx], markersize=2, linewidth=1, alpha=0.7)
        ax.scatter(traj_arr[0, 0], traj_arr[0, 1], marker='s', 
                  color=colors[idx], s=50, label=f'Instance {inst_id}')
        ax.scatter(traj_arr[-1, 0], traj_arr[-1, 1], marker='^', 
                  color=colors[idx], s=50)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Object Trajectories (Top-Down View)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def save_video_frames(
    frames: List[Dict],
    output_dir: str,
    prefix: str = "frame",
) -> None:
    """Save video frames to directory.
    
    Args:
        frames: List of frame dictionaries with 'rgb', 'depth', etc.
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, frame in enumerate(tqdm(frames, desc="Saving frames")):
        if 'rgb' in frame:
            rgb = frame['rgb']
            if rgb.dim() == 3 and rgb.shape[0] == 3:
                rgb = rgb.permute(1, 2, 0)
            rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
            
            img_path = output_path / f"{prefix}_{idx:04d}_rgb.png"
            plt.imsave(img_path, rgb_np)
        
        if 'depth' in frame:
            depth = frame['depth'].cpu().numpy()
            depth_path = output_path / f"{prefix}_{idx:04d}_depth.png"
            plt.imsave(depth_path, depth, cmap='turbo')
        
        if 'semantic' in frame:
            semantic = frame['semantic']
            if semantic.dim() == 3:
                semantic = torch.argmax(semantic, dim=0)
            semantic_np = semantic.cpu().numpy()
            sem_path = output_path / f"{prefix}_{idx:04d}_semantic.png"
            plt.imsave(sem_path, semantic_np, cmap='tab20')

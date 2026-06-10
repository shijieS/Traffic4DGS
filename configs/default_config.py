"""
Default Configuration for Semantic 4DGS Traffic Scene Reconstruction.

@fileoverview Default configuration using YAML + dataclass pattern
@author Semantic 4DGS Team
@version 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Configuration for 4D Gaussian Splatting model.
    
    Mathematical formulation:
    - 3D Gaussian: G(x) = exp(-0.5 * (x-μ)ᵀΣ⁻¹(x-μ))
    - Covariance decomposition: Σ = R @ S @ Sᵀ @ Rᵀ where R∈SO(3), S=diag(s)
    - 4D extension adds temporal dimension with temporal covariance Σₜ
    
    Attributes:
        init_num_gaussians: Initial number of Gaussian primitives
        max_num_gaussians: Maximum number of Gaussians for densification
        spatial_scale: Global scale factor for scene normalization
        position_bound: Scene bounds in meters for Gaussian initialization
        init_scale: Initial scale for covariance matrix
        scale_residual_init: Initial residual for scale optimization
        feature_dim: Dimension of appearance features (RGB/sh)
        semantic_feature_dim: Dimension of semantic feature vectors
        deformation_network: MLP architecture for non-rigid deformation
        se3_refine_rotation: Enable rotation optimization in SE(3)
        se3_refine_translation: Enable translation optimization in SE(3)
        temporal_window: Number of frames in temporal window
        temporal_subsample: Subsampling rate for temporal modeling
    """
    
    init_num_gaussians: int = 5000
    max_num_gaussians: int = 200000
    spatial_scale: float = 1.0
    position_bound: float = 10.0
    init_scale: float = 0.01
    scale_residual_init: float = 0.001
    feature_dim: int = 32
    semantic_feature_dim: int = 256
    se3_refine_rotation: bool = True
    se3_refine_translation: bool = True
    temporal_window: int = 16
    temporal_subsample: int = 2
    deformation_network: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_dims": [64, 128, 256],
        "activation": "relu",
        "output_dim": 3
    })


@dataclass
class Sam2Config:
    """Configuration for SAM2 tracker integration.
    
    SAM2 uses a memory-based architecture for video object segmentation:
    Mₜ = Attention(Qₜ, K_memory, V_memory) + Memory update
    
    The memory mechanism allows tracking objects across frames by maintaining
    a memory bank of previous mask predictions and image features.
    
    Attributes:
        model_type: SAM2 model variant (sam2_t, sam2_s, sam2_m, sam2.1_b++, sam2.1_l)
        checkpoint_path: Path to pretrained SAM2 checkpoint
        device: Device for inference ('cuda' or 'cpu')
        memory_temporal_len: Maximum temporal length for memory bank
        memory_stride: Stride for memory feature extraction
        points_per_side: Points per side for mask generation grid
        points_per_batch: Number of points to process in parallel
        pred_iou_thresh: IoU threshold for mask prediction filtering
        stability_score_thresh: Stability score threshold for mask filtering
        tracked_classes: List of semantic classes to track
    """
    
    model_type: str = "sam2.1_b++"
    checkpoint_path: Optional[str] = None
    device: str = "cuda"
    memory_temporal_len: int = 10
    memory_stride: int = 1
    points_per_side: int = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.8
    stability_score_thresh: float = 0.95
    tracked_classes: List[str] = field(default_factory=lambda: [
        "car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"
    ])


@dataclass
class PointTrackerConfig:
    """Configuration for point tracker (TAPIR/CoTracker).
    
    TAPIR architecture:
    - Initialization: Feature pyramid F + bilinear interpolation
    - Tracking: Correlation search + GRU update
    qₜ = Corr(F₀, Fₜ) + GRU(qₜ₋₁, memory)
    
    CoTracker architecture:
    - Grid-based query points across the video
    - Correlation-based feature matching
    - Iterative refinement with attention
    
    Attributes:
        tracker_type: Point tracker variant ('tapir' or 'cotracker')
        checkpoint_path: Path to pretrained checkpoint
        cotracker_grid_size: Query grid size for CoTracker
        cotracker_backbone: Vision backbone for CoTracker
        tapir_num_stages: Number of refinement stages for TAPIR
        track_on_sam2_masks: Whether to track within SAM2 mask regions
        max_tracks_per_instance: Maximum tracks per object instance
    """
    
    tracker_type: str = "cotracker"
    checkpoint_path: Optional[str] = None
    cotracker_grid_size: int = 10
    cotracker_backbone: str = "vit_b"
    tapir_num_stages: int = 4
    track_on_sam2_masks: bool = True
    max_tracks_per_instance: int = 100


@dataclass
class TrainingConfig:
    """Configuration for training process.
    
    Attributes:
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        learning_rate: Initial learning rate
        lr_decay: Learning rate decay factor
        lr_decay_steps: Steps between learning rate decay
        weight_decay: Weight decay (L2 regularization)
        batch_size: Training batch size
        num_epochs: Number of training epochs
        accumulation_steps: Gradient accumulation steps
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer
        photometric_weight: Weight for photometric loss
        semantic_weight: Weight for semantic loss
        silhouette_weight: Weight for silhouette loss
        tracking_weight: Weight for tracking consistency loss
        regularization_weight: Weight for regularization terms
        max_grad_norm: Maximum gradient norm for clipping
        gradient_check: Enable gradient monitoring
        densify_every: Steps between densification
        densify_threshold: Threshold for Gaussian densification
        prune_threshold: Threshold for Gaussian pruning
        save_every: Steps between checkpoint saves
        val_every: Steps between validation
        max_checkpoints: Maximum number of checkpoints to keep
    """
    
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    lr_decay: float = 0.95
    lr_decay_steps: int = 1000
    weight_decay: float = 1e-4
    batch_size: int = 1
    num_epochs: int = 30
    accumulation_steps: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    photometric_weight: float = 1.0
    semantic_weight: float = 0.1
    silhouette_weight: float = 0.05
    tracking_weight: float = 0.5
    regularization_weight: float = 0.01
    max_grad_norm: float = 1.0
    gradient_check: bool = True
    densify_every: int = 100
    densify_threshold: float = 0.0002
    prune_threshold: float = 0.0001
    save_every: int = 1000
    val_every: int = 500
    max_checkpoints: int = 5


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.
    
    Supports Waymo Open Dataset, nuScenes, and KITTI-360.
    
    Attributes:
        dataset_name: Dataset identifier ('waymo', 'nuscenes', 'kitti360')
        data_root: Root directory for dataset
        sequence_length: Number of frames per sequence
        stride: Frame sampling stride
        image_size: Input image resolution [H, W]
        scale_factor: Downsampling factor
        waymo_split: Waymo split ('training', 'validation')
        waymo_num_frames: Maximum frames per Waymo sequence
        nuscenes_version: nuScenes version string
        nuscenes_sweeps: Number of sweeps for nuScenes
        kitti360_sequence_start: Start index for KITTI-360 sequence
        kitti360_sequence_end: End index for KITTI-360 sequence
        num_classes: Number of semantic classes
        class_names: List of class names
        dynamic_classes: Classes considered as dynamic objects
    """
    
    dataset_name: str = "waymo"
    data_root: str = "./data"
    sequence_length: int = 16
    stride: int = 1
    image_size: List[int] = field(default_factory=lambda: [1920, 1080])
    scale_factor: float = 0.25
    waymo_split: str = "training"
    waymo_num_frames: int = 200
    nuscenes_version: str = "v1.0-trainval"
    nuscenes_sweeps: int = 20
    kitti360_sequence_start: int = 0
    kitti360_sequence_end: int = 100
    num_classes: int = 23
    class_names: List[str] = field(default_factory=lambda: [
        "unlabeled", "car", "truck", "bus", "motorcycle", "bicycle",
        "pedestrian", "rider", "traffic_light", "traffic_sign",
        "sky", "vegetation", "terrain", "building", "pole",
        "fence", "guard_rail", "dynamic", "static", "ground",
        "bridge", "parking", "wall"
    ])
    dynamic_classes: List[str] = field(default_factory=lambda: [
        "car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "rider"
    ])


@dataclass
class RenderConfig:
    """Configuration for rendering pipeline.
    
    Gaussian rendering equation:
    C = Σᵢ cᵢ αᵢ ∏ⱼ<ᵢ (1 - αⱼ)
    
    where:
    - cᵢ = color of Gaussian i (RGB or semantic)
    - αᵢ = opacity × G₂D(projected_covariance)
    
    Attributes:
        render_resolution: Output rendering resolution [H, W]
        downsample_factor: Downsampling factor for faster rendering
        rasterize_mode: Rasterization mode ('antialias', 'sync', 'classic')
        gaussian_scale_threshold: Minimum Gaussian scale to render
        alpha_threshold: Alpha cutoff threshold
        camera_model: Camera model type
        camera_baseline: Stereo baseline for multi-view rendering
        render_rgb: Enable RGB rendering
        render_depth: Enable depth rendering
        render_semantic: Enable semantic map rendering
        render_silhouette: Enable silhouette rendering
        render_normals: Enable normal map rendering
        spherical_harmonics_degrees: SH degrees for view-dependent effects
        use_sh: Use spherical harmonics for appearance
    """
    
    render_resolution: List[int] = field(default_factory=lambda: [1920, 1080])
    downsample_factor: float = 1.0
    rasterize_mode: str = "antialias"
    gaussian_scale_threshold: float = 0.1
    alpha_threshold: float = 0.005
    camera_model: str = "pinhole"
    camera_baseline: float = 0.0
    render_rgb: bool = True
    render_depth: bool = True
    render_semantic: bool = True
    render_silhouette: bool = True
    render_normals: bool = False
    spherical_harmonics_degrees: int = 3
    use_sh: bool = True


@dataclass
class EvalConfig:
    """Configuration for evaluation.
    
    Attributes:
        eval_every: Steps between evaluations
        save_rendered_images: Save rendered images during evaluation
        save_point_clouds: Save point cloud outputs
        compute_lpips: Compute LPIPS perceptual metric
        compute_per_class_metrics: Compute per-class metrics
        metrics: List of metrics to compute
        vis_num_samples: Number of samples for visualization
        vis_output_dir: Output directory for visualizations
    """
    
    eval_every: int = 500
    save_rendered_images: bool = True
    save_point_clouds: bool = False
    compute_lpips: bool = True
    compute_per_class_metrics: bool = True
    metrics: List[str] = field(default_factory=lambda: [
        "psnr", "ssim", "lpips", "depth_error"
    ])
    vis_num_samples: int = 10
    vis_output_dir: str = "./outputs/visualizations"


@dataclass
class LoggingConfig:
    """Configuration for logging and visualization.
    
    Attributes:
        log_dir: Directory for tensorboard logs
        experiment_name: Name for this experiment
        use_tensorboard: Enable tensorboard logging
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity (team/user)
        log_level: Logging level
        log_every: Steps between console logs
        checkpoint_dir: Directory for checkpoints
        resume_from: Checkpoint path to resume from
    """
    
    log_dir: str = "./logs"
    experiment_name: str = "semantic_4dgs"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "semantic-4dgs-traffic"
    wandb_entity: Optional[str] = None
    log_level: str = "INFO"
    log_every: int = 50
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None


@dataclass
class SystemConfig:
    """Configuration for system settings.
    
    Attributes:
        seed: Random seed for reproducibility
        num_gpus: Number of GPUs for distributed training
        distributed_backend: Backend for distributed training
        amp_enabled: Enable automatic mixed precision
        amp_dtype: AMP dtype ('float16' or 'bfloat16')
        cudnn_benchmark: Enable cudnn benchmark mode
        cudnn_deterministic: Enable deterministic cudnn operations
    """
    
    seed: int = 42
    num_gpus: int = 1
    distributed_backend: str = "nccl"
    amp_enabled: bool = True
    amp_dtype: str = "float16"
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False


@dataclass
class Config:
    """Main configuration class combining all sub-configs.
    
    This class provides a unified interface for all configuration options
    and supports loading/saving from YAML files.
    
    Example:
        >>> config = Config.from_yaml("configs/default.yaml")
        >>> config.training.learning_rate = 1e-3
        >>> config.to_yaml("configs/modified.yaml")
    """
    
    model: ModelConfig = field(default_factory=ModelConfig)
    sam2: Sam2Config = field(default_factory=Sam2Config)
    point_tracker: PointTrackerConfig = field(default_factory=PointTrackerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config: Loaded configuration object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        def dict_to_config(d: Dict[str, Any], config_cls: type) -> Any:
            """Recursively convert nested dict to dataclass."""
            if not isinstance(d, dict):
                return d
            field_types = {f.name: f.type for f in config_cls.__dataclass_fields__.values()}
            kwargs = {}
            for key, value in d.items():
                if key in field_types:
                    field_type = field_types[key]
                    if isinstance(value, dict) and not isinstance(field_type, type):
                        kwargs[key] = dict_to_config(value, field_type)
                    else:
                        kwargs[key] = value
            return config_cls(**kwargs)
        
        return dict_to_config(config_dict, cls)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        def config_to_dict(obj: Any) -> Dict[str, Any]:
            if isinstance(obj, dict):
                return {k: config_to_dict(v) for k, v in obj.items()}
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: config_to_dict(getattr(obj, k)) 
                       for k in obj.__dataclass_fields__}
            elif isinstance(obj, list):
                return [config_to_dict(item) for item in obj]
            else:
                return obj
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_to_dict(self), f, default_flow_style=False)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with dictionary of changes.
        
        Args:
            updates: Dictionary of configuration updates
        """
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(value, dict) and hasattr(getattr(self, key), '__dataclass_fields__'):
                    nested_obj = getattr(self, key)
                    for k, v in value.items():
                        if hasattr(nested_obj, k):
                            setattr(nested_obj, k, v)
                else:
                    setattr(self, key, value)


# Default config instance
default_config = Config()


if __name__ == "__main__":
    # Test configuration loading/saving
    import tempfile
    import os
    
    config = Config()
    
    # Save to temporary file
    temp_path = os.path.join(tempfile.gettempdir(), "test_config.yaml")
    config.to_yaml(temp_path)
    print(f"Config saved to {temp_path}")
    
    # Load from file
    loaded_config = Config.from_yaml(temp_path)
    print(f"Config loaded: {loaded_config.model.init_num_gaussians} initial Gaussians")
    
    # Clean up
    os.remove(temp_path)

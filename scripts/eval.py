"""
Enhanced Evaluation Script for Semantic 4D Gaussian Splatting.

Features:
- Complete evaluation pipeline (PSNR/SSIM/LPIPS + trajectory accuracy + segmentation IoU)
- Automatic baseline comparison table generation
- Multi-metric evaluation
- Results export to multiple formats
- Visualization of failure cases

Usage:
    # Basic evaluation
    python scripts/eval.py --config configs/eval.yaml --checkpoint checkpoint.pt
    
    # Compare with baseline
    python scripts/eval.py --config configs/eval.yaml --checkpoint ours.pt --baseline baseline.pt
    
    # Full evaluation with all metrics
    python scripts/eval.py --config configs/eval.yaml --checkpoint checkpoint.pt --full_eval

@author Semantic 4DGS Team
@version 1.0.0
"""

import torch
import torch.nn as nn
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import sys
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.default_config import Config
from models import GaussianField, DynamicField, JointRenderer
from models import PointTracker, TAPIRTracker, CoTracker
from datasets import WaymoDataset, nuScenesDataset, KITTI360Dataset, create_dataloader
from utils.metrics import (
    MetricsCalculator, compute_psnr, compute_ssim, compute_lpips,
    compute_segmentation_iou, compute_trajectory_metrics, compute_depth_metrics
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Semantic 4DGS")
    
    # Config and checkpoint
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint")
    
    # Baseline comparison
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline checkpoint for comparison")
    parser.add_argument("--baselines", nargs='+', default=[],
                        help="Multiple baseline checkpoints")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Output directory")
    parser.add_argument("--save_images", action="store_true",
                        help="Save rendered images")
    parser.add_argument("--save_video", action="store_true",
                        help="Save rendered videos")
    parser.add_argument("--export_csv", action="store_true",
                        help="Export results to CSV")
    parser.add_argument("--export_markdown", action="store_true",
                        help="Export results to Markdown table")
    
    # Device
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    
    # Evaluation options
    parser.add_argument("--full_eval", action="store_true",
                        help="Run full evaluation (all metrics)")
    parser.add_argument("--eval_trajectory", action="store_true",
                        help="Evaluate trajectory accuracy")
    parser.add_argument("--eval_segmentation", action="store_true",
                        help="Evaluate semantic segmentation")
    parser.add_argument("--eval_depth", action="store_true",
                        help="Evaluate depth estimation")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of evaluation samples")
    
    # Metrics
    parser.add_argument("--compute_lpips", action="store_true", default=True,
                        help="Compute LPIPS metric")
    parser.add_argument("--lpips_net", type=str, default="vgg",
                        choices=["vgg", "alex"],
                        help="LPIPS network architecture")
    
    # Visualization
    parser.add_argument("--vis_num_samples", type=int, default=4,
                        help="Number of samples to visualize")
    parser.add_argument("--vis_error_maps", action="store_true",
                        help="Visualize error maps")
    parser.add_argument("--vis_failures_only", action="store_true",
                        help="Only visualize failure cases")
    
    # Trajectory tracker
    parser.add_argument("--tracker_type", type=str, default="tapir",
                        choices=["tapir", "cotracker"],
                        help="External trajectory tracker")
    
    return parser.parse_args()


class EvaluationPipeline:
    """Complete evaluation pipeline for Semantic 4DGS."""
    
    def __init__(
        self,
        config: Config,
        device: torch.device,
        args,
    ) -> None:
        """Initialize evaluation pipeline.
        
        Args:
            config: Configuration object
            device: Compute device
            args: Command line arguments
        """
        self.config = config
        self.device = device
        self.args = args
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator(
            compute_lpips=args.compute_lpips,
            compute_depth=True,
        )
        
        # Results storage
        self.results = {}
        self.per_frame_metrics = []
        
        # External trackers
        self.tracker = None
        
    def setup_model(self, checkpoint_path: str) -> nn.Module:
        """Setup and load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Loaded model
        """
        model = GaussianField(
            num_gaussians=self.config.model.init_num_gaussians,
            feature_dim=self.config.model.feature_dim,
            semantic_feature_dim=self.config.model.semantic_feature_dim,
            device=self.device,
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def setup_tracker(self):
        """Setup external trajectory tracker."""
        if self.args.tracker_type == "tapir":
            self.tracker = TAPIRTracker(device=self.device)
        elif self.args.tracker_type == "cotracker":
            self.tracker = CoTracker(device=self.device)
    
    def evaluate_image_metrics(
        self,
        predictions: Dict,
        targets: Dict,
    ) -> Dict[str, float]:
        """Evaluate image quality metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # PSNR
        metrics['psnr'] = compute_psnr(
            predictions['rgb'],
            targets['rgb'],
        )
        
        # SSIM
        metrics['ssim'] = compute_ssim(
            predictions['rgb'],
            targets['rgb'],
        )
        
        # LPIPS
        if self.args.compute_lpips:
            metrics['lpips'] = compute_lpips(
                predictions['rgb'],
                targets['rgb'],
                net=self.metrics_calc.lpips_net,
            )
        
        return metrics
    
    def evaluate_segmentation_metrics(
        self,
        predictions: Dict,
        targets: Dict,
    ) -> Dict[str, Any]:
        """Evaluate semantic segmentation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            
        Returns:
            Dictionary of segmentation metrics
        """
        if 'semantic' not in predictions or 'semantic' not in targets:
            return {'mIoU': 0.0, 'per_class_iou': {}}
        
        pred_sem = predictions['semantic']
        target_sem = targets['semantic']
        
        # Compute IoU
        iou_dict = compute_segmentation_iou(
            pred_sem,
            target_sem,
            num_classes=self.config.dataset.num_classes,
        )
        
        # Per-class IoU
        class_names = self.config.dataset.class_names
        per_class_iou = {
            class_names.get(cls_id, f"class_{cls_id}"): iou
            for cls_id, iou in iou_dict.items()
        }
        
        metrics = {
            'mIoU': np.mean(list(iou_dict.values())),
            'per_class_iou': per_class_iou,
        }
        
        return metrics
    
    def evaluate_depth_metrics(
        self,
        predictions: Dict,
        targets: Dict,
    ) -> Dict[str, float]:
        """Evaluate depth estimation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            
        Returns:
            Dictionary of depth metrics
        """
        if 'depth' not in predictions:
            return {'abs_rel': 0.0, 'rmse': 0.0}
        
        pred_depth = predictions['depth']
        
        # Use LiDAR depth if available
        if 'points' in targets and len(targets['points']) > 0:
            target_depth = targets['depth']
        else:
            target_depth = targets.get('depth', torch.zeros_like(pred_depth))
        
        metrics = compute_depth_metrics(pred_depth, target_depth)
        
        return metrics
    
    def evaluate_trajectory_metrics(
        self,
        model: nn.Module,
        batch: Dict,
    ) -> Dict[str, float]:
        """Evaluate trajectory accuracy metrics.
        
        Args:
            model: Model to evaluate
            batch: Input batch
            
        Returns:
            Dictionary of trajectory metrics
        """
        if not self.args.eval_trajectory:
            return {}
        
        if self.tracker is None:
            self.setup_tracker()
        
        metrics = {}
        
        # Get 4DGS trajectories
        if hasattr(model, 'field') and hasattr(model.field, 'get_trajectories'):
            gauss_trajectories = model.field.get_trajectories(batch)
        else:
            return metrics
        
        # Get tracker trajectories
        rgb_sequence = batch['rgb']  # [T, 3, H, W]
        
        try:
            track_trajectories = self.tracker.track(rgb_sequence)
            
            # Compute trajectory metrics
            traj_metrics = compute_trajectory_metrics(
                gauss_trajectories,
                track_trajectories,
            )
            
            metrics.update(traj_metrics)
        except Exception as e:
            print(f"Trajectory tracking failed: {e}")
        
        return metrics
    
    def evaluate_batch(
        self,
        model: nn.Module,
        batch: Dict,
        batch_idx: int,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Evaluate a single batch.
        
        Args:
            model: Model to evaluate
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            (frame_metrics, visualizations)
        """
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        
        with torch.no_grad():
            predictions = model(batch)
        
        # Image metrics
        image_metrics = self.evaluate_image_metrics(predictions, batch)
        
        # Segmentation metrics
        seg_metrics = self.evaluate_segmentation_metrics(predictions, batch)
        
        # Depth metrics
        depth_metrics = self.evaluate_depth_metrics(predictions, batch)
        
        # Trajectory metrics
        traj_metrics = self.evaluate_trajectory_metrics(model, batch)
        
        # Combine metrics
        frame_metrics = {
            **image_metrics,
            **{f'seg_{k}': v for k, v in seg_metrics.items() if k != 'per_class_iou'},
            **{f'depth_{k}': v for k, v in depth_metrics.items()},
            **traj_metrics,
        }
        
        # Per-class IoU (flatten)
        if 'per_class_iou' in seg_metrics:
            for cls_name, iou in seg_metrics['per_class_iou'].items():
                frame_metrics[f'iou_{cls_name}'] = iou
        
        # Visualizations
        visualizations = {}
        
        if self.args.save_images:
            visualizations['rgb_gt'] = batch['rgb'][0]
            visualizations['rgb_pred'] = predictions['rgb'][0]
            
            if 'depth' in predictions:
                visualizations['depth'] = predictions['depth'][0]
            
            if 'semantic' in predictions:
                visualizations['semantic'] = predictions['semantic'][0]
        
        return frame_metrics, visualizations
    
    def run_evaluation(
        self,
        dataloader,
        checkpoint_path: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Run complete evaluation.
        
        Args:
            dataloader: Data loader
            checkpoint_path: Model checkpoint
            output_dir: Output directory
            
        Returns:
            Evaluation results
        """
        # Load model
        print(f"Loading checkpoint: {checkpoint_path}")
        model = self.setup_model(checkpoint_path)
        
        # Evaluate
        all_metrics = []
        all_visualizations = []
        
        num_samples = self.args.num_samples or len(dataloader)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if batch_idx >= num_samples:
                break
            
            frame_metrics, visualizations = self.evaluate_batch(
                model, batch, batch_idx
            )
            
            all_metrics.append(frame_metrics)
            all_visualizations.append(visualizations)
            
            self.per_frame_metrics.append(frame_metrics)
        
        # Compute mean metrics
        mean_metrics = self.compute_mean_metrics(all_metrics)
        
        # Save visualizations
        if self.args.save_images:
            self.save_visualizations(all_visualizations, output_dir)
        
        results = {
            'checkpoint': checkpoint_path,
            'num_samples': len(all_metrics),
            'mean_metrics': mean_metrics,
            'per_frame': all_metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        return results
    
    def compute_mean_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Compute mean of all metrics.
        
        Args:
            all_metrics: List of per-frame metrics
            
        Returns:
            Mean metrics dictionary
        """
        if not all_metrics:
            return {}
        
        # Scalar metrics
        scalar_keys = set()
        for m in all_metrics:
            for k, v in m.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    scalar_keys.add(k)
        
        mean_metrics = {}
        for key in scalar_keys:
            values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
            if values:
                mean_metrics[key] = np.mean(values)
                mean_metrics[f'{key}_std'] = np.std(values)
        
        # Per-class IoU aggregation
        iou_keys = [k for k in mean_metrics.keys() if k.startswith('iou_')]
        
        return mean_metrics


class BaselineComparison:
    """Compare results with baselines."""
    
    def __init__(
        self,
        output_dir: Path,
        config: Config,
        device: torch.device,
    ) -> None:
        """Initialize baseline comparison.
        
        Args:
            output_dir: Output directory
            config: Configuration
            device: Compute device
        """
        self.output_dir = output_dir
        self.config = config
        self.device = device
        self.results = {}
    
    def add_result(self, name: str, result: Dict):
        """Add evaluation result.
        
        Args:
            name: Result name
            result: Evaluation results
        """
        self.results[name] = result
    
    def generate_comparison_table(self) -> str:
        """Generate comparison table.
        
        Returns:
            Markdown table string
        """
        if not self.results:
            return "No results to compare."
        
        # Extract metric names
        metric_names = ['psnr', 'ssim', 'lpips', 'seg_mIoU', 'depth_abs_rel']
        
        # Build table
        lines = []
        lines.append("# Evaluation Results Comparison")
        lines.append("")
        lines.append("| Method | " + " | ".join(m.upper() for m in metric_names) + " |")
        lines.append("|" + "|".join(["---"] * (len(metric_names) + 1)) + "|")
        
        for name, result in self.results.items():
            metrics = result.get('mean_metrics', {})
            
            row_values = []
            for metric in metric_names:
                value = metrics.get(metric, 0.0)
                row_values.append(f"{value:.4f}" if isinstance(value, (int, float)) else "N/A")
            
            lines.append(f"| {name} | " + " | ".join(row_values) + " |")
        
        return "\n".join(lines)
    
    def generate_detailed_report(self) -> str:
        """Generate detailed comparison report.
        
        Returns:
            Markdown report string
        """
        report = []
        report.append("# Detailed Evaluation Report")
        report.append("")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        report.append("## Summary")
        report.append("")
        report.append(self.generate_comparison_table())
        report.append("")
        
        # Per-metric comparison
        report.append("## Per-Metric Analysis")
        report.append("")
        
        for name, result in self.results.items():
            report.append(f"### {name}")
            report.append("")
            
            metrics = result.get('mean_metrics', {})
            
            for key, value in sorted(metrics.items()):
                if isinstance(value, (int, float)) and not key.endswith('_std'):
                    std_key = f'{key}_std'
                    std_value = metrics.get(std_key, 0.0)
                    
                    if std_value > 0:
                        report.append(f"- {key}: {value:.4f} ± {std_value:.4f}")
                    else:
                        report.append(f"- {key}: {value:.4f}")
            
            report.append("")
        
        # Improvement analysis
        if len(self.results) > 1:
            report.append("## Improvement Analysis")
            report.append("")
            
            baseline_name = list(self.results.keys())[0]
            baseline_metrics = self.results[baseline_name].get('mean_metrics', {})
            
            for name, result in list(self.results.items())[1:]:
                report.append(f"### vs {baseline_name}")
                report.append("")
                
                metrics = result.get('mean_metrics', {})
                
                improvements = []
                for key in ['psnr', 'ssim', 'seg_mIoU']:
                    if key in baseline_metrics and key in metrics:
                        delta = metrics[key] - baseline_metrics[key]
                        
                        if key == 'lpips' or key == 'depth_abs_rel':
                            # Lower is better
                            if delta < 0:
                                improvements.append(f"- {key}: **{delta:.4f}** (improved)")
                            else:
                                improvements.append(f"- {key}: {delta:.4f}")
                        else:
                            # Higher is better
                            if delta > 0:
                                improvements.append(f"- {key}: **+{delta:.4f}** (improved)")
                            else:
                                improvements.append(f"- {key}: {delta:.4f}")
                
                report.extend(improvements)
                report.append("")
        
        return "\n".join(report)
    
    def export_csv(self, output_path: Path):
        """Export results to CSV.
        
        Args:
            output_path: Output CSV path
        """
        if not self.results:
            return
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            metric_names = list(self.results.values())[0].get('mean_metrics', {}).keys()
            writer.writerow(['Method'] + list(metric_names))
            
            # Data rows
            for name, result in self.results.items():
                metrics = result.get('mean_metrics', {})
                writer.writerow([name] + [metrics.get(k, '') for k in metric_names])
    
    def export_json(self, output_path: Path):
        """Export results to JSON.
        
        Args:
            output_path: Output JSON path
        """
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(output_path, 'w') as f:
            json.dump(convert(self.results), f, indent=2)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(config, device, args)
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset_name=config.dataset.dataset_name,
        root=config.dataset.data_root,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    
    # Evaluate main checkpoint
    main_results = pipeline.run_evaluation(
        dataloader=dataloader,
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
    )
    
    # Baseline comparison
    comparison = BaselineComparison(output_dir, config, device)
    comparison.add_result("Ours", main_results)
    
    # Evaluate baselines
    all_baselines = args.baselines + ([args.baseline] if args.baseline else [])
    
    for baseline_path in all_baselines:
        if baseline_path and Path(baseline_path).exists():
            baseline_name = Path(baseline_path).stem
            
            baseline_pipeline = EvaluationPipeline(config, device, args)
            baseline_results = baseline_pipeline.run_evaluation(
                dataloader=dataloader,
                checkpoint_path=baseline_path,
                output_dir=output_dir / baseline_name,
            )
            
            comparison.add_result(baseline_name, baseline_results)
    
    # Generate reports
    report = comparison.generate_detailed_report()
    
    # Save reports
    report_path = output_dir / "report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    
    # Export CSV
    if args.export_csv:
        csv_path = output_dir / "results.csv"
        comparison.export_csv(csv_path)
        print(f"CSV saved to {csv_path}")
    
    # Export JSON
    json_path = output_dir / "results.json"
    comparison.export_json(json_path)
    print(f"JSON saved to {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Results Summary")
    print("="*60)
    
    for name, result in comparison.results.items():
        metrics = result.get('mean_metrics', {})
        print(f"\n{name}:")
        for key in ['psnr', 'ssim', 'lpips', 'seg_mIoU']:
            if key in metrics:
                print(f"  {key.upper()}: {metrics[key]:.4f}")
    
    print("\n" + "="*60)
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

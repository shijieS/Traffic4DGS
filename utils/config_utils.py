"""
Configuration Utilities for Semantic 4D Gaussian Splatting.

Features:
- Configuration hot-reloading
- Memory optimization utilities
- Inference mode optimizations
- Mixed precision utilities

@author Semantic 4DGS Team
@version 1.0.0
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json
import yaml
from datetime import datetime
import threading


class ConfigHotReloader:
    """Configuration hot-reloading support.
    
    Monitors config file changes and updates configuration
    in real-time during training.
    """
    
    def __init__(
        self,
        config_path: str,
        on_reload: Optional[Callable] = None,
        poll_interval: float = 5.0,
    ) -> None:
        """Initialize config hot reloader.
        
        Args:
            config_path: Path to config file
            on_reload: Callback function on config reload
            poll_interval: Polling interval in seconds
        """
        self.config_path = Path(config_path)
        self.on_reload = on_reload
        self.poll_interval = poll_interval
        
        self._last_mtime = 0.0
        self._current_config = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Initial load
        self.reload()
    
    def reload(self) -> Dict[str, Any]:
        """Reload configuration from file.
        
        Returns:
            Loaded configuration dictionary
        """
        if not self.config_path.exists():
            return {}
        
        try:
            current_mtime = self.config_path.stat().st_mtime
            
            if current_mtime != self._last_mtime:
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix in ['.yaml', '.yml']:
                        new_config = yaml.safe_load(f)
                    elif self.config_path.suffix == '.json':
                        new_config = json.load(f)
                    else:
                        return self._current_config or {}
                
                with self._lock:
                    self._current_config = new_config
                    self._last_mtime = current_mtime
                
                if self.on_reload:
                    self.on_reload(new_config)
                
                return new_config
            
            return self._current_config
            
        except Exception as e:
            print(f"Config reload error: {e}")
            return self._current_config or {}
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration.
        
        Returns:
            Configuration dictionary
        """
        with self._lock:
            return self._current_config or {}
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        def monitor():
            while not self._stop_event.is_set():
                self.reload()
                self._stop_event.wait(self.poll_interval)
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._stop_event.set()
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=1.0)


class MemoryOptimizer:
    """Memory optimization utilities for large scene training.
    
    Features:
    - Gradient checkpointing
    - Mixed precision training
    - Memory-efficient attention
    - Activation caching
    """
    
    @staticmethod
    def set_memory_efficient_mode(model: nn.Module):
        """Enable memory-efficient mode for model.
        
        Args:
            model: PyTorch model
        """
        # Enable gradient checkpointing for supported modules
        for module_name, module in model.named_modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                # Custom handling for specific modules
                pass
        
        # Set memory allocator settings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def apply_gradient_checkpointing(
        model: nn.Module,
        checkpoint_modules: list = None,
    ):
        """Apply gradient checkpointing to model.
        
        Args:
            model: PyTorch model
            checkpoint_modules: List of module types to checkpoint
        """
        if checkpoint_modules is None:
            checkpoint_modules = [
                nn.MultiheadAttention,
                nn.TransformerEncoderLayer,
                nn.TransformerDecoderLayer,
            ]
        
        for module_name, module in model.named_modules():
            for module_type in checkpoint_modules:
                if isinstance(module, module_type):
                    if hasattr(module, 'gradient_checkpointing_enable'):
                        module.gradient_checkpointing_enable()
    
    @staticmethod
    def get_memory_stats(device: torch.device = None) -> Dict[str, float]:
        """Get current memory statistics.
        
        Args:
            device: Compute device
            
        Returns:
            Dictionary of memory statistics
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        stats = {}
        
        if device.type == "cuda":
            stats['allocated_gb'] = torch.cuda.memory_allocated(device) / 1e9
            stats['reserved_gb'] = torch.cuda.memory_reserved(device) / 1e9
            stats['max_allocated_gb'] = torch.cuda.max_memory_allocated(device) / 1e9
            
            try:
                stats['allocated_percent'] = (
                    torch.cuda.memory_allocated(device) / 
                    torch.cuda.get_device_properties(device).total_memory
                ) * 100
            except:
                pass
        
        return stats
    
    @staticmethod
    def clear_memory():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def estimate_model_memory(model: nn.Module) -> Dict[str, float]:
        """Estimate model memory usage.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary of memory estimates
        """
        param_memory = 0.0
        buffer_memory = 0.0
        
        for param in model.parameters():
            param_memory += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_memory += buffer.nelement() * buffer.element_size()
        
        return {
            'parameters_mb': param_memory / 1e6,
            'buffers_mb': buffer_memory / 1e6,
            'total_mb': (param_memory + buffer_memory) / 1e6,
        }


class InferenceOptimizer:
    """Inference mode optimizations.
    
    Features:
    - torch.inference_mode() context
    - torch.no_grad() with better memory handling
    - JIT compilation
    - Model export
    """
    
    @staticmethod
    def optimize_for_inference(
        model: nn.Module,
        use_torch_compile: bool = False,
    ) -> nn.Module:
        """Optimize model for inference.
        
        Args:
            model: PyTorch model
            use_torch_compile: Use torch.compile()
            
        Returns:
            Optimized model
        """
        model.eval()
        
        # Apply torch.compile if available and requested
        if use_torch_compile:
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                print(f"torch.compile not available: {e}")
        
        return model
    
    @staticmethod
    @torch.inference_mode()
    def run_inference(model: nn.Module, batch: Dict) -> Dict:
        """Run inference with inference_mode.
        
        Args:
            model: PyTorch model
            batch: Input batch
            
        Returns:
            Model predictions
        """
        return model(batch)
    
    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        output_path: str,
        sample_input: Dict,
        input_names: list = None,
        output_names: list = None,
        dynamic_axes: Dict = None,
    ):
        """Export model to ONNX format.
        
        Args:
            model: PyTorch model
            output_path: Output ONNX file path
            sample_input: Sample input for tracing
            input_names: Input tensor names
            output_names: Output tensor names
            dynamic_axes: Dynamic axis specifications
        """
        model.eval()
        
        if input_names is None:
            input_names = ['rgb', 'intrinsics', 'extrinsics']
        if output_names is None:
            output_names = ['rgb', 'depth', 'semantic']
        
        torch.onnx.export(
            model,
            (sample_input,),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )
    
    @staticmethod
    def quantize_model(
        model: nn.Module,
        quantization_type: str = "dynamic",
    ) -> nn.Module:
        """Quantize model for faster inference.
        
        Args:
            model: PyTorch model
            quantization_type: 'dynamic', 'static', or 'aware'
            
        Returns:
            Quantized model
        """
        if quantization_type == "dynamic":
            quantized = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
        elif quantization_type == "static":
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # Note: Would need calibration data for actual quantization
            quantized = torch.quantization.convert(model, inplace=True)
        else:
            quantized = model
        
        return quantized


class MixedPrecisionManager:
    """Mixed precision training utilities."""
    
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        loss_scale: str = "dynamic",
    ) -> None:
        """Initialize mixed precision manager.
        
        Args:
            enabled: Enable mixed precision
            dtype: Target dtype
            loss_scale: 'dynamic', 'fixed', or 'none'
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        self.loss_scale = loss_scale
        
        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=65536.0,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
                enabled=(loss_scale != "none"),
            )
    
    @property
    def autocast(self):
        """Get autocast context manager."""
        if self.enabled:
            return torch.cuda.amp.autocast(dtype=self.dtype)
        return torch.cuda.amp.autocast(enabled=False)
    
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision.
        
        Args:
            loss: Loss tensor
            
        Returns:
            Scaled loss
        """
        if self.enabled and self.scaler.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer: torch.optim.Optimizer):
        """Step optimizer with gradient scaling.
        
        Args:
            optimizer: PyTorch optimizer
        """
        if self.enabled and self.scaler.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients for gradient clipping.
        
        Args:
            optimizer: PyTorch optimizer
        """
        if self.enabled and self.scaler.enabled:
            self.scaler.unscale_(optimizer)


def apply_memory_optimizations(
    model: nn.Module,
    config: Dict[str, Any],
) -> nn.Module:
    """Apply memory optimizations based on config.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        
    Returns:
        Optimized model
    """
    # Gradient checkpointing
    if config.get('training', {}).get('gradient_checkpointing', False):
        MemoryOptimizer.apply_gradient_checkpointing(model)
    
    # Memory-efficient mode
    if config.get('training', {}).get('memory_efficient', False):
        MemoryOptimizer.set_memory_efficient_mode(model)
    
    return model


def create_optimized_inference_model(
    model: nn.Module,
    config: Dict[str, Any],
) -> nn.Module:
    """Create optimized inference model.
    
    Args:
        model: Trained model
        config: Configuration
        
    Returns:
        Optimized model for inference
    """
    # Optimize for inference
    use_compile = config.get('inference', {}).get('use_torch_compile', False)
    model = InferenceOptimizer.optimize_for_inference(model, use_compile)
    
    # Apply quantization if configured
    quant_type = config.get('inference', {}).get('quantization', None)
    if quant_type:
        model = InferenceOptimizer.quantize_model(model, quant_type)
    
    return model


# Configuration merging utilities
def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config_with_overrides(
    config_path: str,
    override_path: Optional[str] = None,
    cli_overrides: Dict = None,
) -> Dict:
    """Load configuration with overrides.
    
    Args:
        config_path: Base config path
        override_path: Override config path
        cli_overrides: CLI override dictionary
        
    Returns:
        Final configuration
    """
    # Load base config
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    # Apply override config
    if override_path and Path(override_path).exists():
        with open(override_path, 'r') as f:
            if override_path.endswith('.yaml'):
                overrides = yaml.safe_load(f)
            else:
                overrides = json.load(f)
        config = merge_configs(config, overrides)
    
    # Apply CLI overrides
    if cli_overrides:
        config = merge_configs(config, cli_overrides)
    
    return config

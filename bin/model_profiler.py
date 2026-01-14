"""
Model Profiler for FastSSL
Profiles PyTorch models to get statistics on parameters, model size, timing, and memory usage.

Usage:
    # Basic usage with gin bindings
    python model_profiler.py --gin build_model.model_fn=@diffmae_base --img_size 224 
    
    # With gin config file
    python model_profiler.py --cfgs configs/diffmae_ffcv.gin 
    
    # Profile specific model with custom arguments
    python model_profiler.py --gin build_model.model_fn=@simclr_base --gin build_model.embed_dim=2048
    
    # Profile on CPU instead of CUDA
    python model_profiler.py --gin build_model.model_fn=@diffmae_base --device cpu
    
    # Use mixed precision profiling
    python model_profiler.py --gin build_model.model_fn=@diffmae_base --use_amp
    
    # Skip FLOPs estimation (faster)
    python model_profiler.py --gin build_model.model_fn=@diffmae_base --no_flops
    
    # Save results to file
    python model_profiler.py --gin build_model.model_fn=@diffmae_base --save profile.json
    python model_profiler.py --gin build_model.model_fn=@diffmae_base --save profile.txt --save_format txt
    
    # Use PyTorch profiler for detailed analysis
    python model_profiler.py --gin build_model.model_fn=@diffmae_base --torch_profiler
    python model_profiler.py --gin build_model.model_fn=@diffmae_base --torch_profiler --profiler_trace trace.json
    
    # Profile training step (forward + backward + optimizer)
    python model_profiler.py --gin build_model.model_fn=@diffmae_base --train_step
    python model_profiler.py --gin build_model.model_fn=@diffmae_base --train_step --use_amp

Optional dependencies for FLOPs estimation (install one):
    pip install thop
    # or
    pip install ptflops
    # PyTorch 2.0+ includes FlopCounterMode, so no extra dependencies needed
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import gin
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# Try to import FlopCounterMode (available in PyTorch 2.0+)
try:
    from torch.utils.flop_counter import FlopCounterMode
    FLOP_COUNTER_AVAILABLE = True
except ImportError:
    try:
        # Alternative import location in some versions
        from torch.utils.flop_counter import FlopCounterMode
        FLOP_COUNTER_AVAILABLE = True
    except ImportError:
        FLOP_COUNTER_AVAILABLE = False

from layers import build_model
from util.helper import aug_parse


def collect_machine_info() -> dict:
    """Collect machine and environment information.
    
    Returns:
        Dictionary with machine information
    """
    info = {
        'pytorch_version': torch.__version__,
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
    }
    
    # CUDA information
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info['gpu_count'] = torch.cuda.device_count()
        
        # GPU details
        gpus = []
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024 ** 3),
                'compute_capability': f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}",
            }
            gpus.append(gpu_info)
        info['gpus'] = gpus
        
        # Current GPU
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            info['current_gpu'] = current_device
            info['current_gpu_name'] = torch.cuda.get_device_name(current_device)
    else:
        info['cuda_available'] = False
        info['gpu_count'] = 0
    
    # Additional system info
    try:
        import psutil
        info['memory_total_gb'] = psutil.virtual_memory().total / (1024 ** 3)
        info['memory_available_gb'] = psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass
    
    return info


def count_parameters(model: nn.Module) -> dict:
    """Count total, trainable, and non-trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
    }


def get_model_size(model: nn.Module, precision: int = 32) -> dict:
    """Calculate model size in MB.
    
    Args:
        model: PyTorch model
        precision: Bit precision (32 for float32, 16 for float16, etc.)
    
    Returns:
        Dictionary with model size in MB
    """
    param_size = sum(p.numel() * precision / 8 for p in model.parameters())
    buffer_size = sum(b.numel() * precision / 8 for b in model.buffers())
    total_size = param_size + buffer_size
    
    return {
        'parameters_mb': param_size / (1024 ** 2),
        'buffers_mb': buffer_size / (1024 ** 2),
        'total_mb': total_size / (1024 ** 2),
    }


def profile_forward_pass(
    model: nn.Module,
    dummy_input: torch.Tensor,
    device: torch.device,
    num_warmup: int = 10,
    num_iterations: int = 100,
    use_amp: bool = False,
    **kwargs
) -> dict:
    """Profile forward pass timing and throughput.
    
    Args:
        model: PyTorch model
        dummy_input: Input tensor
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations for timing
        use_amp: Whether to use automatic mixed precision
        **kwargs: Additional arguments to pass to model.forward()
    
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    model.to(device)
    dummy_input = dummy_input.to(device)
    
    # Warmup
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                for _ in range(num_warmup):
                    _ = model(dummy_input, **kwargs)
        else:
            for _ in range(num_warmup):
                _ = model(dummy_input, **kwargs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timing
    batch_size = dummy_input.shape[0]
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input, **kwargs)
            else:
                _ = model(dummy_input, **kwargs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    times = torch.tensor(times)
    
    return {
        'mean_ms': times.mean().item() * 1000,
        'std_ms': times.std().item() * 1000,
        'min_ms': times.min().item() * 1000,
        'max_ms': times.max().item() * 1000,
        'median_ms': times.median().item() * 1000,
        'throughput_samples_per_sec': batch_size / times.mean().item(),
        'throughput_fps': batch_size / times.mean().item() if batch_size == 1 else None,
    }


def profile_train_step(
    model: nn.Module,
    dummy_input: torch.Tensor,
    device: torch.device,
    num_warmup: int = 10,
    num_iterations: int = 100,
    use_amp: bool = False,
    optimizer_fn=None,
    **kwargs
) -> dict:
    """Profile training step (forward + backward + optimizer step).
    
    Args:
        model: PyTorch model
        dummy_input: Input tensor
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations for timing
        use_amp: Whether to use automatic mixed precision
        optimizer_fn: Function to create optimizer, or None to use default
        **kwargs: Additional arguments to pass to model.forward()
    
    Returns:
        Dictionary with training step timing statistics
    """
    model.train()
    model.to(device)
    dummy_input = dummy_input.to(device)
    
    # Create optimizer if not provided
    if optimizer_fn is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optimizer_fn(model.parameters())
    
    # Create scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    # Create dummy target (if needed by model)
    # Most models return (loss, log) from forward()
    dummy_target = None
    if 'targets' in kwargs:
        dummy_target = kwargs['targets']
    elif 'epoch' in kwargs:
        # Some models use epoch parameter
        pass
    
    # Warmup
    for _ in range(num_warmup):
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                loss, log = model(dummy_input, **kwargs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, log = model(dummy_input, **kwargs)
            loss.backward()
            optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Timing
    batch_size = dummy_input.shape[0]
    times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        step_start = time.perf_counter()
        
        # Forward pass
        forward_start = time.perf_counter()
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                loss, log = model(dummy_input, **kwargs)
        else:
            loss, log = model(dummy_input, **kwargs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        forward_end = time.perf_counter()
        forward_times.append(forward_end - forward_start)
        
        # Backward pass
        backward_start = time.perf_counter()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        backward_end = time.perf_counter()
        backward_times.append(backward_end - backward_start)
        
        # Optimizer step
        opt_start = time.perf_counter()
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        opt_end = time.perf_counter()
        optimizer_times.append(opt_end - opt_start)
        
        step_end = time.perf_counter()
        times.append(step_end - step_start)
    
    times = torch.tensor(times)
    forward_times = torch.tensor(forward_times)
    backward_times = torch.tensor(backward_times)
    optimizer_times = torch.tensor(optimizer_times)
    
    return {
        'total': {
            'mean_ms': times.mean().item() * 1000,
            'std_ms': times.std().item() * 1000,
            'min_ms': times.min().item() * 1000,
            'max_ms': times.max().item() * 1000,
            'median_ms': times.median().item() * 1000,
            'throughput_samples_per_sec': batch_size / times.mean().item(),
        },
        'forward': {
            'mean_ms': forward_times.mean().item() * 1000,
            'std_ms': forward_times.std().item() * 1000,
            'mean_percent': (forward_times.mean() / times.mean() * 100).item(),
        },
        'backward': {
            'mean_ms': backward_times.mean().item() * 1000,
            'std_ms': backward_times.std().item() * 1000,
            'mean_percent': (backward_times.mean() / times.mean() * 100).item(),
        },
        'optimizer': {
            'mean_ms': optimizer_times.mean().item() * 1000,
            'std_ms': optimizer_times.std().item() * 1000,
            'mean_percent': (optimizer_times.mean() / times.mean() * 100).item(),
        },
    }


def profile_train_memory(
    model: nn.Module,
    dummy_input: torch.Tensor,
    device: torch.device,
    use_amp: bool = False,
    optimizer_fn=None,
    **kwargs
) -> dict:
    """Profile memory usage during training step.
    
    Args:
        model: PyTorch model
        dummy_input: Input tensor
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        optimizer_fn: Function to create optimizer, or None to use default
        **kwargs: Additional arguments to pass to model.forward()
    
    Returns:
        Dictionary with memory statistics (CUDA only)
    """
    if device.type != 'cuda':
        return {'memory_mb': None, 'memory_gb': None, 'peak_allocated_mb': None}
    
    model.train()
    model.to(device)
    dummy_input = dummy_input.to(device)
    
    # Create optimizer if not provided
    if optimizer_fn is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optimizer_fn(model.parameters())
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    optimizer.zero_grad()
    
    if use_amp and scaler is not None:
        with torch.cuda.amp.autocast():
            loss, log = model(dummy_input, **kwargs)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss, log = model(dummy_input, **kwargs)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()
    peak_allocated = torch.cuda.max_memory_reserved()
    
    return {
        'memory_mb': peak_memory / (1024 ** 2),
        'memory_gb': peak_memory / (1024 ** 3),
        'peak_allocated_mb': peak_allocated / (1024 ** 2),
        'peak_allocated_gb': peak_allocated / (1024 ** 3),
    }


def profile_memory(
    model: nn.Module,
    dummy_input: torch.Tensor,
    device: torch.device,
    use_amp: bool = False,
    **kwargs
) -> dict:
    """Profile memory usage during forward pass.
    
    Args:
        model: PyTorch model
        dummy_input: Input tensor
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        **kwargs: Additional arguments to pass to model.forward()
    
    Returns:
        Dictionary with memory statistics (CUDA only)
    """
    if device.type != 'cuda':
        return {'memory_mb': None, 'memory_gb': None}
    
    model.eval()
    model.to(device)
    dummy_input = dummy_input.to(device)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                _ = model(dummy_input, **kwargs)
        else:
            _ = model(dummy_input, **kwargs)
    
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()
    
    return {
        'memory_mb': peak_memory / (1024 ** 2),
        'memory_gb': peak_memory / (1024 ** 3),
    }


def estimate_flops(
    model: nn.Module,
    dummy_input: torch.Tensor,
    device: torch.device,
    **kwargs
) -> dict:
    """Estimate FLOPs using PyTorch's FlopCounterMode or thop/ptflops if available.
    
    Args:
        model: PyTorch model
        dummy_input: Input tensor
        device: Device to run on
        **kwargs: Additional arguments to pass to model.forward()
    
    Returns:
        Dictionary with FLOP statistics
    """
    if not FLOP_COUNTER_AVAILABLE:
        # Try thop as fallback
        try:
            from thop import profile as thop_profile
            model.eval()
            model.to(device)
            dummy_input = dummy_input.to(device)
            
            with torch.no_grad():
                flops, params = thop_profile(model, inputs=(dummy_input,), verbose=False)
            
            return {
                'total_flops': flops,
                'total_gflops': flops / 1e9,
                'per_sample_gflops': flops / dummy_input.shape[0] / 1e9,
                'method': 'thop',
            }
        except ImportError:
            # Try ptflops as another fallback
            try:
                from ptflops import get_model_complexity_info
                model.eval()
                model.to(device)
                
                # Get input shape without batch dimension for ptflops
                input_shape = tuple(dummy_input.shape[1:])
                
                with torch.no_grad():
                    flops, params = get_model_complexity_info(
                        model, input_shape, as_strings=False, print_per_layer_stat=False
                    )
                
                return {
                    'total_flops': flops * dummy_input.shape[0],  # ptflops gives per-sample
                    'total_gflops': flops * dummy_input.shape[0] / 1e9,
                    'per_sample_gflops': flops / 1e9,
                    'method': 'ptflops',
                }
            except ImportError:
                return {
                    'total_flops': None,
                    'total_gflops': None,
                    'per_sample_gflops': None,
                    'error': 'FLOP counter not available. Install thop (pip install thop) or ptflops (pip install ptflops)',
                }
    
    # Use PyTorch's FlopCounterMode
    try:
        model.eval()
        model.to(device)
        dummy_input = dummy_input.to(device)
        
        with torch.no_grad():
            flop_counter = FlopCounterMode(model, display=False)
            with flop_counter:
                _ = model(dummy_input, **kwargs)
        
        total_flops = flop_counter.get_total_flops()
        
        return {
            'total_flops': total_flops,
            'total_gflops': total_flops / 1e9,
            'per_sample_gflops': total_flops / dummy_input.shape[0] / 1e9,
            'method': 'pytorch',
        }
    except Exception as e:
        return {
            'total_flops': None,
            'total_gflops': None,
            'per_sample_gflops': None,
            'error': str(e),
        }


def save_profile_results(profile_results: dict, output_path: str, model_name: str = "Model", format: str = "json"):
    """Save profile results to a file.
    
    Args:
        profile_results: Dictionary with profile results
        output_path: Path to save the file
        model_name: Name of the model
        format: File format ('json' or 'txt')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "json":
        # Prepare JSON-serializable results
        json_results = {
            'model_name': model_name,
            'machine_info': profile_results.get('machine_info', {}),
            'parameters': profile_results.get('parameters', {}),
            'model_size': profile_results.get('model_size', {}),
            'input_shape': profile_results.get('input_shape'),
            'timing': profile_results.get('timing', {}),
            'memory': profile_results.get('memory', {}),
            'flops': profile_results.get('flops', {}),
            'torch_profiler': profile_results.get('torch_profiler', {}),
            'train_timing': profile_results.get('train_timing', {}),
            'train_memory': profile_results.get('train_memory', {}),
            'suggestions': summary_suggestions(profile_results),
        }
        
        # Convert tensors and numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return str(obj)
        
        json_results = convert_to_serializable(json_results)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✅ Profile results saved to: {output_path}")
    
    elif format.lower() == "txt":
        # Save as formatted text file
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Model Profile: {model_name}\n")
            f.write("=" * 80 + "\n")
            
            # Machine Information
            machine_info = profile_results.get('machine_info', {})
            if machine_info:
                f.write("\n🖥️  Machine Information:\n")
                f.write(f"  PyTorch:      {machine_info.get('pytorch_version', 'N/A')}\n")
                f.write(f"  Python:       {machine_info.get('python_version', 'N/A')}\n")
                f.write(f"  Platform:     {machine_info.get('platform', 'N/A')}\n")
                if machine_info.get('cuda_available'):
                    f.write(f"  CUDA:         {machine_info.get('cuda_version', 'N/A')}\n")
                    if machine_info.get('cudnn_version'):
                        f.write(f"  cuDNN:        {machine_info.get('cudnn_version', 'N/A')}\n")
                    f.write(f"  GPUs:         {machine_info.get('gpu_count', 0)}\n")
                    if machine_info.get('gpus'):
                        for gpu in machine_info.get('gpus', []):
                            current = " (current)" if gpu.get('index') == machine_info.get('current_gpu', -1) else ""
                            f.write(f"    GPU {gpu.get('index')}: {gpu.get('name', 'N/A')}{current}\n")
                            f.write(f"      Memory:    {gpu.get('memory_total_gb', 0):.2f} GB\n")
                            f.write(f"      Compute:   {gpu.get('compute_capability', 'N/A')}\n")
                else:
                    f.write(f"  CUDA:         Not available\n")
                if machine_info.get('memory_total_gb'):
                    f.write(f"  System RAM:   {machine_info.get('memory_total_gb', 0):.2f} GB\n")
                    f.write(f"  Available:    {machine_info.get('memory_available_gb', 0):.2f} GB\n")
            
            # Parameters
            params = profile_results.get('parameters', {})
            f.write("\n📊 Parameters:\n")
            f.write(f"  Total:        {params.get('total', 0):,}\n")
            f.write(f"  Trainable:    {params.get('trainable', 0):,}\n")
            f.write(f"  Non-trainable: {params.get('non_trainable', 0):,}\n")
            
            # Model Size
            size = profile_results.get('model_size', {})
            f.write("\n💾 Model Size:\n")
            f.write(f"  Parameters:   {size.get('parameters_mb', 0):.2f} MB\n")
            f.write(f"  Buffers:      {size.get('buffers_mb', 0):.2f} MB\n")
            f.write(f"  Total:        {size.get('total_mb', 0):.2f} MB\n")
            
            # Timing
            timing = profile_results.get('timing', {})
            if timing:
                f.write("\n⏱️  Forward Pass Timing:\n")
                f.write(f"  Mean:         {timing.get('mean_ms', 0):.3f} ms\n")
                f.write(f"  Std:          {timing.get('std_ms', 0):.3f} ms\n")
                f.write(f"  Min:          {timing.get('min_ms', 0):.3f} ms\n")
                f.write(f"  Max:          {timing.get('max_ms', 0):.3f} ms\n")
                f.write(f"  Median:       {timing.get('median_ms', 0):.3f} ms\n")
                f.write(f"  Throughput:   {timing.get('throughput_samples_per_sec', 0):.2f} samples/sec\n")
                if timing.get('throughput_fps'):
                    f.write(f"  FPS:          {timing.get('throughput_fps', 0):.2f}\n")
            
            # Memory
            memory = profile_results.get('memory', {})
            if memory.get('memory_mb') is not None:
                f.write("\n🔋 Memory Usage:\n")
                f.write(f"  Peak Memory:  {memory.get('memory_mb', 0):.2f} MB ({memory.get('memory_gb', 0):.3f} GB)\n")
            
            # FLOPs
            flops = profile_results.get('flops', {})
            if flops.get('total_flops') is not None:
                f.write("\n🔢 FLOPs:\n")
                f.write(f"  Total:        {flops.get('total_flops', 0):.2e}\n")
                f.write(f"  GFLOPs:       {flops.get('total_gflops', 0):.2f}\n")
                f.write(f"  Per Sample:   {flops.get('per_sample_gflops', 0):.2f} GFLOPs\n")
            elif flops.get('error'):
                f.write(f"\n⚠️  FLOPs estimation failed: {flops.get('error')}\n")
            
            # PyTorch Profiler
            torch_profiler = profile_results.get('torch_profiler', {})
            if torch_profiler and 'error' not in torch_profiler:
                f.write("\n🔥 PyTorch Profiler:\n")
                if torch_profiler.get('cpu_time_ms') is not None:
                    f.write(f"  CPU Time:     {torch_profiler.get('cpu_time_ms', 0):.3f} ms\n")
                if torch_profiler.get('cuda_time_ms') is not None:
                    f.write(f"  CUDA Time:    {torch_profiler.get('cuda_time_ms', 0):.3f} ms\n")
                if torch_profiler.get('cuda_memory_mb') is not None:
                    f.write(f"  CUDA Memory:  {torch_profiler.get('cuda_memory_mb', 0):.2f} MB\n")
                if torch_profiler.get('trace_path'):
                    f.write(f"  Trace saved:  {torch_profiler.get('trace_path')}\n")
                top_kernels = torch_profiler.get('top_cuda_kernels', [])
                if top_kernels:
                    f.write(f"  Top CUDA Kernel: {top_kernels[0].get('key', 'N/A')}\n")
                    f.write(f"    Time: {top_kernels[0].get('time', 0)/1000:.3f} ms\n")
            elif torch_profiler.get('error'):
                f.write(f"\n⚠️  PyTorch profiler failed: {torch_profiler.get('error')}\n")
            
            # Training Step Profiling
            train_timing = profile_results.get('train_timing', {})
            if train_timing and 'error' not in train_timing:
                f.write("\n🚂 Training Step Timing:\n")
                total = train_timing.get('total', {})
                forward = train_timing.get('forward', {})
                backward = train_timing.get('backward', {})
                optimizer = train_timing.get('optimizer', {})
                
                f.write(f"  Total Step:    {total.get('mean_ms', 0):.3f} ms ± {total.get('std_ms', 0):.3f}\n")
                f.write(f"    Throughput:  {total.get('throughput_samples_per_sec', 0):.2f} samples/sec\n")
                if forward:
                    f.write(f"  Forward:       {forward.get('mean_ms', 0):.3f} ms ({forward.get('mean_percent', 0):.1f}%)\n")
                if backward:
                    f.write(f"  Backward:      {backward.get('mean_ms', 0):.3f} ms ({backward.get('mean_percent', 0):.1f}%)\n")
                if optimizer:
                    f.write(f"  Optimizer:     {optimizer.get('mean_ms', 0):.3f} ms ({optimizer.get('mean_percent', 0):.1f}%)\n")
                
                train_memory = profile_results.get('train_memory', {})
                if train_memory and 'error' not in train_memory and train_memory.get('memory_mb') is not None:
                    f.write(f"  Training Memory: {train_memory.get('memory_mb', 0):.2f} MB\n")
            elif train_timing.get('error'):
                f.write(f"\n⚠️  Training step profiling failed: {train_timing.get('error')}\n")
            
            # Input shape
            input_shape = profile_results.get('input_shape', None)
            if input_shape:
                f.write(f"\n📥 Input Shape: {input_shape}\n")
            
            # Suggestions
            suggestions = summary_suggestions(profile_results)
            if suggestions:
                f.write("\n" + "=" * 80 + "\n")
                f.write("💡 Performance Suggestions:\n")
                f.write("=" * 80 + "\n")
                for suggestion in suggestions:
                    f.write(suggestion + "\n")
                f.write("=" * 80 + "\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"\n✅ Profile results saved to: {output_path}")
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'txt'.")


def print_profile_report(profile_results: dict, model_name: str = "Model"):
    """Print a formatted profile report."""
    print("\n" + "=" * 80)
    print(f"Model Profile: {model_name}")
    print("=" * 80)
    
    # Machine Information
    machine_info = profile_results.get('machine_info', {})
    if machine_info:
        print("\n🖥️  Machine Information:")
        print(f"  PyTorch:      {machine_info.get('pytorch_version', 'N/A')}")
        print(f"  Python:       {machine_info.get('python_version', 'N/A')}")
        print(f"  Platform:     {machine_info.get('platform', 'N/A')}")
        if machine_info.get('cuda_available'):
            print(f"  CUDA:         {machine_info.get('cuda_version', 'N/A')}")
            if machine_info.get('cudnn_version'):
                print(f"  cuDNN:        {machine_info.get('cudnn_version', 'N/A')}")
            print(f"  GPUs:         {machine_info.get('gpu_count', 0)}")
            if machine_info.get('gpus'):
                for gpu in machine_info.get('gpus', []):
                    current = " (current)" if gpu.get('index') == machine_info.get('current_gpu', -1) else ""
                    print(f"    GPU {gpu.get('index')}: {gpu.get('name', 'N/A')}{current}")
                    print(f"      Memory:    {gpu.get('memory_total_gb', 0):.2f} GB")
                    print(f"      Compute:   {gpu.get('compute_capability', 'N/A')}")
        else:
            print(f"  CUDA:         Not available")
        if machine_info.get('memory_total_gb'):
            print(f"  System RAM:   {machine_info.get('memory_total_gb', 0):.2f} GB")
            print(f"  Available:   {machine_info.get('memory_available_gb', 0):.2f} GB")
    
    # Parameters
    params = profile_results.get('parameters', {})
    print("\n📊 Parameters:")
    print(f"  Total:        {params.get('total', 0):,}")
    print(f"  Trainable:    {params.get('trainable', 0):,}")
    print(f"  Non-trainable: {params.get('non_trainable', 0):,}")
    
    # Model Size
    size = profile_results.get('model_size', {})
    print("\n💾 Model Size:")
    print(f"  Parameters:   {size.get('parameters_mb', 0):.2f} MB")
    print(f"  Buffers:      {size.get('buffers_mb', 0):.2f} MB")
    print(f"  Total:        {size.get('total_mb', 0):.2f} MB")
    
    # Timing
    timing = profile_results.get('timing', {})
    if timing:
        print("\n⏱️  Forward Pass Timing:")
        print(f"  Mean:         {timing.get('mean_ms', 0):.3f} ms")
        print(f"  Std:          {timing.get('std_ms', 0):.3f} ms")
        print(f"  Min:          {timing.get('min_ms', 0):.3f} ms")
        print(f"  Max:          {timing.get('max_ms', 0):.3f} ms")
        print(f"  Median:       {timing.get('median_ms', 0):.3f} ms")
        print(f"  Throughput:   {timing.get('throughput_samples_per_sec', 0):.2f} samples/sec")
        if timing.get('throughput_fps'):
            print(f"  FPS:          {timing.get('throughput_fps', 0):.2f}")
    
    # Memory
    memory = profile_results.get('memory', {})
    if memory.get('memory_mb') is not None:
        print("\n🔋 Memory Usage:")
        print(f"  Peak Memory:  {memory.get('memory_mb', 0):.2f} MB ({memory.get('memory_gb', 0):.3f} GB)")
    
    # FLOPs
    flops = profile_results.get('flops', {})
    if flops.get('total_flops') is not None:
        print("\n🔢 FLOPs:")
        print(f"  Total:        {flops.get('total_flops', 0):.2e}")
        print(f"  GFLOPs:       {flops.get('total_gflops', 0):.2f}")
        print(f"  Per Sample:   {flops.get('per_sample_gflops', 0):.2f} GFLOPs")
    elif flops.get('error'):
        print(f"\n⚠️  FLOPs estimation failed: {flops.get('error')}")
    
    # PyTorch Profiler
    torch_profiler = profile_results.get('torch_profiler', {})
    if torch_profiler and 'error' not in torch_profiler:
        print("\n🔥 PyTorch Profiler:")
        if torch_profiler.get('cpu_time_ms') is not None:
            print(f"  CPU Time:     {torch_profiler.get('cpu_time_ms', 0):.3f} ms")
        if torch_profiler.get('cuda_time_ms') is not None:
            print(f"  CUDA Time:    {torch_profiler.get('cuda_time_ms', 0):.3f} ms")
        if torch_profiler.get('cuda_memory_mb') is not None:
            print(f"  CUDA Memory:  {torch_profiler.get('cuda_memory_mb', 0):.2f} MB")
        if torch_profiler.get('trace_path'):
            print(f"  Trace saved:  {torch_profiler.get('trace_path')}")
            print(f"                (View at chrome://tracing)")
        
        # Top CUDA kernels
        top_kernels = torch_profiler.get('top_cuda_kernels', [])
        if top_kernels:
            print(f"  Top CUDA Kernel: {top_kernels[0].get('key', 'N/A')}")
            print(f"    Time: {top_kernels[0].get('time', 0)/1000:.3f} ms")
    elif torch_profiler.get('error'):
        print(f"\n⚠️  PyTorch profiler failed: {torch_profiler.get('error')}")
    
    # Training Step Profiling
    train_timing = profile_results.get('train_timing', {})
    if train_timing and 'error' not in train_timing:
        print("\n🚂 Training Step Timing:")
        total = train_timing.get('total', {})
        forward = train_timing.get('forward', {})
        backward = train_timing.get('backward', {})
        optimizer = train_timing.get('optimizer', {})
        
        print(f"  Total Step:    {total.get('mean_ms', 0):.3f} ms ± {total.get('std_ms', 0):.3f}")
        print(f"    Min:         {total.get('min_ms', 0):.3f} ms")
        print(f"    Max:         {total.get('max_ms', 0):.3f} ms")
        print(f"    Throughput:  {total.get('throughput_samples_per_sec', 0):.2f} samples/sec")
        
        if forward:
            print(f"  Forward:       {forward.get('mean_ms', 0):.3f} ms ({forward.get('mean_percent', 0):.1f}%)")
        if backward:
            print(f"  Backward:      {backward.get('mean_ms', 0):.3f} ms ({backward.get('mean_percent', 0):.1f}%)")
        if optimizer:
            print(f"  Optimizer:     {optimizer.get('mean_ms', 0):.3f} ms ({optimizer.get('mean_percent', 0):.1f}%)")
        
        train_memory = profile_results.get('train_memory', {})
        if train_memory and 'error' not in train_memory:
            if train_memory.get('memory_mb') is not None:
                print(f"\n  Training Memory:")
                print(f"    Peak:        {train_memory.get('memory_mb', 0):.2f} MB ({train_memory.get('memory_gb', 0):.3f} GB)")
                if train_memory.get('peak_allocated_mb') is not None:
                    print(f"    Allocated:   {train_memory.get('peak_allocated_mb', 0):.2f} MB ({train_memory.get('peak_allocated_gb', 0):.3f} GB)")
    elif train_timing.get('error'):
        print(f"\n⚠️  Training step profiling failed: {train_timing.get('error')}")
    
    # Input shape
    input_shape = profile_results.get('input_shape', None)
    if input_shape:
        print(f"\n📥 Input Shape: {input_shape}")
    
    print("\n" + "=" * 80 + "\n")


def profile_with_torch_profiler(
    model: nn.Module,
    dummy_input: torch.Tensor,
    device: torch.device,
    num_warmup: int = 5,
    num_iterations: int = 10,
    use_amp: bool = False,
    trace_path: Optional[str] = None,
    **model_kwargs
) -> dict:
    """Profile model using PyTorch's built-in profiler.
    
    Args:
        model: PyTorch model
        dummy_input: Input tensor
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_iterations: Number of profiling iterations
        use_amp: Whether to use automatic mixed precision
        trace_path: Path to save profiler trace (chrome trace format)
        **model_kwargs: Additional arguments to pass to model.forward()
    
    Returns:
        Dictionary with profiler statistics
    """
    model.eval()
    model.to(device)
    dummy_input = dummy_input.to(device)
    
    # Warmup
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                for _ in range(num_warmup):
                    _ = model(dummy_input, **model_kwargs)
        else:
            for _ in range(num_warmup):
                _ = model(dummy_input, **model_kwargs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Determine activities based on device
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    
    # Create profiler schedule
    prof_schedule = schedule(
        skip_first=2,
        wait=1,
        warmup=2,
        active=num_iterations,
        repeat=1
    )
    
    results = {}
    
    # Run profiling
    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=None,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            for i in range(num_warmup + num_iterations + 3):  # +3 for skip_first + wait + repeat
                if use_amp:
                    with torch.cuda.amp.autocast():
                        with record_function("model_forward"):
                            _ = model(dummy_input, **model_kwargs)
                else:
                    with record_function("model_forward"):
                        _ = model(dummy_input, **model_kwargs)
                
                prof.step()
        
        # Export trace if requested (before leaving context)
        if trace_path:
            prof.export_chrome_trace(trace_path)
            results['trace_path'] = trace_path
        
        # Get key statistics
        key_averages = prof.key_averages()
        
        # Extract statistics
        total_cuda_time = 0.0
        total_cpu_time = 0.0
        total_cuda_mem = 0.0
        cuda_kernels = []
        
        for event in key_averages:
            # Collect stats for model_forward events
            if 'model_forward' in event.key:
                total_cpu_time += event.cpu_time_total
                # Check if CUDA attributes exist before accessing
                cuda_time = getattr(event, 'cuda_time_total', 0)
                if cuda_time > 0:
                    total_cuda_time += cuda_time
                cuda_mem = getattr(event, 'cuda_memory_usage', 0)
                if cuda_mem > 0:
                    total_cuda_mem = max(total_cuda_mem, cuda_mem)
            
            # Track CUDA kernels (check if CUDA time > 0, using getattr for safety)
            cuda_time = getattr(event, 'cuda_time_total', 0)
            event_count = getattr(event, 'count', 0)
            if cuda_time > 0 and event_count > 0:
                cuda_kernels.append({
                    'key': event.key,
                    'time': cuda_time,
                    'count': event_count,
                })
        
        # Sort CUDA kernels by time
        cuda_kernels.sort(key=lambda x: x['time'], reverse=True)
        
        results.update({
            'cpu_time_ms': total_cpu_time / 1000.0,  # Convert microseconds to ms
            'cuda_time_ms': total_cuda_time / 1000.0 if total_cuda_time > 0 else None,
            'cuda_memory_mb': total_cuda_mem / (1024 ** 2) if total_cuda_mem > 0 else None,
            'top_cuda_kernels': cuda_kernels[:10],  # Top 10 kernels
            'event_count': len(key_averages),
        })
    
    return results

def summary_suggestions(profile_results: dict) -> list:
    """Provide summary suggestions based on profile results.
    
    Args:
        profile_results: Dictionary with profile results
    
    Returns:
        List of suggestion strings
    """
    suggestions = []
    
    # Analyze model size
    params = profile_results.get('parameters', {})
    model_size = profile_results.get('model_size', {})
    total_params = params.get('total', 0)
    total_size_mb = model_size.get('total_mb', 0)
    
    if total_params > 500_000_000:  # > 500M params
        suggestions.append("⚠️  Model is very large (>500M params). Consider using a smaller model variant or model compression techniques.")
    elif total_params > 100_000_000:  # > 100M params
        suggestions.append("💡 Model is large (>100M params). Consider gradient checkpointing to reduce memory usage.")
    
    if total_size_mb > 2000:  # > 2GB
        suggestions.append("💾 Model size is large (>2GB). This may limit batch size. Consider model quantization or pruning.")
    
    # Analyze inference timing
    timing = profile_results.get('timing', {})
    if timing:
        mean_ms = timing.get('mean_ms', 0)
        throughput = timing.get('throughput_samples_per_sec', 0)
        
        if mean_ms > 100:  # > 100ms per sample
            suggestions.append("🐌 Inference is slow (>100ms/sample). Consider:")
            suggestions.append("   - Using mixed precision (--use_amp)")
            suggestions.append("   - Reducing input resolution")
            suggestions.append("   - Using torch.compile() for optimization")
        
        if throughput < 10:  # < 10 samples/sec
            suggestions.append("📉 Low throughput (<10 samples/sec). Consider increasing batch size or using model parallelism.")
    
    # Analyze memory usage (inference)
    memory = profile_results.get('memory', {})
    if memory.get('memory_mb') is not None:
        mem_mb = memory.get('memory_mb', 0)
        if mem_mb > 8000:  # > 8GB
            suggestions.append("🔋 High inference memory usage (>8GB). Consider:")
            suggestions.append("   - Reducing batch size")
            suggestions.append("   - Using gradient checkpointing")
            suggestions.append("   - Using mixed precision (--use_amp)")
        elif mem_mb > 4000:  # > 4GB
            suggestions.append("💡 Moderate inference memory usage (>4GB). Mixed precision (--use_amp) could help reduce it.")
    
    # Analyze training step profiling
    train_timing = profile_results.get('train_timing', {})
    if train_timing and 'error' not in train_timing:
        total = train_timing.get('total', {})
        forward = train_timing.get('forward', {})
        backward = train_timing.get('backward', {})
        optimizer = train_timing.get('optimizer', {})
        
        total_ms = total.get('mean_ms', 0)
        forward_ms = forward.get('mean_ms', 0)
        backward_ms = backward.get('mean_ms', 0)
        opt_ms = optimizer.get('mean_ms', 0)
        
        forward_pct = forward.get('mean_percent', 0)
        backward_pct = backward.get('mean_percent', 0)
        opt_pct = optimizer.get('mean_percent', 0)
        
        # Check if backward is much slower than forward (typical, but can be optimized)
        if backward_pct > 60:
            suggestions.append("🔄 Backward pass takes >60% of training time. This is normal but consider:")
            suggestions.append("   - Gradient accumulation to reduce backward frequency")
            suggestions.append("   - Gradient checkpointing to trade compute for memory")
        
        # Check optimizer overhead
        if opt_pct > 20:
            suggestions.append("⚙️  Optimizer step takes >20% of training time. Consider:")
            suggestions.append("   - Using a simpler optimizer (SGD instead of Adam)")
            suggestions.append("   - Using fused optimizers (torch.optim.AdamW with fused=True)")
        
        # Check if forward is unusually slow
        if forward_pct > 50 and total_ms > 200:
            suggestions.append("⏩ Forward pass is slow. Consider:")
            suggestions.append("   - Using mixed precision (--use_amp)")
            suggestions.append("   - Reducing model complexity")
            suggestions.append("   - Using torch.compile() for optimization")
        
        # Training throughput
        train_throughput = total.get('throughput_samples_per_sec', 0)
        if train_throughput < 5:
            suggestions.append("📊 Low training throughput (<5 samples/sec). Consider:")
            suggestions.append("   - Increasing batch size (if memory allows)")
            suggestions.append("   - Using data parallelism (multiple GPUs)")
            suggestions.append("   - Optimizing data loading pipeline")
    
    # Analyze training memory
    train_memory = profile_results.get('train_memory', {})
    if train_memory and 'error' not in train_memory:
        train_mem_mb = train_memory.get('memory_mb', 0)
        if train_mem_mb > 16000:  # > 16GB
            suggestions.append("🔋 Very high training memory (>16GB). Critical optimizations needed:")
            suggestions.append("   - Reduce batch size significantly")
            suggestions.append("   - Enable gradient checkpointing")
            suggestions.append("   - Use mixed precision (--use_amp)")
            suggestions.append("   - Consider gradient accumulation with smaller batches")
        elif train_mem_mb > 8000:  # > 8GB
            suggestions.append("💾 High training memory (>8GB). Consider:")
            suggestions.append("   - Reducing batch size")
            suggestions.append("   - Using gradient checkpointing")
            suggestions.append("   - Using mixed precision (--use_amp)")
        
        # Compare training vs inference memory
        if memory.get('memory_mb') is not None:
            inference_mem = memory.get('memory_mb', 0)
            if train_mem_mb > inference_mem * 3:
                suggestions.append("📈 Training memory is >3x inference memory. This is expected due to gradients.")
                suggestions.append("   Consider gradient accumulation to use smaller batches during training.")
    
    # Analyze FLOPs
    flops = profile_results.get('flops', {})
    if flops.get('total_gflops') is not None:
        total_gflops = flops.get('total_gflops', 0)
        per_sample_gflops = flops.get('per_sample_gflops', 0)
        
        if total_gflops > 100:  # > 100 GFLOPs
            suggestions.append("🔢 High computational cost (>100 GFLOPs). Consider:")
            suggestions.append("   - Using a smaller model variant")
            suggestions.append("   - Model pruning or quantization")
            suggestions.append("   - Reducing input resolution")
        
        # Estimate theoretical throughput based on GPU
        # Assuming modern GPU (e.g., A100 ~312 TFLOPS, V100 ~125 TFLOPS)
        # Conservative estimate: 50% utilization = ~50-150 TFLOPS effective
        if per_sample_gflops > 0:
            estimated_max_throughput = 50000 / per_sample_gflops  # Conservative estimate
            if timing and timing.get('throughput_samples_per_sec', 0) < estimated_max_throughput * 0.3:
                suggestions.append("⚡ Throughput is <30% of theoretical maximum. Consider:")
                suggestions.append("   - Checking for CPU-GPU synchronization bottlenecks")
                suggestions.append("   - Using larger batch sizes")
                suggestions.append("   - Profiling with --torch_profiler to identify bottlenecks")
    
    # Device-specific suggestions
    input_shape = profile_results.get('input_shape', None)
    if input_shape:
        batch_size = input_shape[0] if len(input_shape) > 0 else 1
        if batch_size == 1:
            suggestions.append("📦 Batch size is 1. Consider increasing batch size for better GPU utilization.")
        elif batch_size < 8:
            suggestions.append("💡 Small batch size (<8). Larger batches typically improve GPU utilization and throughput.")
    
    # Check if AMP is not being used but could help
    if not profile_results.get('use_amp', False):
        if memory.get('memory_mb', 0) > 2000 or (train_memory and train_memory.get('memory_mb', 0) > 4000):
            suggestions.append("💡 Mixed precision (--use_amp) is not enabled. It could reduce memory usage by ~50% and speed up training.")
    
    # General suggestions if no specific issues found
    if len(suggestions) == 0:
        suggestions.append("✅ Profile looks good! No major bottlenecks detected.")
        suggestions.append("💡 For further optimization, consider:")
        suggestions.append("   - Using --torch_profiler for detailed analysis")
        suggestions.append("   - Profiling with --train_step to analyze training performance")
    
    return suggestions

def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = 'cuda',
    num_warmup: int = 10,
    num_iterations: int = 100,
    use_amp: bool = False,
    estimate_flops_count: bool = True,
    use_torch_profiler: bool = False,
    profiler_trace_path: Optional[str] = None,
    profile_train_step: bool = False,
    optimizer_fn=None,
    **model_kwargs
) -> dict:
    """Profile a PyTorch model.
    
    Args:
        model: PyTorch model to profile
        input_shape: Shape of input tensor (batch_size, channels, height, width)
        device: Device to run on ('cuda' or 'cpu')
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations for timing
        use_amp: Whether to use automatic mixed precision
        estimate_flops_count: Whether to estimate FLOPs
        **model_kwargs: Additional arguments to pass to model.forward()
    
    Returns:
        Dictionary with profile results
    """
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Collect machine information
    machine_info = collect_machine_info()
    
    results = {
        'machine_info': machine_info,
        'parameters': count_parameters(model),
        'model_size': get_model_size(model),
        'input_shape': input_shape,
    }
    
    # Forward pass timing
    try:
        results['timing'] = profile_forward_pass(
            model, dummy_input, device, num_warmup, num_iterations, use_amp, **model_kwargs
        )
    except Exception as e:
        print(f"Warning: Forward pass profiling failed: {e}")
        results['timing'] = None
    
    # Memory profiling (CUDA only)
    if device.type == 'cuda':
        try:
            results['memory'] = profile_memory(model, dummy_input, device, use_amp, **model_kwargs)
        except Exception as e:
            print(f"Warning: Memory profiling failed: {e}")
            results['memory'] = {'memory_mb': None, 'memory_gb': None}
    else:
        results['memory'] = {'memory_mb': None, 'memory_gb': None}
    
    # FLOPs estimation
    if estimate_flops_count:
        try:
            results['flops'] = estimate_flops(model, dummy_input, device, **model_kwargs)
        except Exception as e:
            print(f"Warning: FLOPs estimation failed: {e}")
            results['flops'] = {'error': str(e)}
    
    # PyTorch profiler
    if use_torch_profiler:
        try:
            print("Running PyTorch profiler...")
            profiler_results = profile_with_torch_profiler(
                model, dummy_input, device, 
                num_warmup=5,  # Fewer warmup iterations for profiler
                num_iterations=10,  # Fewer iterations as profiler is slower
                use_amp=use_amp,
                trace_path=profiler_trace_path,
                **model_kwargs
            )
            results['torch_profiler'] = profiler_results
        except Exception as e:
            print(f"Warning: PyTorch profiler failed: {e}")
            results['torch_profiler'] = {'error': str(e)}
    
    # Training step profiling
    if profile_train_step:
        try:
            print("Profiling training step (forward + backward + optimizer)...")
            # Reference the function from module namespace to avoid name conflict with parameter
            import sys
            current_module = sys.modules[__name__]
            profile_train_step_func = getattr(current_module, 'profile_train_step')
            train_timing = profile_train_step_func(
                model, dummy_input, device,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
                use_amp=use_amp,
                optimizer_fn=optimizer_fn,
                **model_kwargs
            )
            results['train_timing'] = train_timing
            
            # Training step memory profiling
            if device.type == 'cuda':
                try:
                    train_memory = profile_train_memory(
                        model, dummy_input, device,
                        use_amp=use_amp,
                        optimizer_fn=optimizer_fn,
                        **model_kwargs
                    )
                    results['train_memory'] = train_memory
                except Exception as e:
                    print(f"Warning: Training memory profiling failed: {e}")
                    results['train_memory'] = {'error': str(e)}
        except Exception as e:
            print(f"Warning: Training step profiling failed: {e}")
            results['train_timing'] = {'error': str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Profile PyTorch models')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--num_warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of timing iterations')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--no_flops', action='store_true', help='Skip FLOPs estimation')
    parser.add_argument('--cfgs', nargs='+', default=[], help='Gin config files')
    parser.add_argument('--gin', nargs='+', default=[], help='Gin bindings')
    parser.add_argument('--model_name', type=str, default=None, help='Model name for report')
    parser.add_argument('--save', type=str, default=None, help='Path to save profile results (JSON or TXT format)')
    parser.add_argument('--save_format', type=str, default='json', choices=['json', 'txt'], 
                       help='Format for saved results (default: json)')
    parser.add_argument('--torch_profiler', action='store_true', 
                       help='Use PyTorch profiler for detailed profiling')
    parser.add_argument('--profiler_trace', type=str, default=None,
                       help='Path to save PyTorch profiler trace (chrome trace format)')
    parser.add_argument('--train_step', action='store_true', default=False,
                       help='Profile training step (forward + backward + optimizer)')
    
    args = parser.parse_args() #aug_parse(parser)
    
    # Load gin configs
    if args.cfgs:
        for cfg in args.cfgs:
            gin.parse_config_file(cfg)
    
    if args.gin:
        gin.parse_config(str.join(' ', args.gin))
    
    # Build model
    try:
        model = build_model.build_model(args)
    except Exception as e:
        print(f"Error building model: {e}")
        print("\nMake sure to specify model with --gin build_model.model_fn=@model_name")
        sys.exit(1)
    
    model_name = args.model_name or model.__class__.__name__
    
    # Create input shape
    input_shape = (args.batch_size, args.channels, args.img_size, args.img_size)
    
    print(f"\nProfiling model: {model_name}")
    print(f"Input shape: {input_shape}")
    print(f"Device: {args.device}")
    
    # Profile model
    results = profile_model(
        model,
        input_shape=input_shape,
        device=args.device,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        use_amp=args.use_amp,
        estimate_flops_count=not args.no_flops,
        use_torch_profiler=args.torch_profiler,
        profiler_trace_path=args.profiler_trace,
        profile_train_step=args.train_step,
    )
    
    # Print report
    print_profile_report(results, model_name)
    
    # Print suggestions
    suggestions = summary_suggestions(results)
    if suggestions:
        print("\n" + "=" * 80)
        print("💡 Performance Suggestions:")
        print("=" * 80)
        for suggestion in suggestions:
            print(suggestion)
        print("=" * 80 + "\n")
    
    # Save results if requested
    if args.save:
        # Infer format from file extension if not explicitly provided
        save_format = args.save_format
        if save_format == 'json':  # Check if we should infer from extension
            save_path = Path(args.save)
            if save_path.suffix.lower() == '.txt':
                save_format = 'txt'
            elif save_path.suffix.lower() == '.json':
                save_format = 'json'
        
        save_profile_results(results, args.save, model_name, format=save_format)
    
    return results


if __name__ == '__main__':
    main()


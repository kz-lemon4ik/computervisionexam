"""
Performance optimization utilities
CPU optimizations for VM environment
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from functools import lru_cache
import time


class ModelOptimizer:
    """Optimize models for CPU inference"""

    @staticmethod
    def optimize_for_cpu(model):
        """Apply CPU-specific optimizations"""
        # Set to evaluation mode
        model.eval()

        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False

        # Use Intel MKL-DNN if available
        try:
            model = torch.jit.script(model)
            print("Model compiled with TorchScript")
        except Exception as e:
            print(f"TorchScript compilation failed: {e}")

        return model

    @staticmethod
    def quantize_model(model, sample_input):
        """Apply dynamic quantization for CPU"""
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            print("Model quantized for CPU")
            return quantized_model
        except Exception as e:
            print(f"Quantization failed: {e}")
            return model


class ImageProcessor:
    """Optimized image processing"""

    def __init__(self, target_size=(128, 64)):
        self.target_size = target_size
        self.interpolation = cv2.INTER_LINEAR

    @lru_cache(maxsize=32)
    def _get_resize_params(self, input_shape):
        """Cached resize parameter calculation"""
        h, w = input_shape[:2]
        target_w, target_h = self.target_size

        # Calculate optimal resize strategy
        scale_w = target_w / w
        scale_h = target_h / h

        return scale_w, scale_h

    def preprocess_batch(self, images):
        """Batch preprocessing for efficiency"""
        if not isinstance(images, list):
            images = [images]

        processed = []
        for img in images:
            # Resize efficiently
            resized = cv2.resize(
                img, self.target_size, interpolation=self.interpolation
            )

            # Normalize
            normalized = resized.astype(np.float32) / 255.0

            # Convert to tensor
            tensor = torch.from_numpy(normalized).permute(2, 0, 1)
            processed.append(tensor)

        # Stack into batch
        if len(processed) == 1:
            return processed[0].unsqueeze(0)
        else:
            return torch.stack(processed)


class PerformanceProfiler:
    """Profile pipeline performance"""

    def __init__(self):
        self.timings = {}
        self.counters = {}

    def start_timer(self, name):
        """Start timing an operation"""
        self.timings[name] = {"start": time.time()}

    def end_timer(self, name):
        """End timing an operation"""
        if name in self.timings and "start" in self.timings[name]:
            elapsed = time.time() - self.timings[name]["start"]

            if "times" not in self.timings[name]:
                self.timings[name]["times"] = []

            self.timings[name]["times"].append(elapsed)
            return elapsed
        return 0

    def increment_counter(self, name):
        """Increment a counter"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += 1

    def get_stats(self):
        """Get performance statistics"""
        stats = {}

        for name, data in self.timings.items():
            if "times" in data and data["times"]:
                times = data["times"]
                stats[name] = {
                    "count": len(times),
                    "avg_time": np.mean(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "total_time": np.sum(times),
                }

        stats["counters"] = self.counters.copy()
        return stats

    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_stats()

        print("\nPerformance Statistics:")
        print("=" * 40)

        for name, data in stats.items():
            if name != "counters":
                print(f"\n{name}:")
                print(f"  Count: {data['count']}")
                print(f"  Avg Time: {data['avg_time']:.4f}s")
                print(f"  Min Time: {data['min_time']:.4f}s")
                print(f"  Max Time: {data['max_time']:.4f}s")
                print(f"  Total Time: {data['total_time']:.4f}s")

        if stats.get("counters"):
            print("\nCounters:")
            for name, count in stats["counters"].items():
                print(f"  {name}: {count}")


def optimize_pipeline_for_cpu(pipeline):
    """Apply comprehensive CPU optimizations to pipeline"""
    print("Applying CPU optimizations...")

    # Optimize models
    if hasattr(pipeline, "baseline_model") and pipeline.baseline_model:
        pipeline.baseline_model = ModelOptimizer.optimize_for_cpu(
            pipeline.baseline_model
        )

    if hasattr(pipeline, "recognition_model") and pipeline.recognition_model:
        pipeline.recognition_model = ModelOptimizer.optimize_for_cpu(
            pipeline.recognition_model
        )

    # Set optimal thread count for CPU
    torch.set_num_threads(2)  # Optimal for most VMs

    # Disable CUDA if accidentally enabled
    torch.cuda.set_device = lambda x: None

    print("CPU optimizations applied")
    return pipeline


def benchmark_operations():
    """Benchmark common operations"""
    print("Benchmarking operations...")

    profiler = PerformanceProfiler()

    # Test image processing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processor = ImageProcessor()

    for i in range(10):
        profiler.start_timer("image_preprocessing")
        processed = processor.preprocess_batch([test_image])
        profiler.end_timer("image_preprocessing")
        profiler.increment_counter("images_processed")

    # Test model inference (mock)
    test_tensor = torch.randn(1, 3, 64, 128)

    for i in range(10):
        profiler.start_timer("model_inference")
        # Simulate model forward pass
        output = torch.nn.functional.conv2d(test_tensor, torch.randn(32, 3, 3, 3))
        profiler.end_timer("model_inference")
        profiler.increment_counter("inferences")

    profiler.print_stats()
    return profiler.get_stats()


if __name__ == "__main__":
    # Run benchmark
    print("Performance Optimization Module")
    print("=" * 40)

    benchmark_operations()

    print("\nOptimization recommendations:")
    print("- Use batch processing when possible")
    print("- Enable TorchScript compilation")
    print("- Apply dynamic quantization for deployment")
    print("- Monitor memory usage in production")

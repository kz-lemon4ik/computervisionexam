#!/usr/bin/env python3
"""
Model optimization script
Optimize trained models for production deployment
"""

import sys
import torch
from pathlib import Path

sys.path.append('src')
from utils.optimization import ModelOptimizer, benchmark_operations
from models.baseline import BaselineCNN
from recognition.pdlpr_model import create_pdlpr_model

def optimize_baseline_model(model_path, output_path):
    """Optimize baseline model for deployment"""
    print(f"Optimizing baseline model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return False
    
    try:
        # Load model
        model = BaselineCNN()
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # Apply optimizations
        optimized_model = ModelOptimizer.optimize_for_cpu(model)
        
        # Test with sample input
        sample_input = torch.randn(1, 3, 64, 128)
        
        # Try quantization
        quantized_model = ModelOptimizer.quantize_model(optimized_model, sample_input)
        
        # Save optimized model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'optimization': 'cpu_optimized_quantized',
            'input_shape': (1, 3, 64, 128)
        }, output_path)
        
        print(f"Optimized model saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        return False

def optimize_pdlpr_model(model_path, output_path):
    """Optimize PDLPR model for deployment"""
    print(f"Optimizing PDLPR model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return False
    
    try:
        # Load model
        model = create_pdlpr_model()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Apply optimizations
        optimized_model = ModelOptimizer.optimize_for_cpu(model)
        
        # Save optimized model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': optimized_model.state_dict(),
            'optimization': 'cpu_optimized',
            'input_shape': (1, 3, 64, 128)
        }, output_path)
        
        print(f"Optimized PDLPR model saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"PDLPR optimization failed: {e}")
        return False

def benchmark_models():
    """Benchmark model performance"""
    print("\nBenchmarking model performance...")
    
    # Benchmark basic operations
    benchmark_operations()
    
    # Test model loading and inference
    models_to_test = [
        'models/baseline_model.pth',
        'models/pdlpr_best.pth',
        'models/optimized/baseline_optimized.pth',
        'models/optimized/pdlpr_optimized.pth'
    ]
    
    for model_path in models_to_test:
        if Path(model_path).exists():
            print(f"\nTesting {model_path}...")
            
            try:
                # Load and time model loading
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                import time
                load_start = time.time()
                
                if 'baseline' in model_path:
                    model = BaselineCNN()
                    if 'optimized' in model_path:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        state_dict = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(state_dict)
                else:
                    model = create_pdlpr_model()
                    checkpoint = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(checkpoint['model_state_dict'])
                
                load_time = time.time() - load_start
                
                # Test inference
                model.eval()
                sample_input = torch.randn(1, 3, 64, 128)
                
                inference_times = []
                for _ in range(10):
                    inf_start = time.time()
                    with torch.no_grad():
                        output = model(sample_input)
                    inference_times.append(time.time() - inf_start)
                
                avg_inference = sum(inference_times) / len(inference_times)
                
                print(f"  Load time: {load_time:.4f}s")
                print(f"  Avg inference: {avg_inference:.4f}s")
                print(f"  Output shape: {output.shape}")
                
            except Exception as e:
                print(f"  Error testing {model_path}: {e}")

def main():
    print("Model Optimization Script")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path('models/optimized')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Optimize models
    success_count = 0
    
    # Optimize baseline model
    if optimize_baseline_model(
        'models/baseline_model.pth',
        'models/optimized/baseline_optimized.pth'
    ):
        success_count += 1
    
    # Optimize PDLPR model
    if optimize_pdlpr_model(
        'models/pdlpr_best.pth', 
        'models/optimized/pdlpr_optimized.pth'
    ):
        success_count += 1
    
    print(f"\nOptimization completed: {success_count} models optimized")
    
    # Run benchmarks
    benchmark_models()
    
    print(f"\nOptimization recommendations:")
    print("- Use optimized models for production")
    print("- Monitor memory usage during inference")
    print("- Consider model pruning for further size reduction")
    print("- Test performance on target hardware")

if __name__ == "__main__":
    main()
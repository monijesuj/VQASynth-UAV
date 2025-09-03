#!/usr/bin/env python3
"""
Test script to measure inference time for VQASynth-UAV components
"""

import time
import os
import tempfile
import argparse
from pathlib import Path
import statistics
from PIL import Image
import numpy as np

# Import VQASynth components
try:
    from vqasynth.depth import DepthEstimator
    from vqasynth.localize import Localizer
    from vqasynth.scene_fusion import SpatialSceneConstructor
    from vqasynth.prompts import PromptGenerator
    print("✅ VQASynth modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import VQASynth modules: {e}")
    print("Please ensure the package is installed: pip install -e .")
    exit(1)


class InferenceTimer:
    """Context manager for timing inference"""
    
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        print(f"🚀 Starting {self.name}...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"✅ {self.name} completed in {duration:.3f} seconds")
        return False
    
    @property
    def duration(self):
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None


def load_test_image(image_path):
    """Load and validate test image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    print(f"📷 Loaded test image: {image_path} ({image.size[0]}x{image.size[1]})")
    return image


def benchmark_depth_estimation(depth_estimator, image, num_runs=3):
    """Benchmark depth estimation component"""
    print(f"\n🔍 Benchmarking Depth Estimation ({num_runs} runs)...")
    
    times = []
    depth_map = None
    focal_length = None
    
    for i in range(num_runs):
        with InferenceTimer(f"Depth Estimation Run {i+1}") as timer:
            depth_map, focal_length = depth_estimator.run(image)
        times.append(timer.duration)
    
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    
    print(f"📊 Depth Estimation Results:")
    print(f"   Average time: {avg_time:.3f} ± {std_time:.3f} seconds")
    print(f"   Depth map shape: {depth_map.shape if hasattr(depth_map, 'shape') else 'N/A'}")
    print(f"   Focal length: {focal_length}")
    
    return depth_map, focal_length, avg_time


def benchmark_localization(localizer, image, num_runs=3):
    """Benchmark object localization component"""
    print(f"\n🎯 Benchmarking Object Localization ({num_runs} runs)...")
    
    times = []
    masks = None
    bounding_boxes = None
    captions = None
    
    for i in range(num_runs):
        with InferenceTimer(f"Localization Run {i+1}") as timer:
            masks, bounding_boxes, captions = localizer.run(image)
        times.append(timer.duration)
    
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    
    print(f"📊 Localization Results:")
    print(f"   Average time: {avg_time:.3f} ± {std_time:.3f} seconds")
    print(f"   Number of objects detected: {len(captions) if captions else 0}")
    print(f"   Captions: {captions[:3] if captions else []}")  # Show first 3
    
    return masks, bounding_boxes, captions, avg_time


def benchmark_scene_fusion(scene_constructor, image, depth_map, focal_length, masks, cache_dir, num_runs=1):
    """Benchmark scene fusion component"""
    print(f"\n🌐 Benchmarking Scene Fusion ({num_runs} runs)...")
    
    times = []
    pointcloud_data = None
    canonicalized = None
    
    for i in range(num_runs):
        with InferenceTimer(f"Scene Fusion Run {i+1}") as timer:
            pointcloud_data, canonicalized = scene_constructor.run(
                str(i), image, depth_map, focal_length, masks, cache_dir
            )
        times.append(timer.duration)
    
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    
    print(f"📊 Scene Fusion Results:")
    print(f"   Average time: {avg_time:.3f} ± {std_time:.3f} seconds")
    print(f"   Point cloud files: {len(pointcloud_data) if pointcloud_data else 0}")
    
    return pointcloud_data, canonicalized, avg_time


def benchmark_prompt_generation(prompt_generator, captions, pointcloud_data, canonicalized, num_runs=3):
    """Benchmark prompt generation component"""
    print(f"\n💭 Benchmarking Prompt Generation ({num_runs} runs)...")
    
    if not captions or not pointcloud_data:
        print("⚠️  Skipping prompt generation - missing captions or pointcloud data")
        return None, 0.0
    
    times = []
    prompts = None
    
    for i in range(num_runs):
        with InferenceTimer(f"Prompt Generation Run {i+1}") as timer:
            prompts = prompt_generator.run(captions, pointcloud_data, canonicalized)
        times.append(timer.duration)
    
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    
    print(f"📊 Prompt Generation Results:")
    print(f"   Average time: {avg_time:.3f} ± {std_time:.3f} seconds")
    print(f"   Number of prompts: {len(prompts) if prompts else 0}")
    if prompts:
        print(f"   Sample prompt: {prompts[0][:100]}..." if len(prompts[0]) > 100 else prompts[0])
    
    return prompts, avg_time


def benchmark_full_pipeline(image_path, num_runs=1):
    """Benchmark the complete VQASynth pipeline"""
    print(f"\n🚀 Running Full Pipeline Benchmark...")
    print("=" * 60)
    
    # Load test image
    image = load_test_image(image_path)
    
    # Initialize components
    with InferenceTimer("Component Initialization"):
        depth_estimator = DepthEstimator(from_onnx=False)
        localizer = Localizer()
        scene_constructor = SpatialSceneConstructor()
        prompt_generator = PromptGenerator()
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        total_times = []
        
        for run in range(num_runs):
            print(f"\n🔄 Pipeline Run {run + 1}/{num_runs}")
            print("-" * 40)
            
            pipeline_start = time.time()
            
            # Run each component
            depth_map, focal_length, depth_time = benchmark_depth_estimation(
                depth_estimator, image, num_runs=1
            )
            
            masks, bounding_boxes, captions, loc_time = benchmark_localization(
                localizer, image, num_runs=1
            )
            
            pointcloud_data, canonicalized, fusion_time = benchmark_scene_fusion(
                scene_constructor, image, depth_map, focal_length, masks, temp_dir, num_runs=1
            )
            
            prompts, prompt_time = benchmark_prompt_generation(
                prompt_generator, captions, pointcloud_data, canonicalized, num_runs=1
            )
            
            pipeline_end = time.time()
            total_time = pipeline_end - pipeline_start
            total_times.append(total_time)
            
            print(f"\n📊 Pipeline Run {run + 1} Summary:")
            print(f"   Depth Estimation: {depth_time:.3f}s")
            print(f"   Object Localization: {loc_time:.3f}s")
            print(f"   Scene Fusion: {fusion_time:.3f}s")
            print(f"   Prompt Generation: {prompt_time:.3f}s")
            print(f"   Total Pipeline Time: {total_time:.3f}s")
    
    if num_runs > 1:
        avg_total = statistics.mean(total_times)
        std_total = statistics.stdev(total_times)
        print(f"\n🏁 Final Results ({num_runs} runs):")
        print(f"   Average Total Time: {avg_total:.3f} ± {std_total:.3f} seconds")
    
    return total_times


def main():
    parser = argparse.ArgumentParser(description="Benchmark VQASynth-UAV inference time")
    parser.add_argument("--image", type=str, 
                       default="examples/assets/warehouse_rgb.jpg",
                       help="Path to test image")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs for averaging")
    parser.add_argument("--component", type=str, 
                       choices=["depth", "localize", "fusion", "prompts", "all"],
                       default="all",
                       help="Specific component to benchmark")
    
    args = parser.parse_args()
    
    print("🔬 VQASynth-UAV Inference Time Benchmark")
    print("=" * 60)
    print(f"Test image: {args.image}")
    print(f"Number of runs: {args.runs}")
    print(f"Component: {args.component}")
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Test image not found: {args.image}")
        print("Available example images:")
        example_dir = Path("examples/assets")
        if example_dir.exists():
            for img_file in example_dir.glob("*.{jpg,jpeg,png}"):
                print(f"  - {img_file}")
        return
    
    try:
        if args.component == "all":
            benchmark_full_pipeline(args.image, args.runs)
        else:
            # Individual component benchmarks would go here
            print(f"Individual component benchmarking for '{args.component}' not implemented yet")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

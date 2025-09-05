#!/usr/bin/env python3
"""
Test individual VQASynth components and measure their inference times
"""

import os
import sys
import time
import torch
import tempfile
from PIL import Image
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def test_depth_estimation():
    """Test depth estimation component"""
    print("🔍 Testing Depth Estimation...")
    
    try:
        from vqasynth.depth import DepthEstimator
        
        # Load test image
        image_path = "examples/assets/warehouse_rgb.jpg"
        if not os.path.exists(image_path):
            print(f"❌ Test image not found: {image_path}")
            return None
            
        image = Image.open(image_path)
        print(f"📷 Loaded image: {image.size}")
        
        # Initialize depth estimator
        start_time = time.time()
        depth_estimator = DepthEstimator()
        init_time = time.time() - start_time
        print(f"⏱️  Depth estimator initialization: {init_time:.2f}s")
        
        # Run depth estimation
        start_time = time.time()
        depth_map, focal_length = depth_estimator.run(image)
        inference_time = time.time() - start_time
        print(f"⏱️  Depth estimation inference: {inference_time:.2f}s")
        print(f"📊 Depth map shape: {depth_map.shape}, Focal length: {focal_length:.2f}")
        
        return {
            'init_time': init_time,
            'inference_time': inference_time,
            'depth_shape': depth_map.shape,
            'focal_length': focal_length
        }
        
    except Exception as e:
        print(f"❌ Depth estimation error: {e}")
        return None

def test_localization():
    """Test object localization component"""
    print("\n🎯 Testing Object Localization...")
    
    try:
        from vqasynth.localize import Localizer
        
        # Load test image
        image_path = "examples/assets/warehouse_rgb.jpg"
        image = Image.open(image_path)
        
        # Initialize localizer
        start_time = time.time()
        localizer = Localizer()
        init_time = time.time() - start_time
        print(f"⏱️  Localizer initialization: {init_time:.2f}s")
        
        # Run localization
        start_time = time.time()
        results = localizer.run(image)
        inference_time = time.time() - start_time
        print(f"⏱️  Localization inference: {inference_time:.2f}s")
        
        masks, captions = results
        print(f"📊 Found {len(masks)} objects with captions: {captions}")
        
        return {
            'init_time': init_time,
            'inference_time': inference_time,
            'num_objects': len(masks),
            'captions': captions
        }
        
    except Exception as e:
        print(f"❌ Localization error: {e}")
        return None

def test_prompt_generation():
    """Test prompt generation component"""
    print("\n💭 Testing Prompt Generation...")
    
    try:
        from vqasynth.prompts import PromptGenerator
        
        # Initialize prompt generator
        start_time = time.time()
        prompt_generator = PromptGenerator()
        init_time = time.time() - start_time
        print(f"⏱️  Prompt generator initialization: {init_time:.2f}s")
        
        # Test with sample captions
        test_captions = ["warehouse shelves", "cardboard boxes", "forklift"]
        
        start_time = time.time()
        prompts = prompt_generator.run(test_captions)
        inference_time = time.time() - start_time
        print(f"⏱️  Prompt generation inference: {inference_time:.2f}s")
        print(f"📊 Generated {len(prompts)} prompts")
        
        if prompts:
            print("📝 Sample prompts:")
            for i, prompt in enumerate(prompts[:3]):
                print(f"   {i+1}. {prompt}")
        
        return {
            'init_time': init_time,
            'inference_time': inference_time,
            'num_prompts': len(prompts),
            'sample_prompts': prompts[:3] if prompts else []
        }
        
    except Exception as e:
        print(f"❌ Prompt generation error: {e}")
        return None

def test_scene_fusion():
    """Test scene fusion component with minimal data"""
    print("\n🔗 Testing Scene Fusion...")
    
    try:
        from vqasynth.scene_fusion import SpatialSceneConstructor
        
        # Load test image
        image_path = "examples/assets/warehouse_rgb.jpg"
        image = Image.open(image_path)
        
        # Create some dummy masks for testing
        import numpy as np
        h, w = image.size[1], image.size[0]
        
        # Create a simple rectangular mask
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask1[h//4:3*h//4, w//4:w//2] = 255
        
        mask2 = np.zeros((h, w), dtype=np.uint8)
        mask2[h//4:3*h//4, w//2:3*w//4] = 255
        
        masks = [mask1, mask2]
        
        # Initialize scene constructor
        start_time = time.time()
        scene_constructor = SpatialSceneConstructor()
        init_time = time.time() - start_time
        print(f"⏱️  Scene constructor initialization: {init_time:.2f}s")
        
        # Test with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            start_time = time.time()
            try:
                point_cloud_files, canonicalized, depth_map, focal = scene_constructor.run(
                    "test_image", image, masks, temp_dir
                )
                inference_time = time.time() - start_time
                print(f"⏱️  Scene fusion inference: {inference_time:.2f}s")
                print(f"📊 Generated {len(point_cloud_files)} point cloud files")
                
                return {
                    'init_time': init_time,
                    'inference_time': inference_time,
                    'num_pointclouds': len(point_cloud_files),
                    'canonicalized': canonicalized
                }
            except Exception as inner_e:
                print(f"❌ Scene fusion runtime error: {inner_e}")
                return None
        
    except Exception as e:
        print(f"❌ Scene fusion error: {e}")
        return None

def main():
    """Run all component tests"""
    print("🧪 VQASynth Component Tests")
    print("=" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    results = {}
    total_start_time = time.time()
    
    # Test individual components
    results['depth'] = test_depth_estimation()
    results['localization'] = test_localization()
    results['prompts'] = test_prompt_generation()
    results['scene_fusion'] = test_scene_fusion()
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n📈 Performance Summary")
    print("=" * 50)
    print(f"🕐 Total test time: {total_time:.2f}s")
    
    for component, result in results.items():
        if result:
            init_time = result.get('init_time', 0)
            inference_time = result.get('inference_time', 0)
            print(f"⚡ {component.capitalize()}:")
            print(f"   Init: {init_time:.2f}s, Inference: {inference_time:.2f}s")
        else:
            print(f"❌ {component.capitalize()}: Failed")
    
    # Calculate total inference time
    total_inference = sum(
        result.get('inference_time', 0) 
        for result in results.values() 
        if result
    )
    total_init = sum(
        result.get('init_time', 0) 
        for result in results.values() 
        if result
    )
    
    print(f"\n🏃 End-to-end estimate:")
    print(f"   First run (with init): {total_init + total_inference:.2f}s")
    print(f"   Subsequent runs: {total_inference:.2f}s")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test System Components
Quick verification of each component without requiring models
"""
import sys
import os

def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"     CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ❌ Transformers: {e}")
        return False
    
    try:
        import cv2
        print(f"  ✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"  ❌ OpenCV: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"  ✅ Pillow")
    except ImportError as e:
        print(f"  ❌ Pillow: {e}")
        return False
    
    try:
        import gradio as gr
        print(f"  ✅ Gradio {gr.__version__}")
    except ImportError as e:
        print(f"  ❌ Gradio: {e}")
        return False
    
    # Optional
    try:
        import pyrealsense2 as rs
        print(f"  ✅ RealSense SDK (optional)")
    except ImportError:
        print(f"  ⚠️  RealSense SDK not available (will use webcam)")
    
    return True


def test_camera_module():
    """Test camera capture module"""
    print("\n🧪 Testing camera capture module...")
    
    try:
        from camera_stream_capture import CameraStreamCapture
        print("  ✅ Module imported successfully")
        
        # Test with webcam (safer than RealSense for testing)
        print("  Testing webcam initialization...")
        camera = CameraStreamCapture(camera_type='webcam', width=640, height=480, fps=30)
        
        # Don't actually start it, just verify it can be created
        print("  ✅ Camera object created")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def test_inference_module():
    """Test inference engine module"""
    print("\n🧪 Testing inference engine module...")
    
    try:
        from online_spatial_inference import SpatialModelLoader, OnlineInferenceEngine
        print("  ✅ Modules imported successfully")
        
        # Create model loader without loading models
        loader = SpatialModelLoader()
        print("  ✅ Model loader created")
        
        # Check for available models
        available = loader.list_available_models()
        print(f"  📦 Available models: {available if available else 'None found'}")
        
        if not available:
            print("  ⚠️  No models found (this is OK for testing)")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def test_interfaces():
    """Test that interface files are valid"""
    print("\n🧪 Testing interface modules...")
    
    try:
        # Just check if files exist and can be imported
        import live_spatial_vqa
        print("  ✅ CLI interface module")
    except Exception as e:
        print(f"  ❌ CLI interface: {e}")
        return False
    
    try:
        import live_spatial_vqa_gradio
        print("  ✅ Gradio interface module")
    except Exception as e:
        print(f"  ❌ Gradio interface: {e}")
        return False
    
    return True


def check_models():
    """Check if models are downloaded"""
    print("\n📦 Checking for models...")
    
    models_found = []
    
    if os.path.exists("SpaceOm"):
        print("  ✅ SpaceOm found")
        models_found.append("SpaceOm")
    else:
        print("  ❌ SpaceOm not found")
    
    if os.path.exists("SpaceThinker-Qwen2.5VL-3B"):
        print("  ✅ SpaceThinker found")
        models_found.append("SpaceThinker")
    else:
        print("  ❌ SpaceThinker not found")
    
    if not models_found:
        print("\n  💡 To download models:")
        print("     git clone https://huggingface.co/remyxai/SpaceOm")
        print("     git clone https://huggingface.co/remyxai/SpaceThinker-Qwen2.5VL-3B")
    
    return len(models_found) > 0


def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 Live Spatial VQA System - Component Tests")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "Camera Module": test_camera_module(),
        "Inference Module": test_inference_module(),
        "Interfaces": test_interfaces(),
        "Models Available": check_models()
    }
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All tests passed! System is ready.")
        print("\n🚀 To start the system:")
        print("   CLI:    python live_spatial_vqa.py")
        print("   Gradio: python live_spatial_vqa_gradio.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before running.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

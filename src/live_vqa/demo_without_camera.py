#!/usr/bin/env python3
"""
Simple Demo - Test the system without camera
Quick demonstration of the inference pipeline
"""
import sys
from PIL import Image
import numpy as np

def create_test_image():
    """Create a simple test image"""
    # Create a colorful test pattern
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles
    img[100:200, 100:300] = [255, 0, 0]  # Red rectangle
    img[250:350, 350:550] = [0, 255, 0]  # Green rectangle
    img[150:250, 400:500] = [0, 0, 255]  # Blue rectangle
    
    # Add some text areas (white)
    img[50:80, 50:200] = [255, 255, 255]
    img[400:430, 450:600] = [255, 255, 255]
    
    return Image.fromarray(img)


def test_without_camera():
    """Test inference pipeline without requiring a camera"""
    print("🧪 Testing Live Spatial VQA - Demo Mode")
    print("=" * 60)
    
    # Test imports
    print("\n1️⃣ Testing imports...")
    try:
        from online_spatial_inference import SpatialModelLoader, OnlineInferenceEngine
        print("   ✅ Modules imported")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Check for models
    print("\n2️⃣ Checking for models...")
    loader = SpatialModelLoader()
    available = loader.list_available_models()
    
    if not available:
        print("   ❌ No models found!")
        print("   Please download models first:")
        print("      git clone https://huggingface.co/remyxai/SpaceOm")
        print("      git clone https://huggingface.co/remyxai/SpaceThinker-Qwen2.5VL-3B")
        return False
    
    print(f"   ✅ Found: {', '.join(available)}")
    
    # Load model
    print(f"\n3️⃣ Loading {available[0]}...")
    if not loader.load_model(available[0]):
        print("   ❌ Failed to load model")
        return False
    
    print("   ✅ Model loaded")
    
    # Create inference engine
    print("\n4️⃣ Creating inference engine...")
    engine = OnlineInferenceEngine(loader)
    print("   ✅ Engine ready")
    
    # Create test image
    print("\n5️⃣ Creating test image...")
    test_image = create_test_image()
    print("   ✅ Test image created (640x480)")
    
    # Update engine with image
    print("\n6️⃣ Processing image...")
    engine.update_frame(test_image)
    print("   ✅ Image processed")
    
    # Ask a test question
    print("\n7️⃣ Testing inference...")
    question = "Describe what you see in this image."
    print(f"   Question: {question}")
    
    result = engine.ask_question(question, max_new_tokens=100)
    
    if result['success']:
        print(f"\n   ✅ Answer: {result['answer']}")
        print(f"   ⏱️  Inference time: {result['inference_time']:.2f}s")
        print(f"   🤖 Model: {result['model']}")
    else:
        print(f"\n   ❌ Failed: {result.get('error', 'Unknown error')}")
        return False
    
    # Show statistics
    print("\n8️⃣ Statistics:")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("\n💡 Next steps:")
    print("   - Run with camera: python live_spatial_vqa_gradio.py")
    print("   - Or CLI version: python live_spatial_vqa.py")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_without_camera()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

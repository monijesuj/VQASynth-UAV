#!/usr/bin/env python3
"""
Demo script for Enhanced Semantic Segmentation
Tests the system with sample images and prompts
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def create_sample_image():
    """Create a sample image for testing"""
    # Create a simple test image with different colored regions
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Background
    img[:] = [50, 50, 50]  # Dark gray
    
    # Floor (bottom region)
    img[300:, :] = [100, 100, 100]  # Light gray
    
    # Red box on the left
    img[200:300, 50:150] = [0, 0, 255]  # Red
    
    # Blue box on the right
    img[200:300, 450:550] = [255, 0, 0]  # Blue
    
    # Green box in center
    img[150:250, 250:350] = [0, 255, 0]  # Green
    
    # Yellow circle (person-like)
    cv2.circle(img, (300, 100), 30, (0, 255, 255), -1)
    
    return img

def test_enhanced_segmentation():
    """Test the enhanced segmentation system"""
    print("🧪 Testing Enhanced Semantic Segmentation System")
    print("=" * 50)
    
    try:
        from enhanced_semantic_segmentation import EnhancedSemanticSegmentation
        
        # Initialize segmenter
        print("🔧 Initializing segmenter...")
        segmenter = EnhancedSemanticSegmentation(device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
        
        # Create sample image
        print("📸 Creating sample image...")
        sample_img = create_sample_image()
        
        # Save sample image
        sample_path = "sample_test_image.jpg"
        cv2.imwrite(sample_path, sample_img)
        print(f"   Saved sample image: {sample_path}")
        
        # Test prompts
        test_prompts = [
            "floor",
            "red box",
            "blue box", 
            "green box",
            "yellow circle",
            "left side",
            "right side",
            "center"
        ]
        
        print("\n🎯 Testing segmentation with different prompts...")
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n{i+1}. Testing prompt: '{prompt}'")
            
            try:
                # Run segmentation
                result, info = segmenter.segment_image(sample_path, prompt)
                
                if result is not None:
                    # Save result
                    output_path = f"demo_result_{i+1}_{prompt.replace(' ', '_')}.jpg"
                    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                    
                    print(f"   ✅ Success!")
                    print(f"   📊 Score: {info['score']:.3f}")
                    print(f"   🏷️  Models: {info['models_used']}")
                    print(f"   📁 Saved: {output_path}")
                else:
                    print(f"   ❌ Failed")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n🎉 Demo complete! Check the generated images.")
        print(f"📁 Sample image: {sample_path}")
        print(f"📁 Results: demo_result_*.jpg")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run the setup script first: ./setup_enhanced_segmentation.sh")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_segmentation()

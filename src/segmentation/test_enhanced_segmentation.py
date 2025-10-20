#!/usr/bin/env python3
"""
Test script for Enhanced Semantic Segmentation
Tests if all required modules can be imported and checkpoints exist
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import clip
        print("✅ CLIP: Available")
    except ImportError as e:
        print(f"❌ CLIP: {e}")
        return False
    
    try:
        # Add the sam2 directory to the path to handle the directory structure issue
        import sys
        from pathlib import Path
        sam2_path = Path(__file__).parent / "sam2"
        if str(sam2_path) not in sys.path:
            sys.path.insert(0, str(sam2_path))
        
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✅ SAM2: Available")
    except ImportError as e:
        print(f"❌ SAM2: {e}")
        return False
    except RuntimeError as e:
        if "parent directory" in str(e):
            print("⚠️  SAM2: Directory structure issue - will try alternative import")
            try:
                # Try importing from the sam2 subdirectory
                sys.path.insert(0, str(Path(__file__).parent / "sam2" / "sam2"))
                from build_sam import build_sam2
                from sam2_image_predictor import SAM2ImagePredictor
                print("✅ SAM2: Available (alternative import)")
            except ImportError as e2:
                print(f"❌ SAM2: {e2}")
                return False
        else:
            print(f"❌ SAM2: {e}")
            return False
    
    try:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        print("✅ CLIPSeg: Available")
    except ImportError as e:
        print(f"❌ CLIPSeg: {e}")
        return False
    
    try:
        import groundingdino
        print("✅ GroundingDINO: Available")
    except ImportError as e:
        print(f"⚠️  GroundingDINO: {e} (optional)")
    
    try:
        from ram.models import ram
        print("✅ RAM: Available")
    except ImportError as e:
        print(f"⚠️  RAM: {e} (optional)")
    
    return True

def test_checkpoints():
    """Test if required checkpoints exist"""
    print("\n🔍 Testing checkpoints...")
    
    checkpoints = [
        ("sam2/checkpoints/sam2.1_hiera_large.pt", "SAM2"),
        ("GroundingDINO/weights/groundingdino_swint_ogc.pth", "GroundingDINO"),
        ("ram_swin_large_14m.pth", "RAM")
    ]
    
    all_good = True
    for path, name in checkpoints:
        if Path(path).exists():
            size = Path(path).stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {name}: {path} ({size:.1f} MB)")
        else:
            print(f"❌ {name}: {path} not found")
            all_good = False
    
    return all_good

def test_enhanced_segmentation_import():
    """Test if the enhanced segmentation module can be imported"""
    print("\n🎯 Testing Enhanced Segmentation module...")
    
    try:
        from enhanced_semantic_segmentation import EnhancedSemanticSegmentation
        print("✅ EnhancedSemanticSegmentation: Available")
        return True
    except ImportError as e:
        print(f"❌ EnhancedSemanticSegmentation: {e}")
        return False

def test_batch_processing_import():
    """Test if the batch processing module can be imported"""
    print("\n📦 Testing Batch Processing module...")
    
    try:
        from batch_segmentation import BatchSegmentationProcessor
        print("✅ BatchSegmentationProcessor: Available")
        return True
    except ImportError as e:
        print(f"❌ BatchSegmentationProcessor: {e}")
        return False

def main():
    print("🚀 Enhanced Semantic Segmentation Test")
    print("=====================================")
    
    imports_ok = test_imports()
    checkpoints_ok = test_checkpoints()
    enhanced_ok = test_enhanced_segmentation_import()
    batch_ok = test_batch_processing_import()
    
    print("\n📊 Summary:")
    if imports_ok and checkpoints_ok and enhanced_ok and batch_ok:
        print("✅ All systems ready! You can run:")
        print("   python enhanced_semantic_segmentation.py")
        print("   python launch_enhanced_segmentation.py --mode gui")
        print("   python demo_enhanced_segmentation.py")
    else:
        print("⚠️  Some components missing. Check the errors above.")
        if not imports_ok:
            print("   - Install missing Python packages")
        if not checkpoints_ok:
            print("   - Download missing model checkpoints")
        if not enhanced_ok or not batch_ok:
            print("   - Check if enhanced segmentation files are present")

if __name__ == "__main__":
    main()

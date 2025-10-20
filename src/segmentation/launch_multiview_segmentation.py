#!/usr/bin/env python3
"""
Launcher for Enhanced VGGT with Spatial Segmentation

This script sets up the environment and launches the enhanced demo.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the Python environment and paths."""
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Add necessary paths
    vggt_path = current_dir / "vggt"
    sam2_path = current_dir / "sam2" 
    vqasynth_path = current_dir / "vqasynth"
    
    # Add to Python path
    if str(vggt_path) not in sys.path:
        sys.path.insert(0, str(vggt_path))
    if str(sam2_path) not in sys.path:
        sys.path.insert(0, str(sam2_path))
    if str(vqasynth_path) not in sys.path:
        sys.path.insert(0, str(vqasynth_path))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = f"{vggt_path}:{sam2_path}:{vqasynth_path}:" + os.environ.get("PYTHONPATH", "")
    
    print(f"✅ Environment set up")
    print(f"   VGGT path: {vggt_path}")
    print(f"   SAM2 path: {sam2_path}")
    print(f"   VQASynth path: {vqasynth_path}")

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("   ⚠️  CUDA not available, using CPU")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import gradio
        print(f"✅ Gradio {gradio.__version__}")
    except ImportError:
        missing_deps.append("gradio")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    # Check VGGT
    try:
        from vggt.models.vggt import VGGT
        print("✅ VGGT available")
    except ImportError:
        print("❌ VGGT not available - check vggt submodule")
        missing_deps.append("vggt")
    
    # Check CLIP
    try:
        import clip
        print("✅ CLIP available")
    except ImportError:
        print("⚠️  CLIP not available - spatial features limited")
    
    # Check SAM2
    try:
        from sam2.build_sam import build_sam2
        print("✅ SAM2 available")
    except ImportError:
        print("⚠️  SAM2 not available - using fallback segmentation")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing packages or run setup_multiview_segmentation.sh")
        return False
    
    return True

def launch_demo(demo_type="enhanced"):
    """Launch the appropriate demo."""
    if demo_type == "enhanced":
        print("🚀 Launching Enhanced VGGT Demo with Spatial Segmentation...")
        try:
            # Import and run the enhanced demo
            from enhanced_vggt_demo import demo
            demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
        except ImportError as e:
            print(f"❌ Error importing enhanced demo: {e}")
            print("Falling back to basic multiview segmentation interface...")
            launch_basic_interface()
    elif demo_type == "basic":
        launch_basic_interface()
    elif demo_type == "original":
        launch_original_vggt()

def launch_basic_interface():
    """Launch the basic multiview segmentation interface."""
    try:
        from multiview_segmentation_interface import launch_interface
        launch_interface()
    except ImportError as e:
        print(f"❌ Error launching basic interface: {e}")
        print("Please check if multiview_segmentation_interface.py exists")

def launch_original_vggt():
    """Launch the original VGGT demo."""
    print("🚀 Launching Original VGGT Demo...")
    os.chdir("vggt")
    subprocess.run([sys.executable, "demo_gradio.py"])

def main():
    """Main launcher function."""
    print("🔧 Enhanced VGGT Multi-View Segmentation Launcher")
    print("=" * 50)
    
    # Set up environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return
    
    print("\n🎯 Available demos:")
    print("1. Enhanced VGGT with Spatial Segmentation (Recommended)")
    print("2. Basic Multi-View Segmentation Interface")
    print("3. Original VGGT Demo")
    
    try:
        choice = input("\nChoose demo (1-3, default=1): ").strip()
        if not choice:
            choice = "1"
        
        if choice == "1":
            launch_demo("enhanced")
        elif choice == "2":
            launch_demo("basic")
        elif choice == "3":
            launch_demo("original")
        else:
            print("Invalid choice, launching enhanced demo...")
            launch_demo("enhanced")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Trying to launch basic interface as fallback...")
        launch_basic_interface()

if __name__ == "__main__":
    main()
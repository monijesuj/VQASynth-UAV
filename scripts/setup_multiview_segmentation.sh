#!/bin/bash

# Multi-View Spatial Segmentation Setup Script
# This script sets up the complete environment for the multi-view segmentation system

echo "🔧 Setting up Multi-View Spatial Segmentation System..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi
else
    echo "⚠️  No NVIDIA GPU detected. CPU-only mode will be used."
fi

# Create conda environment if not exists
if ! conda env list | grep -q "multiview-seg"; then
    echo "📦 Creating conda environment..."
    conda create -n multiview-seg python=3.10 -y
fi

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate multiview-seg

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
else
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# Install basic requirements
echo "📋 Installing basic requirements..."
pip install -r requirements.txt

# Install additional packages for multi-view segmentation
echo "🎯 Installing segmentation packages..."
pip install gradio
pip install opencv-python
pip install pillow
pip install scikit-image
pip install matplotlib
pip install plotly

# Install CLIP
echo "🎨 Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git

# Setup VGGT
echo "🏗️ Setting up VGGT..."
if [ ! -d "vggt" ]; then
    git submodule update --init --recursive
fi

# Install VGGT requirements
if [ -f "vggt/requirements.txt" ]; then
    pip install -r vggt/requirements.txt
fi

# Setup SAM2
echo "🎭 Setting up SAM2..."
if [ ! -d "sam2" ]; then
    echo "Cloning SAM2 repository..."
    git clone https://github.com/facebookresearch/sam2.git
fi

cd sam2
pip install -e .
cd ..

# Download SAM2 checkpoints
echo "⬇️ Downloading SAM2 checkpoints..."
mkdir -p checkpoints
cd checkpoints

# Download SAM2.1 checkpoints
if [ ! -f "sam2.1_hiera_large.pt" ]; then
    echo "Downloading SAM2.1 Hiera Large checkpoint..."
    wget -O sam2.1_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
fi

if [ ! -f "sam2.1_hiera_base_plus.pt" ]; then
    echo "Downloading SAM2.1 Hiera Base+ checkpoint..."
    wget -O sam2.1_hiera_base_plus.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
fi

cd ..

# Setup Molmo (if needed)
echo "🤖 Setting up Molmo components..."
if [ ! -d "molmo" ]; then
    mkdir -p molmo
fi

# Create example data directory
echo "📁 Creating example data directory..."
mkdir -p examples/multiview_data
mkdir -p outputs/segmentation_results

# Set up configuration
echo "⚙️ Creating configuration files..."
cat > multiview_config.yaml << EOF
# Multi-View Segmentation Configuration

# Model settings
models:
  vggt:
    checkpoint_url: "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    device: "cuda"  # or "cpu"
  
  sam2:
    checkpoint_path: "./checkpoints/sam2.1_hiera_large.pt"
    config: "sam2.1_hiera_l.yaml"
  
  clip:
    model: "ViT-L/14@336px"

# Processing settings
processing:
  max_images: 50
  frame_extraction_fps: 2  # frames per second for video processing
  confidence_threshold: 0.5
  
  # 3D reconstruction settings
  depth_confidence_threshold: 0.3
  point_cloud_downsample: 1000  # max points per view

# Segmentation settings
segmentation:
  sam2_multimask: true
  projection_radius: 20  # pixels
  spatial_similarity_threshold: 0.2

# Output settings
output:
  save_overlays: true
  save_masks: true
  save_pointcloud: true
  export_format: "png"  # png, jpg
  pointcloud_format: "ply"  # ply, pcd

# Interface settings
interface:
  port: 7860
  share: true
  debug: false
EOF

echo "🎨 Creating example usage script..."
cat > run_example.py << 'EOF'
#!/usr/bin/env python3
"""
Example usage of the Multi-View Spatial Segmentation System
"""

import os
import sys
from multiview_segmentation import MultiViewSegmentationPipeline

def main():
    print("🔍 Multi-View Spatial Segmentation Example")
    
    # Initialize pipeline
    pipeline = MultiViewSegmentationPipeline()
    
    # Example 1: Process images
    if os.path.exists("examples/multiview_data/"):
        image_files = [f for f in os.listdir("examples/multiview_data/") 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if image_files:
            image_paths = [os.path.join("examples/multiview_data/", f) for f in image_files]
            
            # Run segmentation
            results = pipeline.run_segmentation(
                image_paths, 
                "red chair on the left side"
            )
            
            # Save results
            pipeline.save_results(results, "outputs/segmentation_results/example1/")
            print("✅ Example 1 completed! Check outputs/segmentation_results/example1/")
        else:
            print("ℹ️  No example images found. Add images to examples/multiview_data/ to test.")
    
    # Example 2: Process video (if available)
    video_path = "examples/example_video.mp4"
    if os.path.exists(video_path):
        results = pipeline.run_segmentation(
            video_path,
            "person walking in the background"
        )
        
        pipeline.save_results(results, "outputs/segmentation_results/example2/")
        print("✅ Example 2 completed! Check outputs/segmentation_results/example2/")
    else:
        print("ℹ️  No example video found. Add example_video.mp4 to examples/ to test video processing.")
    
    print("\n🎯 To run the interactive interface:")
    print("python multiview_segmentation_interface.py")

if __name__ == "__main__":
    main()
EOF

chmod +x run_example.py

# Create a simple test script
echo "🧪 Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify the multi-view segmentation setup
"""

import sys
import importlib

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    required_modules = [
        'torch', 'torchvision', 'cv2', 'numpy', 'PIL', 
        'gradio', 'clip', 'matplotlib'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    # Test VGGT imports
    try:
        sys.path.append("vggt/")
        from vggt.models.vggt import VGGT
        print("✅ VGGT")
    except ImportError as e:
        print(f"❌ VGGT: {e}")
        failed_imports.append("VGGT")
    
    # Test SAM2 imports
    try:
        sys.path.append("sam2/")
        from sam2.build_sam import build_sam2
        print("✅ SAM2")
    except ImportError as e:
        print(f"❌ SAM2: {e}")
        failed_imports.append("SAM2")
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {', '.join(failed_imports)}")
        print("Please run the setup script again or install missing packages manually.")
        return False
    else:
        print("\n🎉 All imports successful!")
        return True

def test_cuda():
    """Test CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available. Using CPU mode.")
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")

def main():
    print("🔧 Multi-View Segmentation Setup Test\n")
    
    # Test imports
    imports_ok = test_imports()
    
    print()
    
    # Test CUDA
    test_cuda()
    
    print()
    
    if imports_ok:
        print("🎯 Setup appears successful!")
        print("\nNext steps:")
        print("1. Add example images to examples/multiview_data/")
        print("2. Run: python run_example.py")
        print("3. Launch interface: python multiview_segmentation_interface.py")
    else:
        print("❌ Setup incomplete. Please check the error messages above.")

if __name__ == "__main__":
    main()
EOF

chmod +x test_setup.py

echo ""
echo "🎉 Setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Test the installation: python test_setup.py"
echo "2. Run example: python run_example.py"  
echo "3. Launch interface: python multiview_segmentation_interface.py"
echo ""
echo "📁 Directory structure:"
echo "├── multiview_segmentation.py       # Core segmentation pipeline"
echo "├── multiview_segmentation_interface.py  # Gradio interface"
echo "├── multiview_config.yaml          # Configuration file"
echo "├── run_example.py                 # Example usage"
echo "├── test_setup.py                  # Setup verification"
echo "├── checkpoints/                   # SAM2 model checkpoints"
echo "├── examples/multiview_data/       # Place your test images here"
echo "└── outputs/segmentation_results/  # Output directory"
echo ""
echo "💡 Tips:"
echo "- Add your test images to examples/multiview_data/"
echo "- The system works best with 3-10 images of the same scene"
echo "- Use clear spatial descriptions like 'red chair on the left side'"
echo ""

conda deactivate
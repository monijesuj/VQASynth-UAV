#!/bin/bash

# Enhanced Semantic Segmentation Setup Script
# This script sets up all required models for the enhanced segmentation system

set -e

echo "🚀 Setting up Enhanced Semantic Segmentation System"
echo "=================================================="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✅ Conda found"
    CONDA_AVAILABLE=true
else
    echo "⚠️  Conda not found, using pip only"
    CONDA_AVAILABLE=false
fi

# Create conda environment if conda is available
if [ "$CONDA_AVAILABLE" = true ]; then
    echo "📦 Creating conda environment..."
    conda create -n enhanced-seg python=3.10 -y || echo "Environment might already exist"
    echo "source activate enhanced-seg" >> ~/.bashrc
    source activate enhanced-seg || conda activate enhanced-seg
fi

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
if [ "$CONDA_AVAILABLE" = true ]; then
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install opencv-python pillow scikit-image matplotlib gradio numpy scipy

# Install CLIP
echo "🎨 Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git

# Install transformers for CLIPSeg
echo "🤗 Installing transformers for CLIPSeg..."
pip install transformers

# Setup SAM2
echo "🎯 Setting up SAM2..."
if [ ! -d "sam2" ]; then
    git clone https://github.com/facebookresearch/sam2.git
fi

cd sam2
pip install -e .
cd ..

# Download SAM2 checkpoint
echo "📥 Downloading SAM2 checkpoint..."
mkdir -p sam2/checkpoints
if [ ! -f "sam2/checkpoints/sam2.1_hiera_large.pt" ]; then
    wget -O sam2/checkpoints/sam2.1_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    echo "✅ SAM2 checkpoint downloaded"
else
    echo "✅ SAM2 checkpoint already exists"
fi

# Setup GroundingDINO (optional)
echo "🎯 Setting up GroundingDINO (optional)..."
if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git
fi

cd GroundingDINO
pip install -e .
cd ..

# Download GroundingDINO checkpoint
echo "📥 Downloading GroundingDINO checkpoint..."
mkdir -p GroundingDINO/weights
if [ ! -f "GroundingDINO/weights/groundingdino_swint_ogc.pth" ]; then
    wget -O GroundingDINO/weights/groundingdino_swint_ogc.pth "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    echo "✅ GroundingDINO checkpoint downloaded"
else
    echo "✅ GroundingDINO checkpoint already exists"
fi

# Setup RAM (optional)
echo "🏷️  Setting up RAM (optional)..."
if [ ! -d "ram" ]; then
    git clone https://github.com/xinyu1205/recognize-anything.git ram
fi

cd ram
pip install -e .
cd ..

# Download RAM checkpoint
echo "📥 Downloading RAM checkpoint..."
if [ ! -f "ram_swin_large_14m.pth" ]; then
    wget -O ram_swin_large_14m.pth "https://huggingface.co/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth"
    echo "✅ RAM checkpoint downloaded"
else
    echo "✅ RAM checkpoint already exists"
fi

# Setup CLIPSeg (already handled by transformers)
echo "🎨 CLIPSeg will be downloaded automatically on first use"

# Create test script
echo "🧪 Creating test script..."
cat > test_enhanced_segmentation.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for Enhanced Semantic Segmentation
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
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✅ SAM2: Available")
    except ImportError as e:
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

def main():
    print("🚀 Enhanced Semantic Segmentation Test")
    print("=====================================")
    
    imports_ok = test_imports()
    checkpoints_ok = test_checkpoints()
    
    print("\n📊 Summary:")
    if imports_ok and checkpoints_ok:
        print("✅ All systems ready! You can run:")
        print("   python enhanced_semantic_segmentation.py")
    else:
        print("⚠️  Some components missing. Check the errors above.")
        if not imports_ok:
            print("   - Install missing Python packages")
        if not checkpoints_ok:
            print("   - Download missing model checkpoints")

if __name__ == "__main__":
    main()
EOF

chmod +x test_enhanced_segmentation.py

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "To test the installation, run:"
echo "  python test_enhanced_segmentation.py"
echo ""
echo "To launch the enhanced segmentation interface, run:"
echo "  python enhanced_semantic_segmentation.py"
echo ""
echo "The interface will be available at: http://localhost:7861"
echo ""
echo "📝 Notes:"
echo "- Some models are optional and will be auto-detected"
echo "- First run may take time to download model weights"
echo "- Make sure you have sufficient GPU memory (8GB+ recommended)"
echo ""

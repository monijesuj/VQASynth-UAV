#!/bin/bash
# Quick Start Script for Live Spatial VQA System

set -e

echo "🚀 Live Spatial VQA - Quick Start"
echo "=================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version || { echo "❌ Python 3 not found!"; exit 1; }

# Check if models exist
echo ""
echo "📦 Checking for models..."

MODELS_FOUND=0

if [ -d "SpaceOm" ]; then
    echo "  ✅ SpaceOm found"
    MODELS_FOUND=$((MODELS_FOUND + 1))
else
    echo "  ⚠️  SpaceOm not found"
fi

if [ -d "SpaceThinker-Qwen2.5VL-3B" ]; then
    echo "  ✅ SpaceThinker found"
    MODELS_FOUND=$((MODELS_FOUND + 1))
else
    echo "  ⚠️  SpaceThinker not found"
fi

if [ $MODELS_FOUND -eq 0 ]; then
    echo ""
    echo "❌ No models found!"
    echo ""
    echo "Please download models first:"
    echo "  git clone https://huggingface.co/remyxai/SpaceOm"
    echo "  git clone https://huggingface.co/remyxai/SpaceThinker-Qwen2.5VL-3B"
    exit 1
fi

# Install dependencies
echo ""
read -p "📥 Install/update dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing dependencies..."
    pip install -r requirements_live_vqa.txt
fi

# Check for camera
echo ""
echo "📹 Checking camera availability..."

# Try to detect RealSense
if python3 -c "import pyrealsense2" 2>/dev/null; then
    echo "  ✅ RealSense SDK found"
    CAMERA="realsense"
else
    echo "  ⚠️  RealSense SDK not found, will use webcam"
    CAMERA="webcam"
fi

# Choose interface
echo ""
echo "🎮 Choose interface:"
echo "  1) OpenCV (keyboard controls, local display)"
echo "  2) Gradio (web browser, remote access)"
echo ""
read -p "Enter choice (1 or 2): " -n 1 -r
echo

if [[ $REPLY == "1" ]]; then
    echo ""
    echo "🚀 Starting OpenCV interface..."
    echo ""
    echo "Controls:"
    echo "  SPACE/ENTER - Ask a question"
    echo "  'm' - Switch model"
    echo "  'h' - Show help"
    echo "  'q' - Quit"
    echo ""
    python3 live_spatial_vqa.py --camera $CAMERA
elif [[ $REPLY == "2" ]]; then
    echo ""
    echo "🚀 Starting Gradio web interface..."
    echo ""
    echo "The browser will open automatically."
    echo "If not, navigate to: http://localhost:7860"
    echo ""
    python3 live_spatial_vqa_gradio.py --camera $CAMERA
else
    echo "❌ Invalid choice"
    exit 1
fi

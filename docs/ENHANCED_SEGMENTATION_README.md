# 🎯 Enhanced Semantic Segmentation System

A comprehensive prompt-based segmentation system that combines multiple state-of-the-art models for intelligent object segmentation.

## 🌟 Features

- **Multi-Model Integration**: RAM, CLIP, CLIPSeg, SAM2, GroundingDINO
- **Intelligent Prompt Processing**: Spatial keywords, object recognition, semantic understanding
- **Model Fusion**: Combines results from multiple models for better accuracy
- **Batch Processing**: Process multiple images with the same or different prompts
- **Interactive Interface**: User-friendly Gradio web interface
- **Comprehensive Visualization**: Shows prompts, masks, confidence scores, and model information

## 🏗️ Architecture

```
Input Image + Text Prompt
         ↓
    ┌─────────────────────────┐
    │    Multi-Model Pipeline │
    ├─────────────────────────┤
    │ 1. RAM Object Recognition│ → Object tags & context
    │ 2. CLIP Semantic Analysis│ → Attention maps & similarity
    │ 3. GroundingDINO Detection│ → Bounding boxes
    │ 4. CLIPSeg Segmentation │ → Semantic masks
    │ 5. Spatial Keyword Parse│ → Spatial points
    └─────────────────────────┘
         ↓
    ┌─────────────────────────┐
    │   Enhanced Prompt Gen   │
    │ (Points + Boxes + Masks)│
    └─────────────────────────┘
         ↓
    ┌─────────────────────────┐
    │    SAM2 Segmentation    │
    │  (High-quality masks)   │
    └─────────────────────────┘
         ↓
    ┌─────────────────────────┐
    │    Result Fusion        │
    │ (SAM2 + CLIPSeg + ...)  │
    └─────────────────────────┘
         ↓
    Final Segmentation + Visualization
```

## 🚀 Quick Start

### 1. Setup

Run the automated setup script:

```bash
chmod +x setup_enhanced_segmentation.sh
./setup_enhanced_segmentation.sh
```

Or install manually:

```bash
# Create conda environment
conda create -n enhanced-seg python=3.10 -y
conda activate enhanced-seg

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install dependencies
pip install opencv-python pillow scikit-image matplotlib gradio numpy scipy
pip install git+https://github.com/openai/CLIP.git
pip install transformers

# Setup SAM2
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..

# Download checkpoints
mkdir -p sam2/checkpoints
wget -O sam2/checkpoints/sam2.1_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
```

### 2. Test Installation

```bash
python test_enhanced_segmentation.py
```

### 3. Launch Interface

```bash
python enhanced_semantic_segmentation.py
```

The interface will be available at: http://localhost:7861

## 📦 Models Used

### Core Models (Required)
- **CLIP**: Semantic understanding and attention mapping
- **SAM2**: High-quality segmentation with point/box prompts

### Optional Models (Auto-detected)
- **RAM**: Object recognition and tagging
- **GroundingDINO**: Object detection and localization
- **CLIPSeg**: Semantic segmentation
- **molmo**: Enhanced reasoning (integration pending)

## 🎯 Usage Examples

### Single Image Segmentation

```python
from enhanced_semantic_segmentation import EnhancedSemanticSegmentation

# Initialize
segmenter = EnhancedSemanticSegmentation(device="cuda")

# Segment image
result, info = segmenter.segment_image("path/to/image.jpg", "red chair on the left")
```

### Batch Processing

```bash
# Single prompt
python batch_segmentation.py -i input_images/ -o output_results/ -p "floor"

# Multiple prompts
python batch_segmentation.py -i input_images/ -o output_results/ -m "floor" "chair" "table" "person"
```

### Supported Prompts

**Object-based:**
- "chair", "table", "person", "car", "bottle"
- "red box", "blue chair", "green plant"

**Spatial-based:**
- "floor", "ceiling", "wall"
- "left side", "right side", "center"
- "foreground", "background"

**Complex:**
- "red chair on the left side"
- "person in the center of the room"
- "blue box on the floor"

## 🔧 Configuration

### Model Selection

You can enable/disable specific models in the Gradio interface:

- **Use RAM**: Object recognition and tagging
- **Use GroundingDINO**: Object detection
- **Use CLIPSeg**: Semantic segmentation

### Device Configuration

```python
# Use GPU
segmenter = EnhancedSemanticSegmentation(device="cuda")

# Use CPU
segmenter = EnhancedSemanticSegmentation(device="cpu")
```

## 📊 Output Information

The system provides comprehensive information about the segmentation:

```json
{
  "score": 0.85,
  "models_used": "sam2, clipseg",
  "fusion_method": "intersection",
  "num_points": 12,
  "num_boxes": 1,
  "mask_area": 0.23,
  "ram_tags": "chair, furniture, indoor",
  "confidence_scores": {
    "clip": 0.78,
    "grounding_dino": 0.82,
    "clipseg": 0.70
  }
}
```

## 🆚 Comparison with Other Approaches

| Feature | Enhanced System | Single Image | Multi-View |
|---------|----------------|--------------|------------|
| **Models** | 5+ integrated | 2-3 basic | 3-4 complex |
| **Accuracy** | High (fusion) | High | Variable |
| **Speed** | Medium | Fast | Slow |
| **Complexity** | Medium | Low | High |
| **Use Case** | Production | Quick test | 3D reconstruction |

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Use CPU instead
python enhanced_semantic_segmentation.py --device cpu
```

**2. Missing Checkpoints**
```bash
# Run setup script again
./setup_enhanced_segmentation.sh
```

**3. Import Errors**
```bash
# Test installation
python test_enhanced_segmentation.py
```

**4. Low Segmentation Quality**
- Try simpler prompts: "floor", "wall", "object"
- Enable more models in the interface
- Check image quality and resolution

### Performance Tips

1. **GPU Memory**: Use 8GB+ VRAM for best performance
2. **Image Size**: Resize large images to 1024x1024 for faster processing
3. **Model Selection**: Disable unused models to save memory
4. **Batch Processing**: Use batch mode for multiple images

## 🔄 Future Enhancements

1. **Multi-Object Segmentation**: Segment multiple objects simultaneously
2. **Instance Segmentation**: Separate individual instances
3. **Interactive Refinement**: User-guided mask refinement
4. **Export Options**: COCO, YOLO, and other annotation formats
5. **Real-time Processing**: Video stream segmentation
6. **3D Integration**: Combine with VGGT for 3D segmentation

## 📝 API Reference

### EnhancedSemanticSegmentation

```python
class EnhancedSemanticSegmentation:
    def __init__(self, device: str = "cuda")
    def segment_image(self, image_path: str, text_prompt: str) -> Tuple[Optional[np.ndarray], Dict]
    def get_ram_tags(self, image: np.ndarray) -> List[str]
    def get_grounding_dino_boxes(self, image: np.ndarray, text_prompt: str) -> List[np.ndarray]
    def get_clip_attention_map(self, image: np.ndarray, text_prompt: str) -> Tuple[float, np.ndarray]
    def get_clipseg_mask(self, image: np.ndarray, text_prompt: str) -> Optional[np.ndarray]
```

### BatchSegmentationProcessor

```python
class BatchSegmentationProcessor:
    def __init__(self, device: str = "cuda")
    def process_batch(self, input_dir: str, output_dir: str, text_prompt: str) -> Dict
    def process_with_multiple_prompts(self, input_dir: str, output_dir: str, prompts: List[str]) -> Dict
```

## 🎯 Why This Approach?

1. **Comprehensive**: Combines multiple state-of-the-art models
2. **Robust**: Fallback mechanisms and error handling
3. **Flexible**: Easy to add/remove models
4. **Production-Ready**: Batch processing and comprehensive logging
5. **User-Friendly**: Intuitive interface and clear documentation

## 📄 License

This project is part of the VQASynth-UAV system. See the main LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test script
3. Open an issue on GitHub
4. Check the existing documentation

---

**Ready to start segmenting? Run the setup script and launch the interface!** 🚀

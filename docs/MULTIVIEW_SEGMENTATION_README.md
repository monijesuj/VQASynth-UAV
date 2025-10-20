# Multi-View Spatial Segmentation System

A comprehensive system that combines **VGGT 3D reconstruction**, **SAM2 segmentation**, **CLIP semantic understanding**, and **spatial reasoning** to enable prompt-based object segmentation across multiple views.

## 🌟 Features

- **🏗️ 3D Reconstruction**: Uses VGGT to create accurate 3D point clouds from multiple images or video sequences
- **🎯 Spatial Segmentation**: Segment objects using natural language descriptions like "red chair on the left side"
- **🔄 Multi-View Consistency**: Maintains consistent segmentation across all input views
- **🎨 Interactive Interface**: User-friendly Gradio web interface for easy interaction
- **📊 Rich Outputs**: Generates 2D segmentation masks, 3D point clouds, and overlay visualizations

## 🚀 Quick Start

### 1. Setup

Run the automated setup script:

```bash
chmod +x setup_multiview_segmentation.sh
./setup_multiview_segmentation.sh
```

Or install manually:

```bash
# Create conda environment
conda create -n multiview-seg python=3.10 -y
conda activate multiview-seg

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other requirements
pip install gradio opencv-python pillow scikit-image matplotlib
pip install git+https://github.com/openai/CLIP.git

# Setup SAM2 (if not already done)
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..

# Download SAM2 checkpoints
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
```

### 2. Launch

Choose your preferred interface:

```bash
# Option 1: Launch with automatic setup
python launch_multiview_segmentation.py

# Option 2: Enhanced VGGT demo (recommended)
python enhanced_vggt_demo.py

# Option 3: Basic segmentation interface
python multiview_segmentation_interface.py

# Option 4: Core pipeline only
python multiview_segmentation.py
```

### 3. Usage

1. **Upload** images or video of the same scene from different viewpoints
2. **Enable** spatial segmentation in the interface
3. **Describe** what you want to segment using natural language
4. **Process** and view results in both 3D and 2D

## 📝 Example Usage

### Command Line

```python
from multiview_segmentation import MultiViewSegmentationPipeline

# Initialize pipeline
pipeline = MultiViewSegmentationPipeline()

# Process images
image_paths = ["view1.jpg", "view2.jpg", "view3.jpg"]
results = pipeline.run_segmentation(image_paths, "red chair on the left side")

# Save results
pipeline.save_results(results, "output_directory/")
```

### Web Interface

1. Open the Gradio interface in your browser
2. Upload multiple images of the same scene
3. Enter spatial description: `"blue car in the center"`
4. Click "Reconstruct + Segment"
5. Download results as needed

## 🎯 Spatial Prompt Examples

The system understands complex spatial descriptions:

### Colors + Positions
```
"red chair on the left side"
"blue car in the center" 
"green plant above the table"
"yellow box below the shelf"
```

### Spatial Relationships
```
"person walking in the background"
"bottle near the window"
"laptop on the right desk"
"book behind the monitor"
```

### Complex Descriptions
```
"red and white striped chair on the left"
"person wearing blue shirt in front"
"small green plant next to the computer"
```

## 🔧 System Architecture

```
Input Images/Video
       ↓
   VGGT 3D Reconstruction
       ↓
   Spatial Prompt Processing (CLIP)
       ↓
   3D Object Identification
       ↓
   3D-to-2D Projection
       ↓
   SAM2 Segmentation Refinement
       ↓
   Output Generation
```

### Core Components

1. **`MultiViewSegmentationPipeline`**: Main orchestration class
2. **`SpatialPromptProcessor`**: Handles natural language understanding
3. **VGGT Integration**: 3D reconstruction and camera pose estimation
4. **SAM2 Integration**: High-quality segmentation refinement
5. **CLIP Integration**: Semantic understanding and similarity computation

## 📊 Output Formats

### 2D Results
- **Segmentation masks**: Binary masks for each view
- **Overlay images**: Original images with colored segmentation overlays
- **Detection points**: Key points used for segmentation

### 3D Results
- **Point cloud**: 3D coordinates of segmented regions
- **PLY files**: Standard point cloud format with color information
- **Camera poses**: Estimated camera positions and orientations

### Metadata
- **Similarity scores**: CLIP similarity between prompt and each view
- **Confidence maps**: Per-pixel confidence from depth estimation
- **Processing logs**: Detailed information about the pipeline execution

## ⚙️ Configuration

Customize behavior through `multiview_config.yaml`:

```yaml
models:
  vggt:
    checkpoint_url: "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    device: "cuda"
  
  sam2:
    checkpoint_path: "./checkpoints/sam2.1_hiera_large.pt"
    config: "sam2.1_hiera_l.yaml"
  
  clip:
    model: "ViT-L/14@336px"

processing:
  max_images: 50
  frame_extraction_fps: 2
  confidence_threshold: 0.5

segmentation:
  sam2_multimask: true
  projection_radius: 20
  spatial_similarity_threshold: 0.2
```

## 🧪 Testing

Verify your installation:

```bash
python test_setup.py
```

Run examples:

```bash
python run_example.py
```

## 📁 Project Structure

```
multiview_segmentation/
├── multiview_segmentation.py           # Core pipeline
├── multiview_segmentation_interface.py # Gradio interface
├── enhanced_vggt_demo.py              # Enhanced VGGT demo
├── launch_multiview_segmentation.py   # Launcher script
├── setup_multiview_segmentation.sh    # Setup script
├── test_setup.py                      # Testing script
├── run_example.py                     # Example usage
├── multiview_config.yaml              # Configuration
├── checkpoints/                       # Model checkpoints
│   └── sam2.1_hiera_large.pt
├── examples/
│   └── multiview_data/               # Test images
└── outputs/
    └── segmentation_results/         # Output directory
```

## 🔬 Technical Details

### 3D Reconstruction (VGGT)
- Processes multiple images to estimate camera poses and depth maps
- Generates dense 3D point clouds with confidence scores
- Provides metric-scale reconstruction without external calibration

### Spatial Understanding (CLIP)
- Computes semantic similarity between text prompts and images
- Identifies relevant views containing target objects
- Handles complex spatial descriptions and object relationships

### Segmentation (SAM2)
- Refines segmentation using projected 3D points as prompts
- Generates high-quality masks with precise boundaries
- Supports both automatic and interactive segmentation modes

### Multi-View Consistency
- Projects 3D object locations back to all 2D views
- Ensures consistent segmentation across different camera angles
- Handles occlusions and partial visibility

## 🚨 Troubleshooting

### Common Issues

**CUDA not available:**
```bash
# Install CPU-only PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**SAM2 checkpoint missing:**
```bash
# Download manually
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
```

**Import errors:**
```bash
# Check Python path
export PYTHONPATH="${PWD}/vggt:${PWD}/sam2:${PWD}:${PYTHONPATH}"
```

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster processing
2. **Limit images**: Process 5-15 images for optimal performance
3. **Resize large images**: Use images ~800x600 for faster processing
4. **Clear descriptions**: Use specific spatial descriptions for better results

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

### Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/your-repo/multiview-segmentation.git

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License. See individual component licenses for VGGT, SAM2, and CLIP.

## 🙏 Acknowledgments

- **VGGT**: Meta's Visual Geometry Grounded Transformer
- **SAM2**: Meta's Segment Anything Model 2
- **CLIP**: OpenAI's Contrastive Language-Image Pre-training
- **VQASynth**: Spatial reasoning and VQA synthesis pipeline

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{multiview_spatial_segmentation,
  title={Multi-View Spatial Segmentation with VGGT and SAM2},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/multiview-segmentation}
}
```

---

**Built with ❤️ for the computer vision and robotics community**
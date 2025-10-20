# Live Spatial VQA System 🎥🧠

Real-time Visual Question Answering with Spatial Reasoning using SpaceOm and SpaceThinker models from camera streams.

## Features

- **Real-time Camera Integration**: Support for Intel RealSense D435/D455 cameras and standard webcams
- **Multiple Model Support**: Switch between SpaceOm and SpaceThinker models for different spatial reasoning tasks
- **Dual Interfaces**: 
  - CLI interface with OpenCV for direct interaction
  - Web interface with Gradio for remote access
- **Live Frame Processing**: Ask questions about what the camera sees in real-time
- **Spatial Understanding**: Advanced capabilities for distance estimation, object relationships, and spatial reasoning

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Intel RealSense D435/D455 camera (optional, webcam fallback available)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd live-spatial-vqa
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Intel RealSense SDK (for RealSense cameras):
```bash
# Ubuntu/Debian
sudo apt-get install librealsense2-dev
pip install pyrealsense2

# Or follow: https://github.com/IntelRealSense/librealsense
```

4. Download models:
   - Place SpaceOm model in `./SpaceOm/` directory
   - Place SpaceThinker model in `./SpaceThinker-Qwen2.5VL-3B/` directory

## Quick Start

### CLI Interface
```bash
python src/live_spatial_vqa.py --camera realsense
```

Controls:
- `q`: Ask a question
- `m`: Switch model
- `s`: Show statistics
- `i`: Model info
- `ESC`: Exit

### Web Interface
```bash
python src/live_spatial_vqa_gradio.py --camera realsense --port 7860
```

Then open http://localhost:7860 in your browser.

### Auto-Streaming Web Interface
```bash
python src/live_spatial_vqa_gradio_stream.py --camera realsense --port 7861
```

## Usage Examples

### Distance Estimation
```
Question: "How far is the red object from the camera?"
SpaceThinker: "The red object appears to be approximately 1.2 meters from the camera based on its apparent size and position in the frame."
```

### Spatial Relationships
```
Question: "What objects are to the left of the blue cup?"
SpaceOm: "To the left of the blue cup, I can see a white notebook and a black pen on the desk surface."
```

### Object Counting
```
Question: "How many books are visible on the shelf?"
SpaceOm: "I can count 7 books visible on the shelf, arranged vertically."
```

## Model Comparison

| Model | Best For | Strengths |
|-------|----------|-----------|
| **SpaceOm** | General spatial understanding | Object relationships, scene understanding |
| **SpaceThinker** | Quantitative analysis | Distance estimation, step-by-step reasoning |

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera        │────│  Frame Capture   │────│  Model Inference│
│ (RealSense/Web) │    │   (Threading)    │    │ (SpaceOm/Thinker│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Interface │    │   Web Interface  │    │  Auto-Streaming │
│    (OpenCV)     │    │    (Gradio)      │    │    (Gradio)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Configuration

### Camera Settings
- **Resolution**: 640x480 (default)
- **FPS**: 30 (RealSense) / Variable (Webcam)
- **Format**: RGB24

### Model Settings
- **Device**: CUDA (GPU) preferred, CPU fallback
- **Precision**: FP16 on GPU, FP32 on CPU
- **Memory**: ~7-8GB GPU memory required

## API Reference

### CameraStreamCapture
```python
camera = CameraStreamCapture(camera_type="realsense")
camera.start()
frame = camera.read_pil()  # Returns PIL Image
```

### SpatialModelLoader
```python
loader = SpatialModelLoader(model_dir=".")
loader.load_model("SpaceOm")
models = loader.list_available_models()
```

### OnlineInferenceEngine
```python
engine = OnlineInferenceEngine(model_loader)
result = engine.ask_question("What do you see?", frame)
```

## Troubleshooting

### Camera Issues
- **RealSense not detected**: Install librealsense2-dev and check USB connection
- **Permission denied**: Run `sudo chmod 666 /dev/video*`
- **Low FPS**: Check USB 3.0 connection and system resources

### Model Issues
- **CUDA out of memory**: Reduce batch size or use CPU mode
- **Model not found**: Verify model paths and file permissions
- **Slow inference**: Ensure CUDA is available and GPU has sufficient memory

### Interface Issues
- **Gradio not accessible**: Check firewall settings and port availability
- **Frames not updating**: Click refresh button or restart interface
- **Browser compatibility**: Use Chrome/Firefox for best experience

## Development

### Running Tests
```bash
python tests/test_camera_capture.py
python tests/test_model_loading.py
python tests/test_inference.py
```

### Code Structure
```
src/
├── camera_stream_capture.py      # Camera interface
├── online_spatial_inference.py   # Model loading and inference
├── live_spatial_vqa.py          # CLI interface
├── live_spatial_vqa_gradio.py   # Web interface
└── live_spatial_vqa_gradio_stream.py  # Auto-streaming interface

tests/
├── test_camera_capture.py       # Camera tests
├── test_model_loading.py        # Model tests
└── test_inference.py            # Inference tests

examples/
├── basic_usage.py               # Simple usage examples
└── advanced_config.py           # Advanced configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{live_spatial_vqa,
  title={Live Spatial VQA System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/live-spatial-vqa}
}
```

## Acknowledgments

- SpaceOm and SpaceThinker models
- Intel RealSense SDK
- Gradio framework
- PyTorch and Transformers library
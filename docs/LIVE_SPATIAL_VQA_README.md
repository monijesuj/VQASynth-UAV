# Live Spatial VQA System 🎥🤖

Real-time spatial reasoning and Visual Question Answering from camera streams using **SpaceOm** and **SpaceThinker** models.

## Features ✨

- **🎥 Live Camera Support**: RealSense D435/D455 (RGB only) or standard webcam
- **🤖 Model Selection**: Switch between SpaceOm and SpaceThinker models on-the-fly
- **💬 Interactive VQA**: Ask spatial reasoning questions about live video
- **🖥️ Multiple Interfaces**: 
  - OpenCV-based CLI with keyboard controls
  - Gradio web interface with browser access
- **⚡ Real-time Processing**: Threaded frame capture and buffering
- **📊 Statistics**: Track inference performance and timing

## Models 📦

### SpaceOm
- **Best for**: General spatial understanding and object relationships
- **Strengths**: Overall spatial reasoning capabilities

### SpaceThinker  
- **Best for**: Precise distance measurements
- **Strengths**: Quantitative reasoning, step-by-step thinking, distance estimation

## Installation 🚀

### 1. Basic Requirements

```bash
# Install dependencies
pip install -r requirements_live_vqa.txt
```

### 2. RealSense Camera Support (Optional)

```bash
# For RealSense cameras (D435/D455)
pip install pyrealsense2
```

If you don't have a RealSense camera, the system will automatically fall back to a standard webcam.

### 3. Model Setup

Download the spatial reasoning models to your workspace:

```bash
# SpaceOm
git clone https://huggingface.co/remyxai/SpaceOm

# SpaceThinker
git clone https://huggingface.co/remyxai/SpaceThinker-Qwen2.5VL-3B
```

Or use the Hugging Face CLI:

```bash
huggingface-cli download remyxai/SpaceOm --local-dir SpaceOm
huggingface-cli download remyxai/SpaceThinker-Qwen2.5VL-3B --local-dir SpaceThinker-Qwen2.5VL-3B
```

## Usage 🎮

### Option 1: OpenCV Interface (CLI)

```bash
# With RealSense camera
python live_spatial_vqa.py

# With webcam
python live_spatial_vqa.py --camera webcam

# Custom model directory
python live_spatial_vqa.py --model-dir /path/to/models
```

#### Keyboard Controls:
- **SPACE/ENTER**: Ask a question
- **'m'**: Switch model (SpaceOm ↔ SpaceThinker)
- **'s'**: Save current frame
- **'i'**: Show model information
- **'t'**: Show statistics
- **'h'**: Toggle help
- **'c'**: Clear frame history
- **'q'/ESC**: Quit

### Option 2: Gradio Web Interface

```bash
# Start web interface
python live_spatial_vqa_gradio.py

# With webcam
python live_spatial_vqa_gradio.py --camera webcam

# Custom port
python live_spatial_vqa_gradio.py --port 8080

# Create public sharing link
python live_spatial_vqa_gradio.py --share
```

Then open your browser to `http://localhost:7860` (or the specified port).

## Architecture 🏗️

### Components

1. **`camera_stream_capture.py`**: 
   - Threaded camera capture with frame buffering
   - Support for RealSense and standard webcams
   - Frame queue management

2. **`online_spatial_inference.py`**:
   - Model loading and management (SpaceOm, SpaceThinker)
   - Online inference engine with frame history
   - Performance tracking and statistics

3. **`live_spatial_vqa.py`**:
   - OpenCV-based interactive interface
   - Real-time visualization with overlays
   - Keyboard-driven interaction

4. **`live_spatial_vqa_gradio.py`**:
   - Web-based Gradio interface
   - Browser-accessible UI
   - Auto-refreshing camera feed

## Example Questions 💭

- "How far is the red object from the camera?"
- "What objects are on the table?"
- "Estimate the distance between the cup and the book."
- "Describe the spatial layout of the scene."
- "What is the height of the shelf?"
- "Which object is closest to the camera?"

## System Requirements 💻

- **OS**: Linux (tested on Ubuntu), macOS, Windows
- **Python**: 3.8+
- **GPU**: CUDA-compatible GPU recommended (8GB+ VRAM)
- **RAM**: 16GB+ recommended
- **Camera**: RealSense D435/D455 or any webcam

## Performance Tips ⚡

1. **GPU Acceleration**: Ensure CUDA is properly installed for best performance
2. **Model Selection**: 
   - Use **SpaceOm** for general spatial questions
   - Use **SpaceThinker** for precise distance measurements
3. **Frame Rate**: Adjust camera FPS if experiencing lag
4. **Resolution**: Lower resolution (640x480) provides faster inference

## Troubleshooting 🔧

### Camera Issues

```bash
# Test RealSense connection
python camera_stream_capture.py

# If RealSense fails, system auto-falls back to webcam
```

### Model Loading Issues

```bash
# Verify models exist
ls -la SpaceOm/
ls -la SpaceThinker-Qwen2.5VL-3B/

# Test model loading
python online_spatial_inference.py
```

### CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA unavailable, models will run on CPU (slower)
```

## API Usage 🔌

You can also use the components programmatically:

```python
from camera_stream_capture import CameraStreamCapture
from online_spatial_inference import SpatialModelLoader, OnlineInferenceEngine

# Initialize camera
camera = CameraStreamCapture(camera_type='realsense')
camera.start()

# Load model
loader = SpatialModelLoader()
loader.load_model("SpaceOm")

# Create inference engine
engine = OnlineInferenceEngine(loader)

# Process frames
while True:
    frame = camera.read_pil()
    if frame:
        engine.update_frame(frame)
        
        # Ask question
        result = engine.ask_question("What do you see?")
        print(result['answer'])
    
    # ... your code ...

camera.stop()
```

## Citation 📚

If you use this system in your research, please cite:

```bibtex
@software{live_spatial_vqa,
  title={Live Spatial VQA: Real-time Spatial Reasoning from Camera Streams},
  author={Your Name},
  year={2025}
}
```

Also cite the underlying models:

```bibtex
@model{spaceom,
  title={SpaceOm: Spatial Reasoning Model},
  author={RemyxAI},
  year={2024}
}

@model{spacethinker,
  title={SpaceThinker: Quantitative Spatial Reasoning with Thinking},
  author={RemyxAI},
  year={2024}
}
```

## License 📄

This project follows the same license as the underlying models (Qwen Research License).

## Support 🆘

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues on GitHub
3. Open a new issue with system details and error logs

## Acknowledgments 🙏

- **SpaceOm & SpaceThinker**: RemyxAI
- **Qwen2.5-VL**: Alibaba Cloud
- **Intel RealSense**: Intel Corporation

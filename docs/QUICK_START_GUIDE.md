# Quick Start Guide - Live Spatial VQA 🚀

## System Successfully Created! ✅

You now have a complete online spatial reasoning system with:
- ✅ RealSense camera streaming (with webcam fallback)
- ✅ SpaceOm and SpaceThinker model support
- ✅ Real-time VQA inference
- ✅ Two user interfaces (CLI and Web)

## Files Created 📁

1. **`camera_stream_capture.py`** - Camera streaming with threading
2. **`online_spatial_inference.py`** - Model loading and inference engine
3. **`live_spatial_vqa.py`** - OpenCV CLI interface
4. **`live_spatial_vqa_gradio.py`** - Web browser interface
5. **`test_system_components.py`** - Component testing
6. **`requirements_live_vqa.txt`** - Dependencies
7. **`start_live_vqa.sh`** - Quick start script
8. **`LIVE_SPATIAL_VQA_README.md`** - Full documentation

## Quick Start 🎯

### Option 1: Use the Quick Start Script

```bash
chmod +x start_live_vqa.sh
./start_live_vqa.sh
```

### Option 2: Manual Start

#### Web Interface (Recommended)
```bash
python live_spatial_vqa_gradio.py
```
Then open http://localhost:7860 in your browser.

#### CLI Interface
```bash
python live_spatial_vqa.py
```
Use keyboard controls (press 'h' for help).

### For Webcam Instead of RealSense
```bash
python live_spatial_vqa_gradio.py --camera webcam
# or
python live_spatial_vqa.py --camera webcam
```

## Usage Workflow 📋

### Gradio Web Interface:
1. **Initialize**: Click "🚀 Initialize System"
2. **Refresh**: Click "🔄 Refresh Frame" to update camera view
3. **Ask**: Type question and click "🤔 Ask Question"
4. **Switch**: Change model in dropdown if needed
5. **Monitor**: Check statistics with "📈 Show Statistics"

### CLI Interface:
1. **Start**: Run the script
2. **Question**: Press SPACE or ENTER
3. **Switch Model**: Press 'm'
4. **Help**: Press 'h'
5. **Quit**: Press 'q'

## Example Questions 💭

```
"How far is the cup from the camera?"
"What objects are on the table?"
"Estimate the distance between the red object and blue object."
"Describe the spatial layout of this scene."
"What is the approximate height of the shelf?"
"Which object is closest to me?"
```

## Model Selection 🤖

- **SpaceOm**: Best for general spatial understanding
- **SpaceThinker**: Best for precise distance measurements

## Test Components First 🧪

```bash
python test_system_components.py
```

This will verify:
- ✅ All dependencies installed
- ✅ Camera accessible
- ✅ Models available
- ✅ Modules working

## Troubleshooting 🔧

### Camera Not Found
```bash
# Check if RealSense is connected
rs-enumerate-devices

# If not, use webcam
python live_spatial_vqa_gradio.py --camera webcam
```

### Out of Memory
- Close other GPU applications
- Use only one model at a time
- Reduce camera resolution in code if needed

### Slow Inference
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Models run on GPU by default (much faster than CPU)
- First inference is slower (model warmup)

## Architecture Overview 🏗️

```
Camera Stream → Frame Buffer → Inference Engine → User Interface
     ↓               ↓                ↓                 ↓
RealSense/      Threading      SpaceOm/         CLI/Gradio
 Webcam         & Queue      SpaceThinker       Display
```

## Performance Tips ⚡

1. **GPU Required**: System needs CUDA GPU for real-time performance
2. **Model Loading**: First load takes time, subsequent inferences are fast
3. **Frame Rate**: Camera runs at 30 FPS, inference on-demand
4. **Memory**: ~8GB GPU memory per model

## Next Steps 🎯

1. ✅ Test with: `python test_system_components.py`
2. ✅ Start interface of choice
3. ✅ Initialize system in UI
4. ✅ Start asking spatial questions!

## Support 📞

- Check `LIVE_SPATIAL_VQA_README.md` for detailed documentation
- Review error messages for specific issues
- Test components individually if problems occur

---

**You're all set!** The system is ready for online spatial reasoning from camera streams. 🎉

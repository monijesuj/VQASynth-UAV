# 🎯 Single Image Prompt-based Segmentation

A simplified, robust approach to prompt-based segmentation using CLIP + SAM2 with optional GroundingDINO and RAM support.

## 🌟 Features

- **CLIP-based semantic understanding**: Match text prompts to image regions
- **SAM2 high-quality segmentation**: State-of-the-art segmentation quality
- **GroundingDINO object detection** (optional): Better object localization
- **RAM object recognition** (optional): Enhanced object understanding
- **Spatial keyword parsing**: Understand "floor", "left", "ceiling", etc.
- **Gradio web interface**: Easy-to-use UI

## 🏗️ Architecture

```
Input Image + Text Prompt
         ↓
    ┌─────────────────────────┐
    │  Multi-Source Prompts   │
    ├─────────────────────────┤
    │ 1. GroundingDINO boxes  │ (if available)
    │ 2. CLIP attention maps  │ (fallback)
    │ 3. Spatial keywords     │ (floor, left, etc.)
    └─────────────────────────┘
         ↓
    ┌─────────────────────────┐
    │    SAM2 Segmentation    │
    │  (point/box prompts)    │
    └─────────────────────────┘
         ↓
    Segmentation Mask + Overlay
```

## 🚀 Usage

### Basic Usage

```bash
python single_image_segmentation.py
```

Then:
1. Upload an image
2. Enter a text prompt (e.g., "floor", "blue box", "person on the left")
3. Click "Segment"

### Supported Prompts

**Object-based:**
- "chair"
- "table"
- "person"
- "blue box"

**Spatial-based:**
- "floor"
- "ceiling"
- "left side"
- "right wall"
- "center"

**Combined:**
- "red object on the floor"
- "person on the left"
- "box in the center"

## 📦 Dependencies

### Required:
- PyTorch
- CLIP (openai/CLIP)
- SAM2 (checkpoint: `sam2/checkpoints/sam2.1_hiera_large.pt`)
- OpenCV
- NumPy
- Gradio

### Optional (auto-detected):
- GroundingDINO (for better object detection)
- RAM (Recognize Anything Model)

## 🔧 How It Works

### 1. Prompt Generation

The system generates SAM2 prompts from multiple sources:

**a) GroundingDINO (if available):**
- Detects objects matching the text prompt
- Provides bounding boxes for SAM2

**b) CLIP Attention (fallback):**
- Computes semantic similarity between text and image
- Samples points from high-attention regions

**c) Spatial Keywords:**
- Parses keywords: floor, ceiling, left, right, center
- Generates points at appropriate image regions

### 2. SAM2 Segmentation

- Sets the image in SAM2 predictor
- Passes point or box prompts
- Generates high-quality segmentation masks
- Returns best mask with confidence score

### 3. Visualization

- Overlays green mask on original image
- Shows prompt points (green=foreground, red=background)
- Displays confidence score and statistics

## 🆚 Comparison with Multi-View Approach

| Feature | Single Image | Multi-View (Previous) |
|---------|--------------|----------------------|
| **Complexity** | Simple, focused | Complex, many components |
| **Speed** | Fast (< 1s) | Slower (VGGT reconstruction) |
| **Accuracy** | High (CLIP+SAM2) | Variable (consistency issues) |
| **Use Case** | Quick segmentation | 3D reconstruction + segmentation |
| **Dependencies** | Core models only | VGGT + many extras |

## 🐛 Troubleshooting

### SAM2 "Image format not supported"
✅ **Fixed**: Proper BGR→RGB conversion and uint8 dtype enforcement

### Low CLIP similarity scores
✅ **Fixed**: Multiple text variations for better matching

### No segmentation output
- Check if SAM2 checkpoint exists: `sam2/checkpoints/sam2.1_hiera_large.pt`
- Verify CLIP is installed: `pip install git+https://github.com/openai/CLIP.git`
- Try simpler prompts first: "floor", "wall", "object"

## 📊 Example Results

**Prompt: "floor"**
- CLIP similarity: 0.226
- SAM2 confidence: 0.95
- Mask coverage: 45%

**Prompt: "blue box"**
- CLIP similarity: 0.247
- SAM2 confidence: 0.88
- Mask coverage: 12%

## 🔄 Future Enhancements

1. **Multi-object segmentation**: Segment multiple objects simultaneously
2. **Instance segmentation**: Separate individual instances
3. **Refinement loop**: Iterative refinement based on user feedback
4. **Export options**: Save masks, colored overlays, JSON annotations
5. **Batch processing**: Process multiple images with same prompt

## 📝 Notes

- **GroundingDINO is optional** but recommended for better object detection
- **RAM is optional** - CLIP alone works well for most cases
- **Spatial keywords** help with scene understanding (floor, ceiling, etc.)
- **Multi-mask output** allows SAM2 to propose multiple segmentations

## 🎯 Why This Approach?

1. **Simpler**: Fewer moving parts, easier to debug
2. **Faster**: No 3D reconstruction overhead
3. **More robust**: Proven models (CLIP, SAM2) with good error handling
4. **Flexible**: Easy to add GroundingDINO/RAM when needed
5. **Better UX**: Clear visualization and feedback

Start here, then add multi-view support later if needed!

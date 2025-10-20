# ✅ System Verification Checklist

Use this checklist to verify your Live Spatial VQA system is working correctly.

## 1. Test Components

```bash
python test_system_components.py
```

**Expected Output:**
- ✅ All imports pass
- ✅ Camera module loads
- ✅ Inference module loads
- ✅ Interface modules load
- ✅ At least one model found

## 2. Test CLI Interface

```bash
python live_spatial_vqa.py
```

**Expected Behavior:**
- ✅ Camera initializes
- ✅ Model loads (~10-20 seconds)
- ✅ Live video window appears
- ✅ FPS counter shows ~30 FPS
- ✅ Press SPACE → Terminal asks for question
- ✅ Type question → Get answer in ~1-3 seconds
- ✅ Press 'm' → Model switches
- ✅ Press 'q' → Clean exit

## 3. Test Gradio Web Interface

```bash
python live_spatial_vqa_gradio.py
```

**Expected Behavior:**
- ✅ Opens http://localhost:7860
- ✅ Click "🚀 Initialize System" → Models load
- ✅ Click "🔄 Refresh Frame" → Camera view appears
- ✅ Type question → Click "Ask" → Get answer
- ✅ Model dropdown → Switch works
- ✅ Statistics button → Shows stats

## 4. Test Questions

Try these on both interfaces:

### Basic Questions
- [ ] "What do you see?"
- [ ] "Describe the scene."
- [ ] "What objects are visible?"

### Spatial Questions
- [ ] "What is to the left of the [object]?"
- [ ] "Which object is closest?"
- [ ] "Describe the spatial arrangement."

### Distance Questions (Best with SpaceThinker)
- [ ] "How far is the [object] from the camera?"
- [ ] "What is the distance between [A] and [B]?"
- [ ] "Estimate the height of [object]."

## 5. Test Model Switching

### SpaceOm Test
1. [ ] Select SpaceOm
2. [ ] Ask: "What objects do you see?"
3. [ ] Verify general spatial description

### SpaceThinker Test
1. [ ] Switch to SpaceThinker
2. [ ] Ask: "How far is [object] from camera?"
3. [ ] Verify distance estimate with reasoning

## 6. Performance Check

### CLI Interface
- [ ] FPS: ~30 FPS displayed
- [ ] Inference time: 1-3 seconds
- [ ] GPU memory: ~7-8 GB
- [ ] No lag in video

### Gradio Interface
- [ ] Frame refresh works
- [ ] Answer appears quickly
- [ ] No error messages
- [ ] Statistics accurate

## 7. Error Handling

### Test Graceful Failures
- [ ] Ask empty question → Error message
- [ ] Refresh before init → No crash
- [ ] Switch to unavailable model → Error message
- [ ] Disconnect camera → Fallback message

## 8. Camera Tests

### RealSense
```bash
python live_spatial_vqa.py
```
- [ ] RealSense detected
- [ ] RGB stream works
- [ ] 30 FPS achieved

### Webcam Fallback
```bash
python live_spatial_vqa.py --camera webcam
```
- [ ] Webcam detected
- [ ] Stream works
- [ ] Questions answered

## 9. Documentation Check

- [ ] README exists and is clear
- [ ] Quick start guide helpful
- [ ] Examples work as described
- [ ] Troubleshooting section useful

## 10. Integration Test

**Full Workflow:**
1. [ ] Start system
2. [ ] Initialize models
3. [ ] Capture frame
4. [ ] Ask 3 different questions
5. [ ] Switch model
6. [ ] Ask 2 more questions
7. [ ] Check statistics
8. [ ] Clean exit

## Troubleshooting Results

If any test fails, check:

### Camera Issues
```bash
# Test camera independently
python camera_stream_capture.py

# Try webcam
python live_spatial_vqa.py --camera webcam
```

### Model Issues
```bash
# Verify models exist
ls -la SpaceOm/
ls -la SpaceThinker-Qwen2.5VL-3B/

# Test model loading
python demo_without_camera.py
```

### Performance Issues
```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU memory
nvidia-smi
```

## ✅ System Status

Once all tests pass:
- ✅ System is fully functional
- ✅ Ready for production use
- ✅ All features working
- ✅ Error handling robust

## 📊 Expected Performance Metrics

| Metric | Expected Value |
|--------|---------------|
| Camera FPS | 28-30 FPS |
| Inference Time | 1-3 seconds |
| GPU Memory | 7-8 GB |
| Model Load Time | 10-20 seconds |
| Frame Latency | < 100ms |

## 🎯 Success Criteria

Your system is working perfectly when:
1. ✅ Camera streams at 30 FPS
2. ✅ Both models load successfully
3. ✅ Questions get accurate answers
4. ✅ Model switching works smoothly
5. ✅ No crashes or errors during normal use
6. ✅ Both interfaces (CLI & Web) function
7. ✅ Performance matches expectations

---

**All checks passed?** 🎉

Your Live Spatial VQA system is **production ready**!

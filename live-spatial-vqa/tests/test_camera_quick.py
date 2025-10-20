#!/usr/bin/env python3
"""
Quick test - verify camera and frame capture work
"""
from camera_stream_capture import CameraStreamCapture
import time

print("🧪 Testing camera capture...")

# Test RealSense
camera = CameraStreamCapture(camera_type='realsense', width=640, height=480, fps=30)

if camera.start():
    print("✅ Camera started")
    
    # Wait for frames
    time.sleep(2.0)
    
    # Try to read a frame
    for i in range(5):
        frame_bgr = camera.read()
        frame_pil = camera.read_pil()
        
        print(f"Attempt {i+1}:")
        print(f"  BGR frame: {type(frame_bgr)}, {frame_bgr.shape if frame_bgr is not None else 'None'}")
        print(f"  PIL frame: {type(frame_pil)}, {frame_pil.size if frame_pil is not None else 'None'}")
        
        time.sleep(0.5)
    
    camera.stop()
    print("✅ Test complete")
else:
    print("❌ Camera failed to start")

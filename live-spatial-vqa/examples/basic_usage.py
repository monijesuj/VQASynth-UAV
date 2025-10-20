#!/usr/bin/env python3
"""
Basic usage example for Live Spatial VQA System
Simple demonstration of camera capture and model inference
"""

from src.camera_stream_capture import CameraStreamCapture
from src.online_spatial_inference import SpatialModelLoader, OnlineInferenceEngine
import time


def main():
    """Basic usage example"""
    
    print("🚀 Live Spatial VQA - Basic Example")
    print("=" * 40)
    
    # Initialize camera
    print("📹 Initializing camera...")
    camera = CameraStreamCapture(camera_type="realsense")  # or "webcam"
    
    if not camera.start():
        print("❌ Failed to start camera")
        return
    
    print("✅ Camera started successfully")
    
    # Initialize models
    print("\n🧠 Loading models...")
    model_loader = SpatialModelLoader(model_dir=".")
    available_models = model_loader.list_available_models()
    
    if not available_models:
        print("❌ No models found")
        camera.stop()
        return
    
    print(f"📦 Available models: {available_models}")
    
    # Load first available model
    model_name = available_models[0]
    print(f"🔄 Loading {model_name}...")
    
    if not model_loader.load_model(model_name):
        print(f"❌ Failed to load {model_name}")
        camera.stop()
        return
    
    print(f"✅ {model_name} loaded successfully")
    
    # Initialize inference engine
    inference_engine = OnlineInferenceEngine(model_loader)
    
    # Wait for camera to warm up
    print("\n⏳ Warming up camera...")
    time.sleep(2.0)
    
    # Capture frame and ask question
    print("\n📸 Capturing frame...")
    frame = camera.read_pil()
    
    if frame is None:
        print("❌ Failed to capture frame")
        camera.stop()
        return
    
    print(f"✅ Frame captured: {frame.size}")
    
    # Ask a question
    question = "What objects can you see in this image?"
    print(f"\n❓ Question: {question}")
    print("🤔 Processing...")
    
    try:
        result = inference_engine.ask_question(question, frame)
        
        if result['success']:
            print(f"\n💡 Answer: {result['answer']}")
            print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
            print(f"🎯 Model used: {result['model']}")
        else:
            print(f"❌ Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up...")
    camera.stop()
    print("✅ Done!")


if __name__ == "__main__":
    main()
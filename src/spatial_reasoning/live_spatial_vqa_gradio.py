#!/usr/bin/env python3
"""
Gradio Web Interface for Live Spatial VQA
Browser-based interface with camera stream and interactive questioning
"""
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import threading
import time
from typing import Optional, Tuple

from camera_stream_capture import CameraStreamCapture
from online_spatial_inference import SpatialModelLoader, OnlineInferenceEngine


class GradioLiveSpatialVQA:
    """Gradio-based web interface for live spatial VQA"""
    
    def __init__(self, model_dir: str = ".", camera_type: str = "realsense"):
        """
        Initialize Gradio interface
        
        Args:
            model_dir: Directory containing model folders
            camera_type: 'realsense' or 'webcam'
        """
        self.model_dir = model_dir
        self.camera_type = camera_type
        
        # Components
        self.camera = None
        self.model_loader = None
        self.inference_engine = None
        
        # State
        self.camera_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
    def initialize_system(self) -> Tuple[str, str]:
        """Initialize camera and models"""
        try:
            # Initialize model loader
            self.model_loader = SpatialModelLoader(self.model_dir)
            available_models = self.model_loader.list_available_models()
            
            if not available_models:
                return "❌ No models found", ""
            
            # Load first available model
            if not self.model_loader.load_model(available_models[0]):
                return "❌ Failed to load model", ""
            
            # Initialize inference engine
            self.inference_engine = OnlineInferenceEngine(self.model_loader)
            
            # Initialize camera
            self.camera = CameraStreamCapture(
                camera_type=self.camera_type,
                width=640,
                height=480,
                fps=30
            )
            
            if not self.camera.start():
                return "❌ Failed to start camera", ""
            
            # Start frame capture thread
            self.camera_running = True
            threading.Thread(target=self._capture_loop, daemon=True).start()
            
            # Give camera time to warm up and capture frames
            print("⏳ Waiting for camera frames...")
            
            # Wait up to 5 seconds for first frame
            max_wait = 5.0
            wait_start = time.time()
            while self.current_frame is None and (time.time() - wait_start) < max_wait:
                time.sleep(0.2)
                
            if self.current_frame is not None:
                print("✅ Camera frames ready")
            else:
                print("⚠️  Warning: No frames captured yet. Try clicking 'Refresh Frame' after initialization.")
            
            model_info = f"✅ Loaded: {available_models[0]}"
            status = f"✅ System initialized\n📦 Available models: {', '.join(available_models)}\n📹 Camera frames: {'Ready' if self.current_frame else 'Waiting...'}"
            
            return status, model_info
            
        except Exception as e:
            return f"❌ Initialization failed: {str(e)}", ""
    
    def _capture_loop(self):
        """Continuous frame capture loop"""
        # Wait a bit for camera to be fully ready
        time.sleep(0.5)
        
        frame_count = 0
        last_report = time.time()
        
        while self.camera_running:
            if self.camera is None:
                time.sleep(0.1)
                continue
                
            try:
                frame_pil = self.camera.read_pil()
                
                if frame_pil and isinstance(frame_pil, Image.Image):
                    with self.frame_lock:
                        self.current_frame = frame_pil
                    
                    frame_count += 1
                    
                    # Report every 5 seconds
                    if time.time() - last_report > 5.0:
                        print(f"📊 Captured {frame_count} frames, current size: {frame_pil.size}")
                        last_report = time.time()
                    
                    # Update inference engine
                    if self.inference_engine:
                        self.inference_engine.update_frame(frame_pil)
                else:
                    # Camera not ready yet, wait a bit
                    time.sleep(0.1)
                    continue
                    
            except Exception as e:
                # Suppress repeated errors, just wait
                time.sleep(0.1)
                continue
            
            time.sleep(0.033)  # ~30 FPS
    
    def get_current_frame(self) -> Optional[Image.Image]:
        """Get current camera frame"""
        with self.frame_lock:
            if self.current_frame:
                print(f"📸 Returning frame: {self.current_frame.size}")
                return self.current_frame.copy()
            else:
                print("⚠️  No frame available in get_current_frame")
                return None
    
    def switch_model(self, model_name: str) -> str:
        """Switch to different model"""
        if not self.model_loader:
            return "❌ System not initialized"
        
        try:
            if self.model_loader.switch_model(model_name):
                info = self.model_loader.get_model_info(model_name)
                return f"✅ Switched to {model_name}\n{info.get('description', '')}"
            else:
                return f"❌ Failed to switch to {model_name}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def ask_question(self, question: str) -> Tuple[str, str]:
        """Process question and return answer"""
        if not self.inference_engine:
            return "❌ System not initialized", ""
        
        if not question or not question.strip():
            return "❓ Please enter a question", ""
        
        # Check if we have a current frame
        current_frame = self.get_current_frame()
        if current_frame is None:
            return "❌ No camera frame available. Click 'Refresh Frame' first!", ""
        
        try:
            # Use the current frame explicitly
            result = self.inference_engine.ask_question(
                question, 
                use_latest=True,
                frame=current_frame
            )
            
            if result['success']:
                answer = result['answer']
                metadata = (
                    f"✅ Model: {result['model']}\n"
                    f"⏱️ Time: {result['inference_time']:.2f}s\n"
                    f"📸 Frame captured"
                )
                return answer, metadata
            else:
                return f"❌ {result.get('error', 'Unknown error')}", ""
                
        except Exception as e:
            return f"❌ Error: {str(e)}", ""
    
    def get_stats(self) -> str:
        """Get inference statistics"""
        if not self.inference_engine:
            return "❌ System not initialized"
        
        stats = self.inference_engine.get_stats()
        
        fps = self.camera.get_fps() if self.camera else 0.0
        
        output = "📊 STATISTICS\n" + "=" * 40 + "\n"
        output += f"Camera FPS: {fps:.1f}\n"
        output += f"Total Inferences: {stats['total_inferences']}\n"
        output += f"Average Time: {stats['average_time']:.2f}s\n"
        output += f"Frames in History: {stats['frames_in_history']}\n"
        
        return output
    
    def get_model_info(self, model_name: str) -> str:
        """Get detailed model information"""
        if not self.model_loader:
            return "❌ System not initialized"
        
        info = self.model_loader.get_model_info(model_name)
        
        if not info:
            return f"❌ Model {model_name} not found"
        
        output = f"📦 {info['name']}\n" + "=" * 40 + "\n"
        output += f"Description: {info['description']}\n"
        output += f"Strengths: {', '.join(info['strengths'])}\n"
        output += f"Loaded: {'✅ Yes' if info['loaded'] else '❌ No'}\n"
        
        if info['loaded']:
            output += f"Device: {info.get('device', 'N/A')}\n"
            if 'memory_gb' in info:
                output += f"GPU Memory: {info['memory_gb']:.2f} GB\n"
        
        return output
    
    def cleanup(self):
        """Cleanup resources"""
        self.camera_running = False
        
        if self.camera:
            self.camera.stop()
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        with gr.Blocks(title="Live Spatial VQA", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🎥 Live Spatial VQA System
            Real-time spatial reasoning from camera streams using SpaceOm and SpaceThinker
            
            **Note**: Click 'Refresh Frame' to update the camera view before asking questions.
            """)
            
            with gr.Row():
                # Left column: Camera and controls
                with gr.Column(scale=1):
                    gr.Markdown("### 📹 Camera Stream")
                    
                    camera_image = gr.Image(
                        label="Live Camera Feed (Click Refresh to update)",
                        type="pil",
                        interactive=False,
                        height=400
                    )
                    
                    with gr.Row():
                        init_btn = gr.Button("🚀 Initialize System", variant="primary")
                        refresh_btn = gr.Button("🔄 Refresh Frame", variant="secondary")
                    
                    status_text = gr.Textbox(
                        label="System Status",
                        lines=3,
                        interactive=False
                    )
                
                # Right column: Questions and answers
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 💭 Ask Questions
                    **Important**: Click '🔄 Refresh Frame' before asking questions to ensure you're using the latest camera view!
                    """)
                    
                    model_selector = gr.Dropdown(
                        choices=["SpaceOm", "SpaceThinker"],
                        value="SpaceOm",
                        label="Select Model",
                        interactive=True
                    )
                    
                    model_info_text = gr.Textbox(
                        label="Current Model",
                        lines=2,
                        interactive=False
                    )
                    
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="E.g., How far is the red object from the camera?",
                        lines=2
                    )
                    
                    ask_btn = gr.Button("🤔 Ask Question", variant="primary")
                    
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=5,
                        interactive=False
                    )
                    
                    metadata_output = gr.Textbox(
                        label="Metadata",
                        lines=2,
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Statistics")
                    stats_btn = gr.Button("📈 Show Statistics")
                    stats_output = gr.Textbox(
                        label="Statistics",
                        lines=6,
                        interactive=False
                    )
                
                with gr.Column():
                    gr.Markdown("### 📦 Model Details")
                    model_info_btn = gr.Button("ℹ️ Show Model Info")
                    model_details_output = gr.Textbox(
                        label="Model Information",
                        lines=6,
                        interactive=False
                    )
            
            # Event handlers
            def init_and_show_frame():
                status, info = self.initialize_system()
                frame = self.get_current_frame()
                return status, info, frame
            
            init_btn.click(
                fn=init_and_show_frame,
                outputs=[status_text, model_info_text, camera_image]
            )
            
            refresh_btn.click(
                fn=self.get_current_frame,
                outputs=camera_image
            )
            
            model_selector.change(
                fn=self.switch_model,
                inputs=model_selector,
                outputs=model_info_text
            )
            
            ask_btn.click(
                fn=self.ask_question,
                inputs=question_input,
                outputs=[answer_output, metadata_output]
            )
            
            stats_btn.click(
                fn=self.get_stats,
                outputs=stats_output
            )
            
            model_info_btn.click(
                fn=lambda: self.get_model_info(
                    self.model_loader.current_model_name if self.model_loader else ""
                ),
                outputs=model_details_output
            )
        
        return interface


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gradio Web Interface for Live Spatial VQA")
    
    parser.add_argument(
        "--camera",
        type=str,
        default="realsense",
        choices=["realsense", "webcam"],
        help="Camera type (default: realsense)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default=".",
        help="Model directory (default: current directory)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web interface (default: 7860)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public sharing link"
    )
    
    args = parser.parse_args()
    
    # Create application
    app = GradioLiveSpatialVQA(
        model_dir=args.model_dir,
        camera_type=args.camera
    )
    
    # Create and launch interface
    interface = app.create_interface()
    
    print("🌐 Starting Gradio web interface...")
    print(f"   Camera: {args.camera}")
    print(f"   Model directory: {args.model_dir}")
    print(f"   Port: {args.port}")
    
    try:
        interface.launch(
            server_port=args.port,
            share=args.share,
            inbrowser=True
        )
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()

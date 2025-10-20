#!/usr/bin/env python3
"""
Gradio Web Interface for Live Spatial VQA with Manual Refresh
Browser-based interface with manual camera stream updates
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


class GradioManualSpatialVQA:
    """Gradio-based web interface with manual refresh for live spatial VQA"""
    
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
        self.frame_update_thread = None
        
    def initialize_system(self) -> Tuple[str, str]:
        """Initialize camera and models"""
        try:
            # Initialize model loader
            self.model_loader = SpatialModelLoader(self.model_dir)
            available_models = self.model_loader.list_available_models()
            
            if not available_models:
                return "❌ No models found", ""
            
            # Initialize camera
            self.camera = CameraStreamCapture(camera_type=self.camera_type)
            
            # Start camera
            if not self.camera.start():
                return "❌ Failed to start camera", ""
            
            # Load default model (SpaceOm)
            if "SpaceOm" in available_models:
                self.model_loader.load_model("SpaceOm")
                current_model = "SpaceOm"
            else:
                # Load first available model
                first_model = available_models[0]
                self.model_loader.load_model(first_model)
                current_model = first_model
            
            # Initialize inference engine
            self.inference_engine = OnlineInferenceEngine(self.model_loader)
            
            # Start frame update thread
            self.camera_running = True
            self.frame_update_thread = threading.Thread(target=self._frame_update_loop, daemon=True)
            self.frame_update_thread.start()
            
            # Give camera time to warm up
            time.sleep(1.0)
            
            status = f"✅ System initialized successfully!\n"
            status += f"📹 Camera: Started\n"
            status += f"🧠 Model: {current_model} loaded\n"
            status += f"🎯 Available models: {', '.join(available_models)}\n"
            status += f"💡 Click 'Refresh Frame' to update camera view before asking questions"
            
            return status, current_model
            
        except Exception as e:
            return f"❌ Initialization failed: {str(e)}", ""
    
    def _frame_update_loop(self):
        """Background thread to continuously update frames"""
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
                    
                    # Report every 10 seconds
                    if time.time() - last_report > 10.0:
                        print(f"📊 Captured {frame_count} frames")
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
        """Get current camera frame for display"""
        with self.frame_lock:
            if self.current_frame:
                return self.current_frame.copy()
            else:
                # Return a placeholder if no frame available
                placeholder = Image.new('RGB', (640, 480), color=(128, 128, 128))
                return placeholder
    
    def switch_model(self, model_name: str) -> str:
        """Switch to different model"""
        if not self.model_loader:
            return "❌ System not initialized"
        
        try:
            self.model_loader.load_model(model_name)
            return f"✅ Switched to {model_name}"
        except Exception as e:
            return f"❌ Failed to switch model: {str(e)}"
    
    def ask_question(self, question: str) -> Tuple[str, str]:
        """Process question with current frame"""
        if not question.strip():
            return "Please enter a question.", ""
        
        if not self.inference_engine:
            return "❌ System not initialized", ""
        
        try:
            # Process question using ask_question method
            result = self.inference_engine.ask_question(question)
            
            if result['success']:
                answer = result['answer']
                metadata = f"Model: {result.get('model', 'Unknown')}\n"
                metadata += f"Processing time: {result.get('inference_time', 0):.2f}s\n"
                metadata += f"Timestamp: {time.strftime('%H:%M:%S')}"
                
                return answer, metadata
            else:
                return f"❌ Error: {result.get('error', 'Unknown error')}", ""
                
        except Exception as e:
            return f"❌ Processing failed: {str(e)}", ""
    
    def get_stats(self) -> str:
        """Get system statistics"""
        if not self.inference_engine:
            return "❌ System not initialized"
        
        stats = self.inference_engine.get_stats()
        
        output = "📊 System Statistics\n" + "=" * 30 + "\n"
        output += f"Questions processed: {stats['total_questions']}\n"
        output += f"Successful responses: {stats['successful_responses']}\n"
        output += f"Failed responses: {stats['failed_responses']}\n"
        output += f"Average processing time: {stats['avg_processing_time']:.2f}s\n"
        output += f"Current model: {stats['current_model']}\n"
        
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
        
        if self.frame_update_thread:
            self.frame_update_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.stop()
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        with gr.Blocks(title="Live Spatial VQA - Manual Refresh", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🎥 Live Spatial VQA System
            Real-time spatial reasoning from camera streams using SpaceOm and SpaceThinker
            
            **📌 Instructions:**
            1. Click "🚀 Initialize System" to start camera and load models
            2. Click "🔄 Refresh Frame" to update camera view before asking questions
            3. Select your preferred model and ask questions about what you see
            """)
            
            with gr.Row():
                # Left column: Camera and controls
                with gr.Column(scale=1):
                    gr.Markdown("### 📹 Camera Stream")
                    
                    camera_image = gr.Image(
                        label="Live Camera Feed",
                        type="pil",
                        interactive=False,
                        height=400
                    )
                    
                    with gr.Row():
                        init_btn = gr.Button("🚀 Initialize System", variant="primary")
                        refresh_btn = gr.Button("🔄 Refresh Frame", variant="secondary")
                    
                    status_text = gr.Textbox(
                        label="System Status",
                        lines=4,
                        interactive=False
                    )
                
                # Right column: Questions and answers
                with gr.Column(scale=1):
                    gr.Markdown("### 💭 Ask Questions")
                    
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
                    
                    with gr.Row():
                        stats_btn = gr.Button("📊 Statistics")
                        model_info_btn = gr.Button("ℹ️ Model Info")
                    
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=4,
                        interactive=False
                    )
                    
                    metadata_output = gr.Textbox(
                        label="Processing Details",
                        lines=3,
                        interactive=False
                    )
                    
                    stats_output = gr.Textbox(
                        label="System Statistics",
                        lines=8,
                        interactive=False
                    )
                    
                    model_details_output = gr.Textbox(
                        label="Model Details",
                        lines=6,
                        interactive=False
                    )
            
            # Event handlers
            init_btn.click(
                fn=self.initialize_system,
                outputs=[status_text, model_info_text]
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
    
    parser = argparse.ArgumentParser(description="Gradio Manual Refresh Web Interface for Live Spatial VQA")
    
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
        default=7862,
        help="Port for web interface (default: 7862)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public sharing link"
    )
    
    args = parser.parse_args()
    
    # Create application
    app = GradioManualSpatialVQA(
        model_dir=args.model_dir,
        camera_type=args.camera
    )
    
    # Create and launch interface
    interface = app.create_interface()
    
    print("🌐 Starting Gradio manual refresh web interface...")
    print(f"   Camera: {args.camera}")
    print(f"   Model directory: {args.model_dir}")
    print(f"   Port: {args.port}")
    print("   📌 Manual refresh mode - click 'Refresh Frame' to update camera view")
    
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
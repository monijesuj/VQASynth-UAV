#!/usr/bin/env python3
"""
Live Spatial VQA System
Real-time spatial reasoning from camera streams with model selection
"""
import cv2
import argparse
import sys
import numpy as np
from pathlib import Path

from ..live_vqa.camera_stream_capture import CameraStreamCapture
from .online_spatial_inference import SpatialModelLoader, OnlineInferenceEngine


class LiveSpatialVQA:
    """Interactive live spatial VQA system"""
    
    def __init__(self, model_dir: str = ".", camera_type: str = "realsense"):
        """
        Initialize live VQA system
        
        Args:
            model_dir: Directory containing model folders
            camera_type: 'realsense' or 'webcam'
        """
        self.model_dir = model_dir
        self.camera_type = camera_type
        
        # Initialize components
        self.camera = None
        self.model_loader = None
        self.inference_engine = None
        
        # UI state
        self.current_question = ""
        self.last_answer = ""
        self.show_help = False
        
    def initialize(self) -> bool:
        """Initialize all components"""
        print("🚀 Initializing Live Spatial VQA System")
        print("=" * 60)
        
        # Initialize model loader
        print("\n📦 Setting up models...")
        self.model_loader = SpatialModelLoader(self.model_dir)
        
        available_models = self.model_loader.list_available_models()
        
        if not available_models:
            print("❌ No spatial reasoning models found!")
            print(f"💡 Please ensure SpaceOm or SpaceThinker-Qwen2.5VL-3B")
            print(f"   folders exist in: {self.model_dir}")
            return False
        
        print(f"✅ Found models: {', '.join(available_models)}")
        
        # Load first available model
        print(f"\n🔄 Loading {available_models[0]}...")
        if not self.model_loader.load_model(available_models[0]):
            print("❌ Failed to load model")
            return False
        
        # Initialize inference engine
        self.inference_engine = OnlineInferenceEngine(self.model_loader)
        
        # Initialize camera
        print(f"\n🎥 Starting {self.camera_type} camera...")
        self.camera = CameraStreamCapture(
            camera_type=self.camera_type,
            width=640,
            height=480,
            fps=30
        )
        
        if not self.camera.start():
            print("❌ Failed to start camera")
            return False
        
        print("\n✅ System ready!")
        return True
    
    def print_help(self):
        """Print help information"""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║                    LIVE SPATIAL VQA CONTROLS                  ║
╠══════════════════════════════════════════════════════════════╣
║  'q' or ESC    - Quit application                             ║
║  'h'           - Toggle this help menu                        ║
║  's'           - Save current frame                           ║
║  'm'           - Switch model (SpaceOm ↔ SpaceThinker)       ║
║  'i'           - Show model info                              ║
║  't'           - Show inference statistics                    ║
║  'c'           - Clear frame history                          ║
║  SPACE or ENTER - Trigger question input                      ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(help_text)
    
    def get_question_input(self) -> str:
        """Get question from user via terminal input"""
        print("\n" + "=" * 60)
        print("💭 Ask a spatial reasoning question (or 'skip' to cancel):")
        question = input("❓ Question: ").strip()
        
        if question.lower() in ['skip', 'cancel', '']:
            return ""
        
        return question
    
    def render_frame(self, frame, fps: float):
        """Render frame with overlay information"""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        
        try:
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Validate dimensions
            if h <= 0 or w <= 0 or len(display_frame.shape) != 3:
                return None
            
            # Draw semi-transparent overlay at top
            overlay = display_frame.copy()
            if not isinstance(overlay, np.ndarray):
                return None
                
            cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
            
            # Current model
            model_name = self.model_loader.current_model_name or "None"
            cv2.putText(display_frame, f"Model: {model_name}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Camera type
            cv2.putText(display_frame, f"Camera: {self.camera_type.upper()}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Help indicator
            cv2.putText(display_frame, "Press 'h' for help", (w - 200, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Last answer overlay (bottom)
            if self.last_answer:
                # Wrap text
                max_width = 80
                answer_lines = []
                words = self.last_answer.split()
                current_line = ""
                
                for word in words:
                    test_line = f"{current_line} {word}".strip()
                    if len(test_line) <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            answer_lines.append(current_line)
                        current_line = word
                
                if current_line:
                    answer_lines.append(current_line)
                
                # Show up to 3 lines
                answer_lines = answer_lines[:3]
                
                # Draw answer box
                box_height = len(answer_lines) * 25 + 20
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, h - box_height), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                
                # Draw text
                y_offset = h - box_height + 20
                for line in answer_lines:
                    cv2.putText(display_frame, line, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 25
            
            return display_frame
            
        except Exception as e:
            # If any rendering fails, return None
            return None
    
    def switch_model(self):
        """Switch between available models"""
        available = self.model_loader.list_available_models()
        
        if len(available) < 2:
            print("⚠️  Only one model available, cannot switch")
            return
        
        current = self.model_loader.current_model_name
        current_idx = available.index(current) if current in available else -1
        next_idx = (current_idx + 1) % len(available)
        next_model = available[next_idx]
        
        print(f"\n🔄 Switching from {current} to {next_model}...")
        
        if self.model_loader.switch_model(next_model):
            print(f"✅ Switched to {next_model}")
        else:
            print(f"❌ Failed to switch to {next_model}")
    
    def show_model_info(self):
        """Display current model information"""
        model_name = self.model_loader.current_model_name
        
        if not model_name:
            print("⚠️  No model loaded")
            return
        
        info = self.model_loader.get_model_info(model_name)
        
        print("\n" + "=" * 60)
        print(f"📊 MODEL INFORMATION: {model_name}")
        print("=" * 60)
        
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("=" * 60)
    
    def show_stats(self):
        """Display inference statistics"""
        stats = self.inference_engine.get_stats()
        
        print("\n" + "=" * 60)
        print("📈 INFERENCE STATISTICS")
        print("=" * 60)
        
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        print("=" * 60)
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            print("❌ Initialization failed")
            return 1
        
        self.print_help()
        
        print("\n▶️  Starting live VQA system...")
        print("   Press SPACE or ENTER to ask questions")
        print("   Press 'q' or ESC to quit\n")
        
        try:
            while True:
                # Get current frame
                frame_bgr = self.camera.read()
                
                if frame_bgr is not None and isinstance(frame_bgr, np.ndarray) and frame_bgr.size > 0:
                    # Update inference engine with RGB frame
                    frame_rgb = self.camera.read_pil()
                    if frame_rgb:
                        self.inference_engine.update_frame(frame_rgb)
                    
                    # Render display frame
                    fps = self.camera.get_fps()
                    display_frame = self.render_frame(frame_bgr, fps)
                    
                    if display_frame is not None:
                        cv2.imshow('Live Spatial VQA', display_frame)
                else:
                    # Show waiting message if no frame yet
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "Waiting for camera frames...", (150, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow('Live Spatial VQA', blank)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # q or ESC
                    break
                
                elif key == ord('h'):
                    self.print_help()
                
                elif key == ord('s'):  # Save frame
                    if frame_bgr is not None:
                        import time
                        filename = f"capture_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame_bgr)
                        print(f"💾 Saved: {filename}")
                
                elif key == ord('m'):  # Switch model
                    self.switch_model()
                
                elif key == ord('i'):  # Model info
                    self.show_model_info()
                
                elif key == ord('t'):  # Statistics
                    self.show_stats()
                
                elif key == ord('c'):  # Clear history
                    self.inference_engine.clear_history()
                    print("🗑️  Cleared frame history")
                
                elif key == ord(' ') or key == 13:  # SPACE or ENTER
                    question = self.get_question_input()
                    
                    if question:
                        print(f"🤔 Asking: {question}")
                        print("⏳ Processing...")
                        
                        result = self.inference_engine.ask_question(question)
                        
                        if result['success']:
                            self.last_answer = result['answer']
                            print(f"\n✅ Answer: {result['answer']}")
                            print(f"⏱️  Time: {result['inference_time']:.2f}s")
                        else:
                            print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                        
                        print("\n▶️  Ready for next question...")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        
        finally:
            self.cleanup()
        
        return 0
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        
        if self.camera:
            self.camera.stop()
        
        cv2.destroyAllWindows()
        
        print("✅ Cleanup complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Live Spatial VQA System - Real-time spatial reasoning from camera streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use RealSense camera with models in current directory
  python live_spatial_vqa.py
  
  # Use webcam instead of RealSense
  python live_spatial_vqa.py --camera webcam
  
  # Specify custom model directory
  python live_spatial_vqa.py --model-dir /path/to/models
        """
    )
    
    parser.add_argument(
        "--camera",
        type=str,
        default="realsense",
        choices=["realsense", "webcam"],
        help="Camera type to use (default: realsense)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default=".",
        help="Directory containing model folders (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Create and run application
    app = LiveSpatialVQA(
        model_dir=args.model_dir,
        camera_type=args.camera
    )
    
    sys.exit(app.run())


if __name__ == "__main__":
    main()

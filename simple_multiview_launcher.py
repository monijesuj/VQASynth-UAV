#!/usr/bin/env python3
"""
Simple Launcher for Multi-View Segmentation

This script provides a working launch interface without problematic dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the Python environment and paths."""
    current_dir = Path(__file__).parent.absolute()
    
    # Add necessary paths
    vggt_path = current_dir / "vggt"
    sam2_path = current_dir / "sam2" 
    vqasynth_path = current_dir / "vqasynth"
    
    # Add to Python path
    paths_to_add = [str(vggt_path), str(sam2_path), str(vqasynth_path)]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Set environment variables
    os.environ["PYTHONPATH"] = ":".join(paths_to_add) + ":" + os.environ.get("PYTHONPATH", "")
    
    print(f"✅ Environment configured")
    return True

def check_basic_deps():
    """Check basic dependencies."""
    print("🔍 Checking basic dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA: {torch.cuda.get_device_name()}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        import gradio
        print(f"✅ Gradio {gradio.__version__}")
    except ImportError:
        print("❌ Gradio not available")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not available") 
        return False
    
    return True

def launch_basic_multiview_demo():
    """Launch a basic multiview segmentation demo."""
    print("🚀 Launching Basic Multi-View Segmentation Demo...")
    
    try:
        # Create a simple demo without problematic dependencies
        import gradio as gr
        import numpy as np
        import cv2
        from PIL import Image
        import tempfile
        import shutil
        from datetime import datetime
        
        class SimpleMultiViewDemo:
            def __init__(self):
                self.temp_dirs = []
            
            def process_upload(self, video, images):
                """Handle file uploads."""
                if not video and not images:
                    return None, "Please upload files", []
                
                # Create temp directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_dir = f"temp_demo_{timestamp}"
                os.makedirs(temp_dir, exist_ok=True)
                self.temp_dirs.append(temp_dir)
                
                image_paths = []
                
                # Handle video
                if video:
                    video_path = video if isinstance(video, str) else video.name
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_interval = max(1, int(fps))
                    
                    frame_count = 0
                    extracted = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_count % frame_interval == 0 and extracted < 10:
                            img_path = os.path.join(temp_dir, f"frame_{extracted:03d}.jpg")
                            cv2.imwrite(img_path, frame)
                            image_paths.append(img_path)
                            extracted += 1
                        frame_count += 1
                    cap.release()
                
                # Handle images
                if images:
                    for i, img_file in enumerate(images):
                        img_path = img_file if isinstance(img_file, str) else img_file.name
                        if os.path.exists(img_path):
                            dst_path = os.path.join(temp_dir, f"image_{i:03d}.jpg")
                            img = cv2.imread(img_path)
                            cv2.imwrite(dst_path, img)
                            image_paths.append(dst_path)
                
                status = f"Processed {len(image_paths)} images"
                return temp_dir, status, image_paths
            
            def simple_segmentation(self, temp_dir, prompt):
                """Perform simple segmentation demo."""
                if not temp_dir or not os.path.exists(temp_dir):
                    return [], "No images to process"
                
                if not prompt.strip():
                    return [], "Please enter a description"
                
                # Get images
                image_paths = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                              if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                results = []
                
                # Simple color-based segmentation demo
                for img_path in image_paths:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Create demo mask based on prompt
                    mask = self.create_demo_mask(img, prompt)
                    
                    # Create overlay using safe method
                    overlay = self.create_safe_overlay(img, mask)
                    
                    # Save result
                    result_path = img_path.replace('.jpg', '_segmented.jpg')
                    cv2.imwrite(result_path, overlay)
                    results.append(result_path)
                
                status = f"Processed {len(results)} images with prompt: '{prompt}'"
                return results, status
            
            def create_safe_overlay(self, img, mask):
                """Create overlay with safe array operations."""
                try:
                    if img is None or mask is None:
                        return img if img is not None else np.zeros((100, 100, 3), dtype=np.uint8)
                    
                    overlay = img.copy()
                    # Simple green overlay - just modify the green channel
                    if np.any(mask):
                        # Set green channel to 255 where mask is True
                        overlay[mask] = [0, 255, 0]  # BGR format: Blue=0, Green=255, Red=0
                    return overlay
                except Exception as e:
                    print(f"Error creating overlay: {e}")
                    return img if img is not None else np.zeros((100, 100, 3), dtype=np.uint8)
            
            def create_demo_mask(self, img, prompt):
                """Create a simple demo mask based on prompt keywords."""
                try:
                    h, w = img.shape[:2]
                    mask = np.zeros((h, w), dtype=bool)
                    
                    # Ensure image is valid
                    if img is None or len(img.shape) != 3:
                        return mask
                    
                    # Convert to HSV for color detection
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    
                    prompt_lower = prompt.lower()
                    
                    # Simple color detection
                    if 'red' in prompt_lower:
                        red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                        red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
                        mask |= (red_mask1 | red_mask2).astype(bool)
                    
                    if 'blue' in prompt_lower:
                        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
                        mask |= blue_mask.astype(bool)
                    
                    if 'green' in prompt_lower:
                        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
                        mask |= green_mask.astype(bool)
                    
                    # Spatial filtering
                    if 'left' in prompt_lower:
                        mask[:, w//2:] = False
                    elif 'right' in prompt_lower:
                        mask[:, :w//2] = False
                    elif 'center' in prompt_lower:
                        margin = w // 4
                        mask[:, :margin] = False
                        mask[:, w-margin:] = False
                    
                    if 'top' in prompt_lower or 'above' in prompt_lower:
                        mask[h//2:, :] = False
                    elif 'bottom' in prompt_lower or 'below' in prompt_lower:
                        mask[:h//2, :] = False
                    
                    # If no specific detection, create center region
                    if not mask.any():
                        center_y, center_x = h//2, w//2
                        y1, y2 = max(0, center_y-50), min(h, center_y+50)
                        x1, x2 = max(0, center_x-50), min(w, center_x+50)
                        mask[y1:y2, x1:x2] = True
                    
                    return mask
                
                except Exception as e:
                    print(f"Error creating mask: {e}")
                    h, w = img.shape[:2] if img is not None and len(img.shape) >= 2 else (100, 100)
                    return np.zeros((h, w), dtype=bool)
        
        # Create demo interface
        demo_obj = SimpleMultiViewDemo()
        
        with gr.Blocks(title="Simple Multi-View Segmentation Demo") as demo:
            gr.HTML("<h1>🎯 Simple Multi-View Segmentation Demo</h1>")
            
            temp_dir_state = gr.State()
            
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    image_input = gr.File(file_count="multiple", label="Upload Images")
                    
                    upload_btn = gr.Button("📁 Process Upload", variant="primary")
                    
                    gallery = gr.Gallery(label="Uploaded Images", columns=3, height=200)
                    
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="Spatial Description", 
                        placeholder="e.g., 'red objects on the left', 'blue center area'",
                        lines=2
                    )
                    
                    segment_btn = gr.Button("🎯 Segment", variant="primary")
                    
                    status_output = gr.Textbox(label="Status", interactive=False)
                    
                    result_gallery = gr.Gallery(
                        label="Segmentation Results", 
                        columns=2, 
                        height=300
                    )
            
            # Example prompts
            with gr.Row():
                gr.Button("Red objects").click(lambda: "red objects", outputs=[prompt_input])
                gr.Button("Blue center").click(lambda: "blue objects in center", outputs=[prompt_input]) 
                gr.Button("Green left").click(lambda: "green objects on left", outputs=[prompt_input])
            
            # Event handlers
            upload_btn.click(
                demo_obj.process_upload,
                inputs=[video_input, image_input],
                outputs=[temp_dir_state, status_output, gallery]
            )
            
            segment_btn.click(
                demo_obj.simple_segmentation,
                inputs=[temp_dir_state, prompt_input],
                outputs=[result_gallery, status_output]
            )
            
            gr.Markdown("""
            ### 📖 Instructions
            1. Upload a video or multiple images of the same scene
            2. Enter a spatial description (colors and positions work best)
            3. Click "Segment" to see results
            
            **Note:** This is a simplified demo. For full 3D reconstruction and advanced segmentation, 
            use the complete setup with VGGT and SAM2 models.
            """)
        
        demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
        
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        return False

def launch_original_vggt():
    """Launch original VGGT demo."""
    print("🚀 Launching Original VGGT Demo...")
    try:
        os.chdir("vggt")
        subprocess.run([sys.executable, "demo_gradio.py"])
    except Exception as e:
        print(f"❌ Error launching VGGT demo: {e}")

def main():
    """Main function."""
    print("🎯 Simple Multi-View Segmentation Launcher")
    print("=" * 50)
    
    setup_environment()
    
    if not check_basic_deps():
        print("❌ Basic dependencies missing. Please install PyTorch, Gradio, and OpenCV.")
        return
    
    print("\n🚀 Available options:")
    print("1. Simple Multi-View Segmentation Demo (Recommended)")
    print("2. Original VGGT 3D Reconstruction Demo")
    print("3. Exit")
    
    try:
        choice = input("\nChoose option (1-3, default=1): ").strip()
        if not choice:
            choice = "1"
        
        if choice == "1":
            launch_basic_multiview_demo()
        elif choice == "2":
            launch_original_vggt()
        elif choice == "3":
            print("👋 Goodbye!")
        else:
            print("Invalid choice. Launching simple demo...")
            launch_basic_multiview_demo()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
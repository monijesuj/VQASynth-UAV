#!/usr/bin/env python3
"""
Working Multi-View Segmentation Demo

A simplified, fully working demo that demonstrates multi-view spatial segmentation
concepts without requiring the full VGGT/SAM2 setup.
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr
import tempfile
import shutil
from datetime import datetime
from PIL import Image


class SimpleSegmentationDemo:
    """A working demonstration of multi-view segmentation concepts."""
    
    def __init__(self):
        self.temp_dirs = []  # Track temporary directories for cleanup
    
    def process_uploads(self, input_video, input_images):
        """Process uploaded files and create image gallery."""
        if not input_video and not input_images:
            return None, "Please upload images or video.", []
        
        # Create temporary directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_dir = f"temp_demo_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dirs.append(temp_dir)
        
        image_paths = []
        
        try:
            # Handle video
            if input_video is not None:
                video_path = input_video if isinstance(input_video, str) else input_video.name
                vs = cv2.VideoCapture(video_path)
                fps = vs.get(cv2.CAP_PROP_FPS)
                frame_interval = max(1, int(fps))  # 1 frame per second
                
                count = 0
                frame_num = 0
                while True:
                    ret, frame = vs.read()
                    if not ret:
                        break
                    count += 1
                    if count % frame_interval == 0:
                        frame_path = os.path.join(temp_dir, f"frame_{frame_num:03d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        image_paths.append(frame_path)
                        frame_num += 1
                        if frame_num >= 10:  # Limit frames
                            break
                vs.release()
            
            # Handle images
            if input_images is not None:
                for i, img_file in enumerate(input_images):
                    img_path = img_file if isinstance(img_file, str) else img_file.name
                    if os.path.exists(img_path):
                        dst_path = os.path.join(temp_dir, f"image_{i:03d}.jpg")
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Resize if too large
                            h, w = img.shape[:2]
                            if max(h, w) > 800:
                                scale = 800 / max(h, w)
                                new_w, new_h = int(w * scale), int(h * scale)
                                img = cv2.resize(img, (new_w, new_h))
                            cv2.imwrite(dst_path, img)
                            image_paths.append(dst_path)
            
            image_paths = sorted(image_paths)
            status = f"Processed {len(image_paths)} images. Ready for segmentation."
            
            return temp_dir, status, image_paths
        
        except Exception as e:
            return None, f"Error processing upload: {str(e)}", []
    
    def run_segmentation(self, temp_dir, prompt, confidence):
        """Run simple segmentation based on color and spatial keywords."""
        if not temp_dir or not os.path.exists(temp_dir):
            return [], [], "Please upload images first."
        
        if not prompt or prompt.strip() == "":
            return [], [], "Please enter a description."
        
        try:
            # Get image paths
            image_paths = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            image_paths = sorted(image_paths)
            
            if len(image_paths) == 0:
                return [], [], "No images found to process."
            
            overlays = []
            masks = []
            
            for img_path in image_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Create mask based on prompt
                mask = self.create_simple_mask(img, prompt, confidence)
                
                # Create overlay
                overlay_img = self.create_overlay(img, mask)
                overlays.append(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
                
                # Create mask visualization
                mask_viz = np.zeros_like(img)
                mask_viz[mask] = 255
                masks.append(cv2.cvtColor(mask_viz, cv2.COLOR_BGR2RGB))
            
            status = f"✅ Segmented {len(overlays)} images with: '{prompt}'"
            return overlays, masks, status
        
        except Exception as e:
            return [], [], f"Error during segmentation: {str(e)}"
    
    def create_simple_mask(self, img, prompt, confidence):
        """Create mask based on color detection and spatial keywords."""
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        
        prompt_lower = prompt.lower()
        
        # Color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        if 'red' in prompt_lower:
            # Red detection
            mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
            color_mask = (mask1 | mask2) > 0
            mask |= color_mask
        
        if 'blue' in prompt_lower:
            # Blue detection
            color_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255)) > 0
            mask |= color_mask
        
        if 'green' in prompt_lower:
            # Green detection
            color_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255)) > 0
            mask |= color_mask
        
        if 'yellow' in prompt_lower:
            # Yellow detection
            color_mask = cv2.inRange(hsv, (20, 50, 50), (40, 255, 255)) > 0
            mask |= color_mask
        
        # Apply spatial constraints
        if 'left' in prompt_lower:
            mask[:, w//2:] = False
        elif 'right' in prompt_lower:
            mask[:, :w//2] = False
        
        if 'top' in prompt_lower or 'above' in prompt_lower:
            mask[h//2:, :] = False
        elif 'bottom' in prompt_lower or 'below' in prompt_lower:
            mask[:h//2, :] = False
        
        if 'center' in prompt_lower:
            # Keep only center region
            center_mask = np.zeros_like(mask)
            center_mask[h//4:3*h//4, w//4:3*w//4] = True
            mask &= center_mask
        
        # Apply confidence-based filtering (morphological operations)
        if confidence > 0.5:
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
        
        return mask
    
    def create_overlay(self, img, mask):
        """Create visualization overlay."""
        overlay = img.copy()
        
        # Simple approach: just set green pixels where mask is True
        overlay[mask, 1] = 255  # Set green channel to max
        overlay[mask, 0] = 0    # Set blue channel to 0
        overlay[mask, 2] = 0    # Set red channel to 0
        
        return overlay
    
    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(title="Simple Multi-View Segmentation Demo", theme=gr.themes.Soft()) as demo:
            
            # State for temporary directory
            temp_dir_state = gr.State()
            
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1>🎯 Simple Multi-View Segmentation Demo</h1>
                <p>Upload images and use natural language to segment objects</p>
                <p><em>Note: This is a simplified demo using basic color detection</em></p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 Upload")
                    
                    input_video = gr.Video(label="Upload Video (optional)")
                    input_images = gr.File(
                        file_count="multiple",
                        label="Upload Images (optional)",
                        file_types=["image"]
                    )
                    
                    upload_btn = gr.Button("📤 Process Upload", variant="primary")
                    
                    image_gallery = gr.Gallery(
                        label="Input Images",
                        columns=2,
                        height=300
                    )
                    
                    gr.Markdown("### 🎯 Segmentation")
                    
                    prompt_input = gr.Textbox(
                        label="Description",
                        placeholder="e.g., 'red objects on the left'",
                        lines=2
                    )
                    
                    confidence_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.1,
                        label="Confidence Threshold"
                    )
                    
                    segment_btn = gr.Button("🔍 Segment", variant="primary")
                
                with gr.Column(scale=2):
                    status_display = gr.Markdown("Upload images and enter description to begin.")
                    
                    with gr.Tabs():
                        with gr.TabItem("🎨 Overlays"):
                            overlay_gallery = gr.Gallery(
                                label="Segmentation Results",
                                columns=2,
                                height=400
                            )
                        
                        with gr.TabItem("🎭 Masks"):
                            mask_gallery = gr.Gallery(
                                label="Binary Masks",
                                columns=2,
                                height=400
                            )
            
            # Example prompts
            gr.HTML("""
            <div style="background: #f0f9ff; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                <h3>💡 Example Prompts:</h3>
                <ul>
                    <li>"red objects on the left"</li>
                    <li>"blue items in the center"</li>
                    <li>"green things at the top"</li>
                    <li>"yellow objects on the right"</li>
                </ul>
                <p><em>Combine colors with positions for better results!</em></p>
            </div>
            """)
            
            # Event handlers
            upload_btn.click(
                fn=self.process_uploads,
                inputs=[input_video, input_images],
                outputs=[temp_dir_state, status_display, image_gallery]
            )
            
            segment_btn.click(
                fn=self.run_segmentation,
                inputs=[temp_dir_state, prompt_input, confidence_slider],
                outputs=[overlay_gallery, mask_gallery, status_display]
            )
        
        return demo
    
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()


def main():
    """Main function to run the demo."""
    print("🎯 Working Multi-View Segmentation Demo")
    print("=" * 50)
    
    try:
        # Create demo instance
        demo_instance = SimpleSegmentationDemo()
        
        # Create interface
        interface = demo_instance.create_interface()
        
        # Launch
        print("🚀 Launching demo...")
        print("📱 Access at: http://127.0.0.1:7860")
        
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True
        )
    
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        if 'demo_instance' in locals():
            demo_instance.cleanup()
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    main()
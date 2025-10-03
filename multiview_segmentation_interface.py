"""
Gradio Interface for Multi-View Spatial Segmentation

This provides an interactive web interface for the multi-view segmentation pipeline,
allowing users to upload images/videos, enter spatial prompts, and visualize results.
"""

import gradio as gr
import numpy as np
import cv2
import os
import shutil
from datetime import datetime
import torch
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

try:
    from multiview_segmentation import MultiViewSegmentationPipeline
except ImportError:
    print("Warning: MultiViewSegmentationPipeline not available. Running in demo mode.")
    MultiViewSegmentationPipeline = None


class MultiViewSegmentationInterface:
    """Gradio interface for multi-view segmentation."""
    
    def __init__(self):
        self.pipeline = None
        self.current_results = None
        self.setup_pipeline()
    
    def setup_pipeline(self):
        """Initialize the segmentation pipeline."""
        if MultiViewSegmentationPipeline is not None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.pipeline = MultiViewSegmentationPipeline(device=device)
                print("Multi-view segmentation pipeline initialized successfully!")
            except Exception as e:
                print(f"Error initializing pipeline: {e}")
                self.pipeline = None
        else:
            self.pipeline = None
    
    def process_upload(self, input_video, input_images):
        """Handle file uploads and create image gallery."""
        if not input_video and not input_images:
            return None, "Please upload a video or images.", []
        
        # Create temporary directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_dir = f"temp_upload_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)
        
        image_paths = []
        
        try:
            # Handle video
            if input_video is not None:
                video_path = input_video if isinstance(input_video, str) else input_video.name
                vs = cv2.VideoCapture(video_path)
                fps = vs.get(cv2.CAP_PROP_FPS)
                frame_interval = max(1, int(fps * 0.5))  # Extract 2 frames per second
                
                count = 0
                frame_num = 0
                while True:
                    ret, frame = vs.read()
                    if not ret:
                        break
                    count += 1
                    if count % frame_interval == 0:
                        image_path = os.path.join(temp_dir, f"frame_{frame_num:06d}.png")
                        cv2.imwrite(image_path, frame)
                        image_paths.append(image_path)
                        frame_num += 1
                vs.release()
            
            # Handle images
            if input_images is not None:
                for i, img_file in enumerate(input_images):
                    img_path = img_file if isinstance(img_file, str) else img_file.name
                    if os.path.exists(img_path):
                        # Copy to temp directory
                        dst_path = os.path.join(temp_dir, f"image_{i:06d}.png")
                        img = cv2.imread(img_path)
                        cv2.imwrite(dst_path, img)
                        image_paths.append(dst_path)
            
            image_paths = sorted(image_paths)
            status_msg = f"Uploaded {len(image_paths)} images. Ready for segmentation."
            
            return temp_dir, status_msg, image_paths
        
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None, f"Error processing upload: {str(e)}", []
    
    def run_segmentation(self, temp_dir, spatial_prompt, conf_threshold, show_points):
        """Run the multi-view segmentation pipeline."""
        if self.pipeline is None:
            return None, None, "Pipeline not available. Please check installation.", None
        
        if not temp_dir or not os.path.exists(temp_dir):
            return None, None, "Please upload images first.", None
        
        if not spatial_prompt or spatial_prompt.strip() == "":
            return None, None, "Please enter a spatial description.", None
        
        try:
            # Get image paths from temp directory
            image_paths = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
            image_paths = sorted(image_paths)
            
            if len(image_paths) == 0:
                return None, None, "No images found to process.", None
            
            # Run segmentation
            results = self.pipeline.run_segmentation(image_paths, spatial_prompt)
            self.current_results = results
            
            # Create visualizations
            overlays, masks, status = self.create_visualizations(results, show_points)
            
            # Create 3D visualization info
            num_views = len(results.get('segmentation_results', {}))
            total_points = sum(len(seg['points']) for seg in results['segmentation_results'].values())
            
            viz_info = f"Segmentation complete!\nViews processed: {num_views}\nTotal detected points: {total_points}"
            
            return overlays, masks, status, viz_info
        
        except Exception as e:
            return None, None, f"Error during segmentation: {str(e)}", None
    
    def create_visualizations(self, results, show_points=True):
        """Create overlay and mask visualizations."""
        overlays = []
        masks = []
        
        try:
            segmentation_results = results.get('segmentation_results', {})
            
            for view_idx in sorted(segmentation_results.keys()):
                seg_result = segmentation_results[view_idx]
                image_path = seg_result['image_path']
                mask = seg_result['mask']
                points = seg_result['points']
                
                # Load original image
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Create overlay
                overlay = image_rgb.copy()
                
                # Apply mask overlay (semi-transparent green)
                mask_colored = np.zeros_like(overlay)
                mask_colored[mask] = [0, 255, 0]  # Green
                overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
                
                # Add points if requested
                if show_points and len(points) > 0:
                    for point in points:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(overlay, (x, y), 8, (255, 0, 0), -1)  # Red points
                        cv2.circle(overlay, (x, y), 10, (255, 255, 255), 2)  # White border
                
                overlays.append(overlay)
                
                # Create mask visualization
                mask_viz = np.zeros_like(image_rgb)
                mask_viz[mask] = [255, 255, 255]  # White mask
                masks.append(mask_viz)
            
            status = f"Created visualizations for {len(overlays)} views"
            return overlays, masks, status
        
        except Exception as e:
            return [], [], f"Error creating visualizations: {str(e)}"
    
    def export_results(self, output_format):
        """Export segmentation results."""
        if self.current_results is None:
            return None, "No results to export. Please run segmentation first."
        
        try:
            # Create export directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = f"segmentation_export_{timestamp}"
            os.makedirs(export_dir, exist_ok=True)
            
            # Save results using the pipeline's save method
            self.pipeline.save_results(self.current_results, export_dir)
            
            # Create a zip file if requested
            if output_format == "zip":
                import zipfile
                zip_path = f"{export_dir}.zip"
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for root, dirs, files in os.walk(export_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, export_dir)
                            zipf.write(file_path, arcname)
                
                return zip_path, f"Results exported to {zip_path}"
            else:
                return export_dir, f"Results exported to {export_dir}"
        
        except Exception as e:
            return None, f"Error exporting results: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface."""
        theme = gr.themes.Soft()
        
        with gr.Blocks(
            theme=theme,
            title="Multi-View Spatial Segmentation",
            css="""
            .main-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            .status-box {
                background: linear-gradient(45deg, #f0f9ff, #e0f2fe);
                border-radius: 10px;
                padding: 1rem;
                border: 1px solid #0ea5e9;
            }
            .example-prompt {
                background: #f8fafc;
                border-left: 4px solid #0ea5e9;
                padding: 0.5rem;
                margin: 0.5rem 0;
            }
            """
        ) as demo:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🔍 Multi-View Spatial Segmentation</h1>
                <p>Segment objects across multiple views using natural language spatial descriptions</p>
                <p>Powered by VGGT + SAM2 + CLIP + Spatial Reasoning</p>
            </div>
            """)
            
            # Hidden state for temporary directory
            temp_dir_state = gr.State()
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 Input")
                    
                    # File upload
                    input_video = gr.Video(label="Upload Video (optional)")
                    input_images = gr.File(
                        file_count="multiple", 
                        label="Upload Images (optional)",
                        file_types=["image"]
                    )
                    
                    # Upload button
                    upload_btn = gr.Button("📤 Process Upload", variant="primary")
                    
                    # Image gallery
                    image_gallery = gr.Gallery(
                        label="Uploaded Images",
                        columns=3,
                        height=300,
                        show_download_button=False
                    )
                    
                    # Spatial prompt
                    gr.Markdown("### 💬 Spatial Description")
                    
                    # Example prompts
                    gr.HTML("""
                    <div class="example-prompt">
                        <strong>Example prompts:</strong><br>
                        • "red chair on the left side"<br>
                        • "person walking in the background"<br>
                        • "blue car in front of the building"<br>
                        • "green plant near the window"<br>
                        • "white box on the table"
                    </div>
                    """)
                    
                    spatial_prompt = gr.Textbox(
                        label="Describe what to segment",
                        placeholder="Enter spatial description (e.g., 'red chair on the left side')",
                        lines=2
                    )
                    
                    # Parameters
                    with gr.Accordion("⚙️ Parameters", open=False):
                        conf_threshold = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.5, step=0.1,
                            label="Confidence Threshold"
                        )
                        show_points = gr.Checkbox(label="Show Detection Points", value=True)
                    
                    # Segmentation button
                    segment_btn = gr.Button("🎯 Run Segmentation", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Results")
                    
                    # Status display
                    status_display = gr.Markdown("Upload images and enter spatial description to begin.", elem_classes=["status-box"])
                    
                    # Visualization tabs
                    with gr.Tabs():
                        with gr.TabItem("🎨 Overlays"):
                            overlay_gallery = gr.Gallery(
                                label="Segmentation Overlays",
                                columns=2,
                                height=400,
                                show_download_button=True
                            )
                        
                        with gr.TabItem("🎭 Masks"):
                            mask_gallery = gr.Gallery(
                                label="Segmentation Masks",
                                columns=2,
                                height=400,
                                show_download_button=True
                            )
                        
                        with gr.TabItem("🏗️ 3D Info"):
                            viz_info = gr.Textbox(
                                label="3D Reconstruction Info",
                                lines=5,
                                interactive=False
                            )
                    
                    # Export section
                    with gr.Accordion("💾 Export Results", open=False):
                        export_format = gr.Radio(
                            choices=["folder", "zip"],
                            value="folder",
                            label="Export Format"
                        )
                        export_btn = gr.Button("📥 Export Results")
                        export_status = gr.Textbox(label="Export Status", interactive=False)
            
            # Event handlers
            upload_btn.click(
                fn=self.process_upload,
                inputs=[input_video, input_images],
                outputs=[temp_dir_state, status_display, image_gallery]
            )
            
            segment_btn.click(
                fn=self.run_segmentation,
                inputs=[temp_dir_state, spatial_prompt, conf_threshold, show_points],
                outputs=[overlay_gallery, mask_gallery, status_display, viz_info]
            )
            
            export_btn.click(
                fn=self.export_results,
                inputs=[export_format],
                outputs=[export_status]
            )
            
            # Example section
            gr.Markdown("### 🎯 Try Examples")
            
            # Example prompt buttons instead of file examples
            with gr.Row():
                example_btn1 = gr.Button("🪑 Red chair on left", size="sm")
                example_btn2 = gr.Button("🚶 Person in background", size="sm") 
                example_btn3 = gr.Button("🚗 Blue car in center", size="sm")
            
            # Example button handlers
            example_btn1.click(
                lambda: "red chair on the left side",
                outputs=[spatial_prompt]
            )
            example_btn2.click(
                lambda: "person walking in the background", 
                outputs=[spatial_prompt]
            )
            example_btn3.click(
                lambda: "blue car in the center",
                outputs=[spatial_prompt]
            )
            
            # Instructions
            with gr.Accordion("📖 Instructions", open=False):
                gr.Markdown("""
                **How to use:**
                1. **Upload**: Choose either a video file or multiple images of the same scene
                2. **Process**: Click "Process Upload" to prepare your images
                3. **Describe**: Enter a spatial description of what you want to segment
                4. **Segment**: Click "Run Segmentation" to process
                5. **Review**: Check results in the Overlays and Masks tabs
                6. **Export**: Download your results as needed
                
                **Tips:**
                - Use clear, specific descriptions with colors, positions, and objects
                - Multiple views of the same scene work best
                - Higher confidence thresholds filter out uncertain detections
                - The system works by understanding spatial relationships in 3D space
                """)
        
        return demo


def launch_interface():
    """Launch the Gradio interface."""
    interface = MultiViewSegmentationInterface()
    demo = interface.create_interface()
    
    # Launch with sharing enabled
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    launch_interface()
#!/usr/bin/env python3
"""
Spatial Reasoning Explorer for Drone Navigation
Interactive tool to ask spatial questions about images, point clouds, and videos
"""
import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2
import os

# Try to import spatial reasoning models
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False
    print("⚠️  Qwen2.5-VL not available. Install: pip install transformers torch")

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    print("⚠️  Open3D not available. Install: pip install open3d")

class SpatialReasoningExplorer:
    def __init__(self):
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load local spatial reasoning models (SpaceOm or SpaceThinker only)"""
        if HAS_QWEN:
            # Only use local spatial reasoning models
            model_options = [
                "./SpaceOm",  # SpaceOm - best overall spatial reasoning
                "./SpaceThinker-Qwen2.5VL-3B"  # SpaceThinker - most accurate distances
            ]
            
            for model_name in model_options:
                try:
                    print(f"🔄 Loading {model_name}...")
                    
                    self.processor = AutoProcessor.from_pretrained(
                        model_name, 
                        trust_remote_code=True
                    )
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_name, 
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    print(f"✅ Successfully loaded {model_name}")
                    return True
                    
                except Exception as e:
                    print(f"⚠️  Failed to load {model_name}: {str(e)}")
                    continue
            
            print("❌ Could not load SpaceOm or SpaceThinker models")
        return False
    
    def analyze_spatial_relationships(self, image, question):
        """Answer spatial questions about an image"""
        if not self.model:
            return "❌ No spatial reasoning model available. Please install required packages."
        
        try:
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Process input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = self.processor.process_vision_info(messages)
            inputs = self.processor(
                text=[text], 
                images=image_inputs, 
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False
                )
                
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids 
                in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return response
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def estimate_depth_and_scale(self, image):
        """Estimate depth map for spatial understanding"""
        try:
            # Simple depth estimation using image analysis
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Basic depth cues (this is simplified - real depth would use VGGT/DepthPro)
            blur_map = cv2.GaussianBlur(gray, (21, 21), 0)
            depth_estimate = 255 - blur_map  # Simplified: sharper = closer
            
            # Create depth visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.imshow(image)
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            im = ax2.imshow(depth_estimate, cmap='viridis')
            ax2.set_title("Estimated Depth Map")
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, label='Estimated Depth')
            
            plt.tight_layout()
            plt.savefig('/tmp/depth_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return '/tmp/depth_analysis.png'
            
        except Exception as e:
            return f"❌ Depth estimation error: {str(e)}"
    
    def analyze_point_cloud(self, pc_file):
        """Analyze spatial relationships in point cloud"""
        if not HAS_O3D:
            return "❌ Open3D not available for point cloud analysis"
        
        try:
            # Load point cloud
            pcd = o3d.io.read_point_cloud(pc_file)
            points = np.asarray(pcd.points)
            
            if len(points) == 0:
                return "❌ Empty point cloud"
            
            # Basic spatial analysis
            x_range = points[:, 0].max() - points[:, 0].min()
            y_range = points[:, 1].max() - points[:, 1].min()  
            z_range = points[:, 2].max() - points[:, 2].min()
            
            centroid = points.mean(axis=0)
            
            analysis = f"""
🔮 Point Cloud Spatial Analysis:
• Total points: {len(points):,}
• X extent: {x_range:.2f}m (width)
• Y extent: {y_range:.2f}m (depth)  
• Z extent: {z_range:.2f}m (height)
• Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})
• Volume estimate: {x_range * y_range * z_range:.2f} cubic meters
            """
            
            return analysis
            
        except Exception as e:
            return f"❌ Point cloud analysis error: {str(e)}"

# Initialize spatial reasoning explorer
explorer = SpatialReasoningExplorer()

def spatial_qa_interface(image, question):
    """Main interface for spatial Q&A"""
    if image is None:
        return "Please upload an image first."
    
    if not question.strip():
        return "Please enter a spatial question."
    
    return explorer.analyze_spatial_relationships(image, question)

def depth_analysis_interface(image):
    """Depth analysis interface"""
    if image is None:
        return None
    
    return explorer.estimate_depth_and_scale(image)

def pc_analysis_interface(pc_file):
    """Point cloud analysis interface"""
    if pc_file is None:
        return "Please upload a point cloud file (.pcd, .ply, .xyz)"
    
    return explorer.analyze_point_cloud(pc_file.name)

# Drone navigation specific questions
drone_questions = [
    "What is the height of the tallest object in this scene?",
    "How much clearance is there between the drone and obstacles?", 
    "Which direction has the most open space for navigation?",
    "What is the distance between the two largest objects?",
    "Is there enough vertical space for a drone to pass through?",
    "What are the dimensions of the landing area?",
    "How far is the building from the trees?",
    "What is the safest flight path through this scene?"
]

# Create Gradio interface
with gr.Blocks(title="Spatial Reasoning Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚁 Spatial Reasoning Explorer for Drone Navigation
    
    Interactive tool to analyze spatial relationships in images and point clouds.
    Ask questions about heights, distances, clearances, and navigation paths.
    """)
    
    with gr.Tabs():
        # Image Spatial Analysis Tab
        with gr.TabItem("📸 Image Spatial Analysis"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    question_input = gr.Textbox(
                        placeholder="Ask about heights, distances, relationships...",
                        label="Spatial Question",
                        lines=2
                    )
                    
                    gr.Markdown("**🚁 Drone Navigation Questions:**")
                    question_buttons = []
                    for q in drone_questions:
                        btn = gr.Button(q, size="sm")
                        btn.click(lambda q=q: q, outputs=question_input)
                    
                    analyze_btn = gr.Button("🔍 Analyze Spatial Relationships", variant="primary")
                    
                with gr.Column():
                    answer_output = gr.Textbox(
                        label="Spatial Analysis Result",
                        lines=8,
                        max_lines=20
                    )
            
            analyze_btn.click(
                fn=spatial_qa_interface,
                inputs=[image_input, question_input],
                outputs=answer_output
            )
        
        # Depth Analysis Tab  
        with gr.TabItem("📏 Depth & Scale Analysis"):
            with gr.Row():
                with gr.Column():
                    depth_image_input = gr.Image(type="pil", label="Upload Image for Depth Analysis")
                    depth_btn = gr.Button("📊 Generate Depth Map", variant="primary")
                    
                with gr.Column():
                    depth_output = gr.Image(label="Depth Analysis Result")
            
            depth_btn.click(
                fn=depth_analysis_interface,
                inputs=depth_image_input,
                outputs=depth_output
            )
        
        # Point Cloud Analysis Tab
        with gr.TabItem("🔮 Point Cloud Analysis"):
            with gr.Row():
                with gr.Column():
                    pc_input = gr.File(
                        label="Upload Point Cloud (.pcd, .ply, .xyz)",
                        file_types=[".pcd", ".ply", ".xyz"]
                    )
                    pc_btn = gr.Button("🔍 Analyze Point Cloud", variant="primary")
                    
                with gr.Column():
                    pc_output = gr.Textbox(
                        label="Point Cloud Analysis",
                        lines=10
                    )
            
            pc_btn.click(
                fn=pc_analysis_interface,
                inputs=pc_input,
                outputs=pc_output
            )
    
    gr.Markdown("""
    ## 🎯 Usage Examples:
    - **Heights**: "What is the height of the building?" "How tall is the tree?"
    - **Distances**: "How far apart are the two cars?" "Distance from drone to obstacle?"  
    - **Clearances**: "Is there enough space for a 2-meter drone?" "What's the clearance under the bridge?"
    - **Navigation**: "Which path has the most open space?" "Where is the safest landing zone?"
    """)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
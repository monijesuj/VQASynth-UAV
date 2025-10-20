#!/usr/bin/env python3
"""
Point Cloud to Image Converter for Spatial VLM
Convert point clouds to visual representations that can be analyzed by VL models
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image
import os

def point_cloud_to_image(pcd_path, output_size=(512, 512), views=['front', 'side', 'top']):
    """
    Convert point cloud to multiple view images for VL model analysis
    
    Args:
        pcd_path: Path to .pcd file
        output_size: Output image size (width, height)
        views: List of views to generate ['front', 'side', 'top', '3d']
    
    Returns:
        List of PIL Images showing different viewpoints
    """
    try:
        # Try to use Open3D if available
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            # Default color based on height (Z-axis)
            colors = plt.cm.viridis((points[:, 2] - points[:, 2].min()) / 
                                   (points[:, 2].max() - points[:, 2].min()))[:, :3]
            
    except ImportError:
        print("Open3D not available, trying manual PCD parsing...")
        # Manual PCD file parsing
        points, colors = parse_pcd_file(pcd_path)
        if colors is None:
            colors = plt.cm.viridis((points[:, 2] - points[:, 2].min()) / 
                                   (points[:, 2].max() - points[:, 2].min()))[:, :3]
    
    if len(points) == 0:
        raise ValueError("No points found in point cloud")
    
    # Generate multiple views
    view_images = []
    
    for view in views:
        fig = plt.figure(figsize=(8, 8))
        
        if view == '3d':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors, s=1, alpha=0.6)
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_zlabel('Z (meters)')
            ax.set_title('3D Point Cloud View')
            
        elif view == 'front':  # X-Z plane (front view)
            ax = fig.add_subplot(111)
            ax.scatter(points[:, 0], points[:, 2], c=colors, s=1, alpha=0.6)
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Z (meters)')
            ax.set_title('Front View (X-Z plane)')
            ax.set_aspect('equal')
            
        elif view == 'side':  # Y-Z plane (side view)
            ax = fig.add_subplot(111)
            ax.scatter(points[:, 1], points[:, 2], c=colors, s=1, alpha=0.6)
            ax.set_xlabel('Y (meters)')
            ax.set_ylabel('Z (meters)')
            ax.set_title('Side View (Y-Z plane)')
            ax.set_aspect('equal')
            
        elif view == 'top':  # X-Y plane (top view)
            ax = fig.add_subplot(111)
            ax.scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.6)
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_title('Top View (X-Y plane)')
            ax.set_aspect('equal')
        
        # Convert plot to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img = img.resize(output_size, Image.Resampling.LANCZOS)
        view_images.append(img)
        plt.close(fig)
    
    return view_images

def parse_pcd_file(pcd_path):
    """Manually parse PCD file if Open3D not available"""
    points = []
    colors = None
    
    with open(pcd_path, 'r') as f:
        header = True
        data_format = 'ascii'
        has_color = False
        
        for line in f:
            line = line.strip()
            
            if header:
                if line.startswith('FIELDS'):
                    fields = line.split()[1:]
                    has_color = 'rgb' in fields or ('r' in fields and 'g' in fields and 'b' in fields)
                elif line.startswith('DATA'):
                    data_format = line.split()[1]
                    header = False
            else:
                if data_format == 'ascii':
                    try:
                        values = list(map(float, line.split()))
                        if len(values) >= 3:
                            points.append(values[:3])  # x, y, z
                    except ValueError:
                        continue
    
    points = np.array(points)
    return points, colors

def analyze_point_cloud_with_vlm(pcd_path, question, spatial_model_func):
    """
    Analyze point cloud by converting to images and using VL model
    
    Args:
        pcd_path: Path to point cloud file
        question: Spatial question to ask
        spatial_model_func: Function that takes (images, question) and returns analysis
    
    Returns:
        Analysis result with point cloud context
    """
    try:
        # Convert point cloud to multiple view images
        view_images = point_cloud_to_image(pcd_path, views=['3d', 'front', 'side', 'top'])
        
        # Create enhanced question with point cloud context
        enhanced_question = f"""
This is a point cloud visualization showing multiple views of a 3D scene:
- Image 1: 3D perspective view
- Image 2: Front view (X-Z plane) 
- Image 3: Side view (Y-Z plane)
- Image 4: Top view (X-Y plane)

Original question: {question}

Please analyze the spatial relationships in this 3D point cloud data.
"""
        
        # Use VL model to analyze the visualized point cloud
        result = spatial_model_func(view_images, enhanced_question)
        
        return f"🔮 **Point Cloud Analysis** (from {os.path.basename(pcd_path)}):\n\n{result}"
        
    except Exception as e:
        return f"❌ Error analyzing point cloud: {str(e)}"

# Example usage function
def demo_point_cloud_analysis():
    """Demo function showing how to use point cloud analysis"""
    
    # Example point cloud files from your VQASynth output
    pcd_files = [
        "/home/isr-lab3/James/vqasynth_output/pointclouds/pointcloud_0_0.pcd",
        "/home/isr-lab3/James/vqasynth_output/pointclouds/pointcloud_1_0.pcd"
    ]
    
    for pcd_file in pcd_files:
        if os.path.exists(pcd_file):
            print(f"\n📊 Analyzing {pcd_file}")
            
            try:
                # Convert to images
                images = point_cloud_to_image(pcd_file)
                print(f"✅ Generated {len(images)} view images")
                
                # Save example images
                for i, img in enumerate(images):
                    output_path = f"pcd_view_{i}.png"
                    img.save(output_path)
                    print(f"💾 Saved {output_path}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    demo_point_cloud_analysis()
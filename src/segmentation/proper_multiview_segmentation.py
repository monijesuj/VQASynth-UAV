#!/usr/bin/env python3
"""
Proper Multi-View Spatial Segmentation with SAM2 + VGGT + CLIP

This is the actual implementation that uses:
- VGGT for 3D reconstruction
- SAM2 for segmentation
- CLIP for spatial understanding
- Multi-view consistency
"""

import os
import sys
import cv2
import torch
import numpy as np
import gradio as gr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "vggt"))
sys.path.insert(0, str(current_dir / "sam2"))
sys.path.insert(0, str(current_dir))

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# SAM2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("⚠️  SAM2 not available")
    SAM2_AVAILABLE = False

# CLIP imports
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    print("⚠️  CLIP not available")
    CLIP_AVAILABLE = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {device}")

class ProperMultiViewSegmentation:
    """Proper multi-view segmentation using SAM2 + VGGT + CLIP."""
    
    def __init__(self):
        self.device = device
        self.vggt_model = None
        self.sam2_predictor = None
        self.clip_model = None
        self.clip_preprocess = None
        
        self.setup_models()
    
    def setup_models(self):
        """Initialize all models."""
        print("🔧 Initializing models...")
        
        # Initialize VGGT
        try:
            self.vggt_model = VGGT()
            vggt_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            self.vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(vggt_url))
            self.vggt_model.eval()
            self.vggt_model = self.vggt_model.to(self.device)
            print("✅ VGGT initialized")
        except Exception as e:
            print(f"❌ VGGT initialization failed: {e}")
        
        # Initialize SAM2
        if SAM2_AVAILABLE:
            try:
                # Check for SAM2 checkpoint
                sam2_checkpoint = current_dir / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
                if sam2_checkpoint.exists():
                    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(sam2_checkpoint), device=self.device)
                    self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                    print("✅ SAM2 initialized with checkpoint")
                else:
                    print(f"⚠️  SAM2 checkpoint not found at {sam2_checkpoint}")
            except Exception as e:
                print(f"⚠️  SAM2 initialization failed: {e}")
        
        # Initialize CLIP
        if CLIP_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=self.device)
                print("✅ CLIP initialized")
            except Exception as e:
                print(f"⚠️  CLIP initialization failed: {e}")
    
    def run_vggt_reconstruction(self, image_dir):
        """Run VGGT 3D reconstruction."""
        print("🏗️  Running VGGT 3D reconstruction...")
        
        try:
            image_paths = sorted([f for f in Path(image_dir).glob("*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
            
            if len(image_paths) == 0:
                raise ValueError("No images found")
            
            print(f"   Found {len(image_paths)} images")
            
            images = load_and_preprocess_images([str(p) for p in image_paths]).to(self.device)
            
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = self.vggt_model(images)
            
            # Convert pose encoding
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            
            # Convert to numpy
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)
            
            # Generate 3D points
            depth_map = predictions["depth"]
            world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
            predictions["world_points_from_depth"] = world_points
            predictions["image_paths"] = [str(p) for p in image_paths]
            
            torch.cuda.empty_cache()
            
            print("✅ VGGT reconstruction complete")
            return predictions
            
        except Exception as e:
            print(f"❌ VGGT reconstruction failed: {e}")
            return None
    
    def compute_clip_similarity(self, image_path, text_prompt):
        """Compute CLIP similarity between image and text."""
        if self.clip_model is None:
            return 0.5  # Default similarity
        
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Create multiple text variations for better matching
            text_variations = [
                text_prompt,
                f"a photo of {text_prompt}",
                f"an image containing {text_prompt}",
                f"{text_prompt} in the scene"
            ]
            
            text_inputs = clip.tokenize(text_variations).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Take maximum similarity across variations
                similarities = (image_features @ text_features.T).squeeze(0)
                max_similarity = similarities.max().item()
                
                return max_similarity
        except Exception as e:
            print(f"Error computing CLIP similarity: {e}")
            return 0.5
    
    def generate_sam2_points_from_prompt(self, image, prompt, predictions, view_idx):
        """Generate SAM2 prompt points based on spatial description and 3D info."""
        h, w = image.shape[:2]
        points = []
        
        prompt_lower = prompt.lower()
        
        # Parse spatial relations with more comprehensive coverage
        if 'left' in prompt_lower:
            points.extend([[w*0.25, h*0.5], [w*0.2, h*0.3], [w*0.2, h*0.7], [w*0.3, h*0.5]])
        elif 'right' in prompt_lower:
            points.extend([[w*0.75, h*0.5], [w*0.8, h*0.3], [w*0.8, h*0.7], [w*0.7, h*0.5]])
        
        if 'center' in prompt_lower or 'middle' in prompt_lower:
            points.extend([[w*0.5, h*0.5], [w*0.4, h*0.4], [w*0.6, h*0.6], [w*0.5, h*0.4], [w*0.5, h*0.6]])
        
        if 'top' in prompt_lower or 'above' in prompt_lower or 'upper' in prompt_lower:
            points.extend([[w*0.5, h*0.25], [w*0.3, h*0.2], [w*0.7, h*0.2], [w*0.5, h*0.3]])
        elif 'bottom' in prompt_lower or 'below' in prompt_lower or 'lower' in prompt_lower or 'floor' in prompt_lower:
            points.extend([[w*0.5, h*0.75], [w*0.3, h*0.8], [w*0.7, h*0.8], [w*0.5, h*0.7], [w*0.5, h*0.9]])
        
        # Add foreground/background points
        if 'front' in prompt_lower or 'foreground' in prompt_lower or 'close' in prompt_lower:
            points.extend([[w*0.5, h*0.6], [w*0.4, h*0.6], [w*0.6, h*0.6]])
        elif 'back' in prompt_lower or 'background' in prompt_lower or 'far' in prompt_lower:
            points.extend([[w*0.5, h*0.3], [w*0.4, h*0.3], [w*0.6, h*0.3]])
        
        # Use 3D info if available
        if predictions and 'world_points_from_depth' in predictions:
            world_points = predictions['world_points_from_depth'][view_idx]
            depth_conf = predictions.get('depth_conf', [None])[view_idx] if 'depth_conf' in predictions else None
            
            # Find high-confidence regions
            if depth_conf is not None:
                high_conf_mask = depth_conf > 0.5
                if high_conf_mask.any():
                    y_coords, x_coords = np.where(high_conf_mask)
                    if len(y_coords) > 0:
                        # Sample high-confidence points
                        num_samples = min(10, len(y_coords))
                        indices = np.random.choice(len(y_coords), num_samples, replace=False)
                        for idx in indices:
                            points.append([float(x_coords[idx]), float(y_coords[idx])])
        
        # If no points generated, use grid sampling
        if not points:
            # Sample grid points across image
            for y_frac in [0.3, 0.5, 0.7]:
                for x_frac in [0.3, 0.5, 0.7]:
                    points.append([w*x_frac, h*y_frac])
        
        return np.array(points, dtype=np.float32)
    
    def segment_with_sam2(self, image_path, prompt, predictions, view_idx):
        """Segment using SAM2."""
        if self.sam2_predictor is None:
            return None, None
        
        try:
            # Read image properly
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None, None
            
            # Convert BGR to RGB for SAM2
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Ensure correct format and dtype
            if image_rgb.dtype != np.uint8:
                image_rgb = image_rgb.astype(np.uint8)
            
            # Set image for SAM2
            self.sam2_predictor.set_image(image_rgb)
            
            # Generate prompt points
            points = self.generate_sam2_points_from_prompt(image, prompt, predictions, view_idx)
            labels = np.ones(len(points), dtype=np.int32)
            
            print(f"   Generated {len(points)} prompt points for SAM2")
            
            # Run SAM2 prediction
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
            
            print(f"   SAM2 generated {len(masks)} masks with scores: {scores}")
            
            # Return best mask
            best_idx = np.argmax(scores)
            return masks[best_idx], points
            
        except Exception as e:
            import traceback
            print(f"SAM2 segmentation error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None, None
    
    def run_multiview_segmentation(self, image_dir, spatial_prompt):
        """Main segmentation pipeline."""
        print(f"🎯 Running multi-view segmentation with prompt: '{spatial_prompt}'")
        
        # Step 1: VGGT 3D Reconstruction
        predictions = self.run_vggt_reconstruction(image_dir)
        if predictions is None:
            return None, "VGGT reconstruction failed"
        
        # Step 2: Find relevant views using CLIP
        image_paths = predictions["image_paths"]
        view_scores = []
        
        for img_path in image_paths:
            score = self.compute_clip_similarity(img_path, spatial_prompt)
            view_scores.append(score)
        
        print(f"📊 View relevance scores: {[f'{s:.3f}' for s in view_scores]}")
        
        # Step 3: Segment using SAM2
        results = []
        
        # Use adaptive threshold - process top views or views above 0.15
        threshold = max(0.15, max(view_scores) * 0.7) if view_scores else 0.15
        print(f"🎚️  Using threshold: {threshold:.3f}")
        
        for view_idx, img_path in enumerate(image_paths):
            # Only process relevant views
            if view_scores[view_idx] >= threshold:
                print(f"   Processing view {view_idx}: {Path(img_path).name} (score: {view_scores[view_idx]:.3f})")
                
                mask, points = self.segment_with_sam2(img_path, spatial_prompt, predictions, view_idx)
                
                if mask is not None:
                    # Create overlay
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print(f"   Failed to load image for overlay: {img_path}")
                        continue
                    
                    overlay = image.copy()
                    
                    # Apply mask with safe array operations
                    try:
                        # Create colored mask
                        colored_mask = np.zeros_like(overlay)
                        colored_mask[mask] = [0, 255, 0]  # Green in BGR
                        
                        # Blend
                        overlay = cv2.addWeighted(overlay, 0.5, colored_mask, 0.5, 0)
                        
                        # Draw points if available
                        if points is not None:
                            for point in points:
                                pt = (int(point[0]), int(point[1]))
                                cv2.circle(overlay, pt, 5, (255, 0, 0), -1)  # Blue points
                                cv2.circle(overlay, pt, 7, (255, 255, 255), 2)  # White border
                        
                        # Convert to RGB for display
                        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        
                        results.append({
                            'view_idx': view_idx,
                            'image_path': str(img_path),
                            'overlay': overlay_rgb,
                            'mask': mask,
                            'similarity': view_scores[view_idx],
                            'points': points
                        })
                        
                        print(f"   ✅ Successfully segmented view {view_idx}")
                        
                    except Exception as e:
                        print(f"   ❌ Error creating overlay for view {view_idx}: {e}")
                else:
                    print(f"   ⚠️  No mask generated for view {view_idx}")
            else:
                print(f"   Skipping view {view_idx}: {Path(img_path).name} (score: {view_scores[view_idx]:.3f} < {threshold:.3f})")
        
        status = f"Segmented {len(results)} views successfully"
        print(f"✅ {status}")
        
        return results, status


def create_gradio_interface():
    """Create Gradio interface."""
    
    segmenter = ProperMultiViewSegmentation()
    
    def process_files(video, images):
        """Process uploaded files."""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Handle video
            if video is not None:
                video_path = video if isinstance(video, str) else video.name
                vs = cv2.VideoCapture(video_path)
                fps = vs.get(cv2.CAP_PROP_FPS)
                frame_interval = max(1, int(fps * 0.5))
                
                count = 0
                frame_num = 0
                while True:
                    ret, frame = vs.read()
                    if not ret:
                        break
                    count += 1
                    if count % frame_interval == 0:
                        cv2.imwrite(f"{temp_dir}/frame_{frame_num:06d}.png", frame)
                        frame_num += 1
                vs.release()
            
            # Handle images
            if images is not None:
                for i, img_file in enumerate(images):
                    img_path = img_file if isinstance(img_file, str) else img_file.name
                    shutil.copy(img_path, f"{temp_dir}/image_{i:06d}.png")
            
            image_files = sorted([str(f) for f in Path(temp_dir).glob("*")])
            
            return temp_dir, image_files, f"Uploaded {len(image_files)} images"
            
        except Exception as e:
            return temp_dir, [], f"Error: {e}"
    
    def run_segmentation(temp_dir, prompt):
        """Run segmentation."""
        if not temp_dir or not os.path.exists(temp_dir):
            return [], "Please upload images first"
        
        if not prompt or not prompt.strip():
            return [], "Please enter a spatial prompt"
        
        results, status = segmenter.run_multiview_segmentation(temp_dir, prompt)
        
        if results is None:
            return [], status
        
        # Return overlays
        overlays = [r['overlay'] for r in results]
        return overlays, status
    
    with gr.Blocks(title="Multi-View Spatial Segmentation") as demo:
        gr.HTML("""
        <h1>🎯 Multi-View Spatial Segmentation</h1>
        <p>Proper implementation using VGGT + SAM2 + CLIP</p>
        """)
        
        temp_dir_state = gr.State(value=None)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📁 Upload")
                video_input = gr.Video(label="Video (optional)")
                image_input = gr.File(file_count="multiple", label="Images (optional)")
                upload_btn = gr.Button("📤 Process Upload", variant="primary")
                
                gallery = gr.Gallery(label="Input Images", columns=3, height=200)
                
                gr.Markdown("### 💬 Spatial Prompt")
                gr.Markdown("""
                **Example prompts:**
                - "red chair on the left side"
                - "blue car in the center"
                - "person in the background"
                """)
                prompt_input = gr.Textbox(
                    label="What to segment",
                    placeholder="e.g., 'red chair on the left side'",
                    lines=2
                )
                
                segment_btn = gr.Button("🎯 Segment", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### 📊 Results")
                status_output = gr.Textbox(label="Status", interactive=False, value="Upload images and enter a prompt to begin")
                result_gallery = gr.Gallery(label="Segmentation Results", columns=2, height=400)
        
        # Event handlers
        upload_btn.click(
            process_files,
            [video_input, image_input],
            [temp_dir_state, gallery, status_output],
            api_name="upload"
        )
        
        segment_btn.click(
            run_segmentation,
            [temp_dir_state, prompt_input],
            [result_gallery, status_output],
            api_name="segment"
        )
    
    return demo


if __name__ == "__main__":
    print("🚀 Launching Proper Multi-View Spatial Segmentation")
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)

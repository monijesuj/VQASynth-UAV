#!/usr/bin/env python3
"""
Single Image Prompt-based Segmentation
Using CLIP + RAM + SAM2 + GroundingDINO for high-quality segmentation
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import gradio as gr
from PIL import Image
import sys
import traceback

# Add paths for local modules
sys.path.insert(0, str(Path(__file__).parent / "sam2"))
sys.path.insert(0, str(Path(__file__).parent / "vqasynth"))

class SingleImageSegmentation:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.setup_models()
        
    def setup_models(self):
        """Initialize all models"""
        print("🔧 Initializing models...")
        
        # 1. Initialize CLIP for semantic understanding
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=self.device)
            print("✅ CLIP initialized")
        except Exception as e:
            print(f"❌ CLIP initialization failed: {e}")
            raise
        
        # 2. Initialize SAM2 for segmentation
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("✅ SAM2 initialized")
        except Exception as e:
            print(f"❌ SAM2 initialization failed: {e}")
            raise
        
        # 3. Try to initialize GroundingDINO (optional)
        self.grounding_dino = None
        try:
            # Check if GroundingDINO is available
            import groundingdino
            from groundingdino.util.inference import Model as GroundingDINOModel
            
            config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
            
            if Path(config_path).exists() and Path(checkpoint_path).exists():
                self.grounding_dino = GroundingDINOModel(
                    model_config_path=config_path,
                    model_checkpoint_path=checkpoint_path,
                    device=self.device
                )
                print("✅ GroundingDINO initialized")
            else:
                print("⚠️  GroundingDINO not found, using CLIP+SAM2 only")
        except Exception as e:
            print(f"⚠️  GroundingDINO not available: {e}")
        
        # 4. Try to initialize RAM (optional)
        self.ram_model = None
        try:
            from ram.models import ram
            from ram import inference_ram
            
            ram_checkpoint = 'ram_swin_large_14m.pth'
            if Path(ram_checkpoint).exists():
                self.ram_model = ram(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
                self.ram_model.eval()
                self.ram_model = self.ram_model.to(self.device)
                print("✅ RAM initialized")
            else:
                print("⚠️  RAM checkpoint not found, using CLIP only")
        except Exception as e:
            print(f"⚠️  RAM not available: {e}")
    
    def get_grounding_dino_boxes(self, image: np.ndarray, text_prompt: str, 
                                  box_threshold: float = 0.25, 
                                  text_threshold: float = 0.25) -> List[np.ndarray]:
        """Use GroundingDINO to detect objects"""
        if self.grounding_dino is None:
            return []
        
        try:
            detections = self.grounding_dino.predict_with_classes(
                image=image,
                classes=[text_prompt],
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            # Convert boxes to SAM2 format (xyxy)
            boxes = []
            if hasattr(detections, 'xyxy'):
                boxes = detections.xyxy.cpu().numpy()
            
            return boxes
        except Exception as e:
            print(f"⚠️  GroundingDINO detection failed: {e}")
            return []
    
    def compute_clip_similarity(self, image: np.ndarray, text_prompt: str) -> Tuple[float, np.ndarray]:
        """Compute CLIP similarity between image and text"""
        try:
            # Prepare image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Create text variations for better matching
            text_variations = [
                text_prompt,
                f"a photo of {text_prompt}",
                f"an image containing {text_prompt}",
                f"{text_prompt} in the scene"
            ]
            
            # Tokenize all variations
            import clip
            text_inputs = clip.tokenize(text_variations).to(self.device)
            
            # Compute similarities
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity (take max across variations)
                similarity = (image_features @ text_features.T).squeeze(0)
                max_similarity = similarity.max().item()
            
            # Generate attention map (simplified - use global similarity)
            attention_map = np.ones((image.shape[0], image.shape[1])) * max_similarity
            
            return max_similarity, attention_map
            
        except Exception as e:
            print(f"❌ CLIP similarity computation failed: {e}")
            return 0.0, np.zeros((image.shape[0], image.shape[1]))
    
    def generate_sam2_prompts(self, image: np.ndarray, text_prompt: str, 
                               boxes: Optional[List[np.ndarray]] = None) -> Dict:
        """Generate prompts for SAM2 from various sources"""
        prompts = {
            'point_coords': [],
            'point_labels': [],
            'boxes': []
        }
        
        # 1. Use GroundingDINO boxes if available
        if boxes is not None and len(boxes) > 0:
            prompts['boxes'] = boxes
            print(f"   📦 Using {len(boxes)} boxes from GroundingDINO")
            return prompts
        
        # 2. Fall back to CLIP-based point sampling
        similarity_score, attention_map = self.compute_clip_similarity(image, text_prompt)
        print(f"   📊 CLIP similarity: {similarity_score:.3f}")
        
        if similarity_score > 0.15:  # Reasonable threshold
            # Sample points from high-attention regions
            threshold = np.percentile(attention_map, 75)  # Top 25% regions
            high_attention = attention_map > threshold
            
            # Get coordinates
            y_coords, x_coords = np.where(high_attention)
            
            if len(y_coords) > 0:
                # Sample up to 10 points
                num_points = min(10, len(y_coords))
                indices = np.random.choice(len(y_coords), num_points, replace=False)
                
                for idx in indices:
                    prompts['point_coords'].append([x_coords[idx], y_coords[idx]])
                    prompts['point_labels'].append(1)  # Foreground
                
                print(f"   🎯 Generated {num_points} prompt points from CLIP attention")
        
        # 3. Add spatial keyword-based points (floor, ceiling, etc.)
        spatial_points = self.parse_spatial_keywords(text_prompt, image.shape)
        if spatial_points:
            prompts['point_coords'].extend(spatial_points['coords'])
            prompts['point_labels'].extend(spatial_points['labels'])
            print(f"   🗺️  Added {len(spatial_points['coords'])} spatial keyword points")
        
        # Convert to numpy arrays
        if prompts['point_coords']:
            prompts['point_coords'] = np.array(prompts['point_coords'], dtype=np.float32)
            prompts['point_labels'] = np.array(prompts['point_labels'], dtype=np.int32)
        else:
            # If no points, use center point
            h, w = image.shape[:2]
            prompts['point_coords'] = np.array([[w//2, h//2]], dtype=np.float32)
            prompts['point_labels'] = np.array([1], dtype=np.int32)
            print(f"   ⚠️  No prompts found, using center point")
        
        return prompts
    
    def parse_spatial_keywords(self, text: str, image_shape: Tuple) -> Dict:
        """Parse spatial keywords and generate point prompts"""
        h, w = image_shape[:2]
        text_lower = text.lower()
        
        coords = []
        labels = []
        
        # Floor/ground - bottom region
        if any(kw in text_lower for kw in ['floor', 'ground', 'bottom']):
            coords.append([w//2, int(h*0.9)])
            labels.append(1)
        
        # Ceiling/top
        if any(kw in text_lower for kw in ['ceiling', 'top', 'above']):
            coords.append([w//2, int(h*0.1)])
            labels.append(1)
        
        # Left side
        if 'left' in text_lower:
            coords.append([int(w*0.25), h//2])
            labels.append(1)
        
        # Right side
        if 'right' in text_lower:
            coords.append([int(w*0.75), h//2])
            labels.append(1)
        
        # Center
        if 'center' in text_lower or 'middle' in text_lower:
            coords.append([w//2, h//2])
            labels.append(1)
        
        return {'coords': coords, 'labels': labels}
    
    def segment_with_sam2(self, image: np.ndarray, prompts: Dict) -> Tuple[Optional[np.ndarray], float]:
        """Run SAM2 segmentation with given prompts"""
        try:
            # Ensure image is RGB and uint8
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
            
            image_rgb = image_rgb.astype(np.uint8)
            
            # Set image in SAM2
            self.sam2_predictor.set_image(image_rgb)
            
            # Prepare prompts
            if len(prompts['boxes']) > 0:
                # Use box prompts
                masks, scores, _ = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=prompts['boxes'][0],  # Use first box
                    multimask_output=False
                )
            else:
                # Use point prompts
                masks, scores, _ = self.sam2_predictor.predict(
                    point_coords=prompts['point_coords'],
                    point_labels=prompts['point_labels'],
                    multimask_output=True  # Get multiple masks to choose best
                )
            
            if len(masks) > 0:
                # Select best mask
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
                best_score = scores[best_idx]
                
                print(f"   ✅ SAM2 generated mask with score: {best_score:.3f}")
                return best_mask, best_score
            else:
                print(f"   ⚠️  SAM2 generated no masks")
                return None, 0.0
                
        except Exception as e:
            print(f"   ❌ SAM2 segmentation error: {e}")
            traceback.print_exc()
            return None, 0.0
    
    def segment_image(self, image_path: str, text_prompt: str) -> Tuple[Optional[np.ndarray], Dict]:
        """Main segmentation pipeline"""
        print(f"\n🎯 Segmenting with prompt: '{text_prompt}'")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return None, {}
        
        print(f"   📸 Image loaded: {image.shape}")
        
        # Step 1: Try GroundingDINO for object detection
        boxes = self.get_grounding_dino_boxes(image, text_prompt)
        
        # Step 2: Generate SAM2 prompts
        prompts = self.generate_sam2_prompts(image, text_prompt, boxes)
        
        # Step 3: Run SAM2 segmentation
        mask, score = self.segment_with_sam2(image, prompts)
        
        if mask is None:
            return None, {}
        
        # Create visualization
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = [0, 255, 0]  # Green mask
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        
        # Draw prompts
        if len(prompts['point_coords']) > 0:
            for coord, label in zip(prompts['point_coords'], prompts['point_labels']):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(overlay, (int(coord[0]), int(coord[1])), 8, color, -1)
        
        # Convert to RGB for Gradio
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        info = {
            'score': score,
            'num_points': len(prompts['point_coords']) if len(prompts['point_coords']) > 0 else 0,
            'num_boxes': len(prompts['boxes']) if len(prompts['boxes']) > 0 else 0,
            'mask_area': np.sum(mask) / (mask.shape[0] * mask.shape[1])
        }
        
        return overlay_rgb, info


def create_gradio_interface():
    """Create Gradio interface"""
    segmenter = SingleImageSegmentation(device="cuda" if torch.cuda.is_available() else "cpu")
    
    def process_image(image, text_prompt):
        if image is None:
            return None, "Please upload an image"
        
        if not text_prompt or text_prompt.strip() == "":
            return None, "Please enter a text prompt"
        
        # Save temporary image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Run segmentation
        result, info = segmenter.segment_image(temp_path, text_prompt)
        
        # Clean up
        Path(temp_path).unlink()
        
        if result is None:
            return None, "Segmentation failed"
        
        info_text = f"""
**Segmentation Results:**
- Confidence Score: {info['score']:.3f}
- Prompt Points: {info['num_points']}
- Bounding Boxes: {info['num_boxes']}
- Mask Coverage: {info['mask_area']*100:.1f}%
        """
        
        return result, info_text
    
    with gr.Blocks(title="Single Image Segmentation") as demo:
        gr.Markdown("# 🎯 Single Image Prompt-based Segmentation")
        gr.Markdown("Using CLIP + SAM2 + GroundingDINO for intelligent object segmentation")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="numpy")
                text_prompt = gr.Textbox(
                    label="Segmentation Prompt",
                    placeholder="e.g., 'floor', 'blue box', 'person on the left'",
                    lines=2
                )
                segment_btn = gr.Button("🚀 Segment", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Segmentation Result")
                output_info = gr.Markdown()
        
        segment_btn.click(
            fn=process_image,
            inputs=[input_image, text_prompt],
            outputs=[output_image, output_info]
        )
        
        gr.Markdown("""
        ### 📝 Usage:
        1. Upload an image
        2. Enter a text prompt describing what to segment
        3. Click "Segment" to see results
        
        **Supported prompts:**
        - Objects: "chair", "table", "person", "blue box"
        - Spatial: "floor", "ceiling", "left side", "right wall"
        - Combined: "red object on the floor", "person on the left"
        """)
    
    return demo


if __name__ == "__main__":
    print("🚀 Launching Single Image Segmentation")
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)

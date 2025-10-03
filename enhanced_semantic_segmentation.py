#!/usr/bin/env python3
"""
Enhanced Semantic Segmentation System
Combining RAM, CLIP, CLIPSeg, SAM2, GroundingDINO, and molmo for comprehensive prompt-based segmentation
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import gradio as gr
from PIL import Image
import sys
import traceback
import json
import time
from datetime import datetime

# Add paths for local modules
sys.path.insert(0, str(Path(__file__).parent / "sam2"))
sys.path.insert(0, str(Path(__file__).parent / "vqasynth"))
sys.path.insert(0, str(Path(__file__).parent))

class EnhancedSemanticSegmentation:
    """Enhanced segmentation system with multiple model integration"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.models = {}
        self.setup_models()
        
    def setup_models(self):
        """Initialize all available models"""
        print("🔧 Initializing Enhanced Segmentation Models...")
        
        # 1. CLIP for semantic understanding
        self._setup_clip()
        
        # 2. SAM2 for high-quality segmentation
        self._setup_sam2()
        
        # 3. GroundingDINO for object detection
        self._setup_grounding_dino()
        
        # 4. RAM for object recognition
        self._setup_ram()
        
        # 5. CLIPSeg for semantic segmentation
        self._setup_clipseg()
        
        # 6. molmo for enhanced reasoning (if available)
        self._setup_molmo()
        
        print("✅ Model initialization complete!")
    
    def _setup_clip(self):
        """Setup CLIP model"""
        try:
            import clip
            self.models['clip'] = {}
            self.models['clip']['model'], self.models['clip']['preprocess'] = clip.load("ViT-L/14@336px", device=self.device)
            print("✅ CLIP initialized")
        except Exception as e:
            print(f"❌ CLIP initialization failed: {e}")
            self.models['clip'] = None
    
    def _setup_sam2(self):
        """Setup SAM2 model"""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            
            if Path(sam2_checkpoint).exists():
                sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
                self.models['sam2'] = SAM2ImagePredictor(sam2_model)
                print("✅ SAM2 initialized")
            else:
                print("⚠️  SAM2 checkpoint not found")
                self.models['sam2'] = None
        except Exception as e:
            print(f"❌ SAM2 initialization failed: {e}")
            self.models['sam2'] = None
    
    def _setup_grounding_dino(self):
        """Setup GroundingDINO model"""
        try:
            import groundingdino
            from groundingdino.util.inference import Model as GroundingDINOModel
            
            config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
            
            if Path(config_path).exists() and Path(checkpoint_path).exists():
                self.models['grounding_dino'] = GroundingDINOModel(
                    model_config_path=config_path,
                    model_checkpoint_path=checkpoint_path,
                    device=self.device
                )
                print("✅ GroundingDINO initialized")
            else:
                print("⚠️  GroundingDINO not found")
                self.models['grounding_dino'] = None
        except Exception as e:
            print(f"⚠️  GroundingDINO not available: {e}")
            self.models['grounding_dino'] = None
    
    def _setup_ram(self):
        """Setup RAM model"""
        try:
            from ram.models import ram
            from ram import inference_ram
            
            ram_checkpoint = 'ram_swin_large_14m.pth'
            if Path(ram_checkpoint).exists():
                self.models['ram'] = ram(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
                self.models['ram'].eval()
                self.models['ram'] = self.models['ram'].to(self.device)
                print("✅ RAM initialized")
            else:
                print("⚠️  RAM checkpoint not found")
                self.models['ram'] = None
        except Exception as e:
            print(f"⚠️  RAM not available: {e}")
            self.models['ram'] = None
    
    def _setup_clipseg(self):
        """Setup CLIPSeg model"""
        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            
            self.models['clipseg'] = {}
            self.models['clipseg']['processor'] = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
            self.models['clipseg']['model'] = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
            self.models['clipseg']['model'] = self.models['clipseg']['model'].to(self.device)
            print("✅ CLIPSeg initialized")
        except Exception as e:
            print(f"⚠️  CLIPSeg not available: {e}")
            self.models['clipseg'] = None
    
    def _setup_molmo(self):
        """Setup molmo model (if available)"""
        try:
            # Check if molmo is available
            molmo_path = Path("molmo")
            if molmo_path.exists():
                # Add molmo to path and try to import
                sys.path.insert(0, str(molmo_path))
                # This would need to be adapted based on actual molmo interface
                print("✅ molmo path found (integration pending)")
                self.models['molmo'] = None  # Placeholder
            else:
                print("⚠️  molmo not found")
                self.models['molmo'] = None
        except Exception as e:
            print(f"⚠️  molmo not available: {e}")
            self.models['molmo'] = None
    
    def get_ram_tags(self, image: np.ndarray) -> List[str]:
        """Get object tags from RAM model"""
        if self.models['ram'] is None:
            return []
        
        try:
            from ram import inference_ram
            from ram.models import ram
            
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Get tags
            tags = inference_ram(pil_image, self.models['ram'])
            return tags
        except Exception as e:
            print(f"⚠️  RAM tagging failed: {e}")
            return []
    
    def get_grounding_dino_boxes(self, image: np.ndarray, text_prompt: str, 
                                  box_threshold: float = 0.25, 
                                  text_threshold: float = 0.25) -> List[np.ndarray]:
        """Get bounding boxes from GroundingDINO"""
        if self.models['grounding_dino'] is None:
            return []
        
        try:
            detections = self.models['grounding_dino'].predict_with_classes(
                image=image,
                classes=[text_prompt],
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            boxes = []
            if hasattr(detections, 'xyxy'):
                boxes = detections.xyxy.cpu().numpy()
            
            return boxes
        except Exception as e:
            print(f"⚠️  GroundingDINO detection failed: {e}")
            return []
    
    def get_clip_attention_map(self, image: np.ndarray, text_prompt: str) -> Tuple[float, np.ndarray]:
        """Get CLIP attention map and similarity score"""
        if self.models['clip'] is None:
            return 0.0, np.zeros((image.shape[0], image.shape[1]))
        
        try:
            # Prepare image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_input = self.models['clip']['preprocess'](pil_image).unsqueeze(0).to(self.device)
            
            # Create text variations for better matching
            text_variations = [
                text_prompt,
                f"a photo of {text_prompt}",
                f"an image containing {text_prompt}",
                f"{text_prompt} in the scene",
                f"a {text_prompt} object"
            ]
            
            # Tokenize all variations
            import clip
            text_inputs = clip.tokenize(text_variations).to(self.device)
            
            # Compute similarities
            with torch.no_grad():
                image_features = self.models['clip']['model'].encode_image(image_input)
                text_features = self.models['clip']['model'].encode_text(text_inputs)
                
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
            print(f"❌ CLIP attention computation failed: {e}")
            return 0.0, np.zeros((image.shape[0], image.shape[1]))
    
    def get_clipseg_mask(self, image: np.ndarray, text_prompt: str) -> Optional[np.ndarray]:
        """Get segmentation mask from CLIPSeg"""
        if self.models['clipseg'] is None:
            return None
        
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Process with CLIPSeg
            inputs = self.models['clipseg']['processor'](
                text=[text_prompt], 
                images=[pil_image], 
                padding=True, 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.models['clipseg']['model'](**inputs)
                preds = outputs.logits.unsqueeze(1)
                preds = torch.sigmoid(preds)
                preds = (preds > 0.5).float()
            
            # Convert to numpy
            mask = preds.squeeze().cpu().numpy()
            
            # Resize to original image size
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            return mask.astype(np.uint8)
            
        except Exception as e:
            print(f"⚠️  CLIPSeg segmentation failed: {e}")
            return None
    
    def generate_enhanced_prompts(self, image: np.ndarray, text_prompt: str) -> Dict:
        """Generate enhanced prompts using multiple models"""
        prompts = {
            'point_coords': [],
            'point_labels': [],
            'boxes': [],
            'clipseg_mask': None,
            'ram_tags': [],
            'confidence_scores': {}
        }
        
        print(f"🎯 Generating enhanced prompts for: '{text_prompt}'")
        
        # 1. Get RAM tags for context
        ram_tags = self.get_ram_tags(image)
        prompts['ram_tags'] = ram_tags
        if ram_tags:
            print(f"   🏷️  RAM tags: {', '.join(ram_tags[:5])}")  # Show first 5 tags
        
        # 2. Try GroundingDINO for object detection
        boxes = self.get_grounding_dino_boxes(image, text_prompt)
        if boxes:
            prompts['boxes'] = boxes
            prompts['confidence_scores']['grounding_dino'] = 0.8  # Placeholder
            print(f"   📦 GroundingDINO found {len(boxes)} boxes")
        
        # 3. Get CLIP attention map
        clip_similarity, attention_map = self.get_clip_attention_map(image, text_prompt)
        prompts['confidence_scores']['clip'] = clip_similarity
        print(f"   📊 CLIP similarity: {clip_similarity:.3f}")
        
        # 4. Get CLIPSeg mask
        clipseg_mask = self.get_clipseg_mask(image, text_prompt)
        if clipseg_mask is not None:
            prompts['clipseg_mask'] = clipseg_mask
            prompts['confidence_scores']['clipseg'] = 0.7  # Placeholder
            print(f"   🎨 CLIPSeg mask generated")
        
        # 5. Generate point prompts from CLIP attention
        if clip_similarity > 0.15:
            threshold = np.percentile(attention_map, 75)
            high_attention = attention_map > threshold
            
            y_coords, x_coords = np.where(high_attention)
            
            if len(y_coords) > 0:
                num_points = min(15, len(y_coords))
                indices = np.random.choice(len(y_coords), num_points, replace=False)
                
                for idx in indices:
                    prompts['point_coords'].append([x_coords[idx], y_coords[idx]])
                    prompts['point_labels'].append(1)
                
                print(f"   🎯 Generated {num_points} CLIP attention points")
        
        # 6. Add spatial keyword points
        spatial_points = self.parse_spatial_keywords(text_prompt, image.shape)
        if spatial_points:
            prompts['point_coords'].extend(spatial_points['coords'])
            prompts['point_labels'].extend(spatial_points['labels'])
            print(f"   🗺️  Added {len(spatial_points['coords'])} spatial points")
        
        # 7. Convert to numpy arrays
        if prompts['point_coords']:
            prompts['point_coords'] = np.array(prompts['point_coords'], dtype=np.float32)
            prompts['point_labels'] = np.array(prompts['point_labels'], dtype=np.int32)
        else:
            # Fallback to center point
            h, w = image.shape[:2]
            prompts['point_coords'] = np.array([[w//2, h//2]], dtype=np.float32)
            prompts['point_labels'] = np.array([1], dtype=np.int32)
            print(f"   ⚠️  No points found, using center point")
        
        return prompts
    
    def parse_spatial_keywords(self, text: str, image_shape: Tuple) -> Dict:
        """Parse spatial keywords and generate point prompts"""
        h, w = image_shape[:2]
        text_lower = text.lower()
        
        coords = []
        labels = []
        
        # Floor/ground - bottom region
        if any(kw in text_lower for kw in ['floor', 'ground', 'bottom', 'down']):
            coords.append([w//2, int(h*0.9)])
            labels.append(1)
        
        # Ceiling/top
        if any(kw in text_lower for kw in ['ceiling', 'top', 'above', 'up']):
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
        
        # Foreground/background
        if 'foreground' in text_lower or 'front' in text_lower:
            coords.append([w//2, int(h*0.7)])
            labels.append(1)
        
        if 'background' in text_lower or 'back' in text_lower:
            coords.append([w//2, int(h*0.3)])
            labels.append(1)
        
        return {'coords': coords, 'labels': labels}
    
    def segment_with_sam2(self, image: np.ndarray, prompts: Dict) -> Tuple[Optional[np.ndarray], float]:
        """Run SAM2 segmentation with enhanced prompts"""
        if self.models['sam2'] is None:
            return None, 0.0
        
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
            self.models['sam2'].set_image(image_rgb)
            
            # Prepare prompts
            if len(prompts['boxes']) > 0:
                # Use box prompts
                masks, scores, _ = self.models['sam2'].predict(
                    point_coords=None,
                    point_labels=None,
                    box=prompts['boxes'][0],
                    multimask_output=False
                )
            else:
                # Use point prompts
                masks, scores, _ = self.models['sam2'].predict(
                    point_coords=prompts['point_coords'],
                    point_labels=prompts['point_labels'],
                    multimask_output=True
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
            return None, 0.0
    
    def fuse_segmentation_results(self, sam2_mask: Optional[np.ndarray], 
                                  clipseg_mask: Optional[np.ndarray],
                                  sam2_score: float) -> Tuple[np.ndarray, Dict]:
        """Fuse results from multiple segmentation models"""
        fusion_info = {
            'models_used': [],
            'fusion_method': 'none',
            'confidence': sam2_score
        }
        
        if sam2_mask is not None and clipseg_mask is not None:
            # Fuse SAM2 and CLIPSeg results
            fusion_info['models_used'] = ['sam2', 'clipseg']
            fusion_info['fusion_method'] = 'intersection'
            
            # Intersection of both masks
            fused_mask = np.logical_and(sam2_mask, clipseg_mask).astype(np.uint8)
            
            # If intersection is too small, use union
            if np.sum(fused_mask) < 100:
                fused_mask = np.logical_or(sam2_mask, clipseg_mask).astype(np.uint8)
                fusion_info['fusion_method'] = 'union'
            
            fusion_info['confidence'] = (sam2_score + 0.7) / 2  # Average confidence
            
        elif sam2_mask is not None:
            fused_mask = sam2_mask
            fusion_info['models_used'] = ['sam2']
            fusion_info['fusion_method'] = 'sam2_only'
            
        elif clipseg_mask is not None:
            fused_mask = clipseg_mask
            fusion_info['models_used'] = ['clipseg']
            fusion_info['fusion_method'] = 'clipseg_only'
            fusion_info['confidence'] = 0.7
            
        else:
            # No segmentation available
            return np.zeros((100, 100), dtype=np.uint8), fusion_info
        
        return fused_mask, fusion_info
    
    def segment_image(self, image_path: str, text_prompt: str) -> Tuple[Optional[np.ndarray], Dict]:
        """Main enhanced segmentation pipeline"""
        print(f"\n🎯 Enhanced Segmentation with prompt: '{text_prompt}'")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return None, {}
        
        print(f"   📸 Image loaded: {image.shape}")
        
        # Generate enhanced prompts
        prompts = self.generate_enhanced_prompts(image, text_prompt)
        
        # Run SAM2 segmentation
        sam2_mask, sam2_score = self.segment_with_sam2(image, prompts)
        
        # Get CLIPSeg mask
        clipseg_mask = prompts.get('clipseg_mask')
        
        # Fuse results
        final_mask, fusion_info = self.fuse_segmentation_results(
            sam2_mask, clipseg_mask, sam2_score
        )
        
        if np.sum(final_mask) == 0:
            return None, {}
        
        # Create enhanced visualization
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[final_mask] = [0, 255, 0]  # Green mask
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        
        # Draw prompts
        if len(prompts['point_coords']) > 0:
            for coord, label in zip(prompts['point_coords'], prompts['point_labels']):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(overlay, (int(coord[0]), int(coord[1])), 8, color, -1)
        
        # Draw bounding boxes
        if len(prompts['boxes']) > 0:
            for box in prompts['boxes']:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Convert to RGB for Gradio
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Prepare comprehensive info
        info = {
            'score': fusion_info['confidence'],
            'models_used': ', '.join(fusion_info['models_used']),
            'fusion_method': fusion_info['fusion_method'],
            'num_points': len(prompts['point_coords']) if len(prompts['point_coords']) > 0 else 0,
            'num_boxes': len(prompts['boxes']) if len(prompts['boxes']) > 0 else 0,
            'mask_area': np.sum(final_mask) / (final_mask.shape[0] * final_mask.shape[1]),
            'ram_tags': ', '.join(prompts['ram_tags'][:5]) if prompts['ram_tags'] else 'None',
            'confidence_scores': prompts['confidence_scores']
        }
        
        return overlay_rgb, info


def create_enhanced_gradio_interface():
    """Create enhanced Gradio interface"""
    segmenter = EnhancedSemanticSegmentation(device="cuda" if torch.cuda.is_available() else "cpu")
    
    def process_image(image, text_prompt, use_ram, use_grounding_dino, use_clipseg):
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
**Enhanced Segmentation Results:**
- **Confidence Score**: {info['score']:.3f}
- **Models Used**: {info['models_used']}
- **Fusion Method**: {info['fusion_method']}
- **Prompt Points**: {info['num_points']}
- **Bounding Boxes**: {info['num_boxes']}
- **Mask Coverage**: {info['mask_area']*100:.1f}%
- **RAM Tags**: {info['ram_tags']}

**Model Confidence Scores:**
{json.dumps(info['confidence_scores'], indent=2)}
        """
        
        return result, info_text
    
    with gr.Blocks(title="Enhanced Semantic Segmentation") as demo:
        gr.Markdown("# 🎯 Enhanced Semantic Segmentation System")
        gr.Markdown("Combining RAM, CLIP, CLIPSeg, SAM2, and GroundingDINO for intelligent object segmentation")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="numpy")
                text_prompt = gr.Textbox(
                    label="Segmentation Prompt",
                    placeholder="e.g., 'floor', 'blue box', 'person on the left', 'red chair'",
                    lines=2
                )
                
                with gr.Row():
                    use_ram = gr.Checkbox(label="Use RAM", value=True)
                    use_grounding_dino = gr.Checkbox(label="Use GroundingDINO", value=True)
                    use_clipseg = gr.Checkbox(label="Use CLIPSeg", value=True)
                
                segment_btn = gr.Button("🚀 Segment", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Segmentation Result")
                output_info = gr.Markdown()
        
        segment_btn.click(
            fn=process_image,
            inputs=[input_image, text_prompt, use_ram, use_grounding_dino, use_clipseg],
            outputs=[output_image, output_info]
        )
        
        gr.Markdown("""
        ### 📝 Usage:
        1. Upload an image
        2. Enter a text prompt describing what to segment
        3. Select which models to use (optional)
        4. Click "Segment" to see results
        
        **Supported prompts:**
        - **Objects**: "chair", "table", "person", "blue box", "red car"
        - **Spatial**: "floor", "ceiling", "left side", "right wall", "center"
        - **Combined**: "red object on the floor", "person on the left", "blue box in center"
        - **Complex**: "red chair on the left side of the room"
        
        **Models:**
        - **RAM**: Object recognition and tagging
        - **CLIP**: Semantic understanding and attention
        - **SAM2**: High-quality segmentation
        - **GroundingDINO**: Object detection and localization
        - **CLIPSeg**: Semantic segmentation
        """)
    
    return demo


if __name__ == "__main__":
    print("🚀 Launching Enhanced Semantic Segmentation")
    demo = create_enhanced_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7861)

"""
Multi-View Spatial Segmentation System

This module combines VGGT 3D reconstruction, RAM object recognition, CLIP semantic understanding,
and SAM2 segmentation for prompt-based multi-view segmentation with spatial reasoning.

The system processes multiple images or video sequences, performs 3D reconstruction,
and enables complex spatial description-based segmentation across all views.
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import clip
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Add paths for various components
sys.path.append("vggt/")
sys.path.append("sam2/")
sys.path.append("vqasynth/")

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# VQASynth imports
from vqasynth.embeddings import EmbeddingGenerator, TagFilter


class SpatialPromptProcessor:
    """Processes complex spatial descriptions to identify objects and their relationships."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=device)
        self.embedding_generator = EmbeddingGenerator(model_name="ViT-L/14@336px", device=device)
        
        # Spatial relationship keywords
        self.spatial_keywords = {
            'left': ['left', 'left side', 'leftmost', 'to the left'],
            'right': ['right', 'right side', 'rightmost', 'to the right'],
            'above': ['above', 'on top', 'higher', 'upper', 'over'],
            'below': ['below', 'under', 'lower', 'beneath', 'underneath'],
            'front': ['front', 'in front', 'foreground', 'closer', 'nearer'],
            'back': ['back', 'behind', 'background', 'farther', 'further'],
            'center': ['center', 'middle', 'central', 'in the center'],
            'near': ['near', 'close', 'nearby', 'adjacent', 'next to'],
            'far': ['far', 'distant', 'away', 'remote']
        }
        
        # Color keywords
        self.color_keywords = [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown',
            'black', 'white', 'gray', 'grey', 'silver', 'gold', 'cyan', 'magenta'
        ]
        
        # Object keywords (expandable)
        self.object_keywords = [
            'person', 'man', 'woman', 'child', 'chair', 'table', 'car', 'truck',
            'box', 'container', 'building', 'tree', 'plant', 'door', 'window',
            'wall', 'floor', 'ceiling', 'shelf', 'cabinet', 'monitor', 'screen',
            'book', 'laptop', 'phone', 'bottle', 'cup', 'plate', 'fork', 'knife'
        ]
    
    def parse_spatial_prompt(self, prompt: str) -> Dict[str, List[str]]:
        """Parse spatial description into components."""
        prompt_lower = prompt.lower()
        
        parsed = {
            'objects': [],
            'colors': [],
            'spatial_relations': [],
            'descriptors': []
        }
        
        # Extract objects
        for obj in self.object_keywords:
            if obj in prompt_lower:
                parsed['objects'].append(obj)
        
        # Extract colors
        for color in self.color_keywords:
            if color in prompt_lower:
                parsed['colors'].append(color)
        
        # Extract spatial relations
        for relation, keywords in self.spatial_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    parsed['spatial_relations'].append(relation)
                    break
        
        # Store full prompt as descriptor
        parsed['descriptors'].append(prompt)
        
        return parsed
    
    def compute_text_similarity(self, text_prompt: str, image_features: torch.Tensor) -> float:
        """Compute similarity between text prompt and image features."""
        text_tokens = clip.tokenize([text_prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
        return similarity.item()


class MultiViewSegmentationPipeline:
    """Main pipeline for multi-view spatial segmentation."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.vggt_model = None
        self.sam2_predictor = None
        self.spatial_processor = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize all required models."""
        print("Initializing models...")
        
        # Initialize VGGT
        self.vggt_model = VGGT()
        vggt_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(vggt_url))
        self.vggt_model.eval()
        self.vggt_model = self.vggt_model.to(self.device)
        
        # Initialize SAM2
        sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        if os.path.exists(sam2_checkpoint):
            sam2_model = build_sam2("sam2.1_hiera_l.yaml", sam2_checkpoint, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        else:
            print("Warning: SAM2 checkpoint not found. SAM2 features will be limited.")
            self.sam2_predictor = None
        
        # Initialize spatial processor
        self.spatial_processor = SpatialPromptProcessor(device=self.device)
        
        print("Models initialized successfully!")
    
    def process_input(self, input_data: Union[str, List[str]]) -> Tuple[str, List[str]]:
        """Process input video or images into a temporary directory structure."""
        timestamp = torch.randint(0, 1000000, (1,)).item()
        target_dir = f"temp_multiview_{timestamp}"
        target_dir_images = os.path.join(target_dir, "images")
        
        os.makedirs(target_dir_images, exist_ok=True)
        
        image_paths = []
        
        if isinstance(input_data, str) and input_data.endswith(('.mp4', '.avi', '.mov')):
            # Process video
            vs = cv2.VideoCapture(input_data)
            fps = vs.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps * 0.5))  # 2 frames per second
            
            count = 0
            frame_num = 0
            while True:
                ret, frame = vs.read()
                if not ret:
                    break
                count += 1
                if count % frame_interval == 0:
                    image_path = os.path.join(target_dir_images, f"frame_{frame_num:06d}.png")
                    cv2.imwrite(image_path, frame)
                    image_paths.append(image_path)
                    frame_num += 1
            vs.release()
        else:
            # Process image list
            if isinstance(input_data, str):
                input_data = [input_data]
            
            for i, img_path in enumerate(input_data):
                if os.path.exists(img_path):
                    dst_path = os.path.join(target_dir_images, f"image_{i:06d}.png")
                    img = cv2.imread(img_path)
                    cv2.imwrite(dst_path, img)
                    image_paths.append(dst_path)
        
        return target_dir, sorted(image_paths)
    
    def run_3d_reconstruction(self, target_dir: str) -> Dict:
        """Run VGGT 3D reconstruction on images."""
        print("Running 3D reconstruction...")
        
        # Load and preprocess images
        image_names = [f for f in os.listdir(os.path.join(target_dir, "images")) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_names = sorted([os.path.join(target_dir, "images", f) for f in image_names])
        
        if len(image_names) == 0:
            raise ValueError("No images found for reconstruction")
        
        images = load_and_preprocess_images(image_names).to(self.device)
        
        # Run inference
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.vggt_model(images)
        
        # Convert pose encoding to extrinsic and intrinsic matrices
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        
        # Convert tensors to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        
        # Generate world points from depth map
        depth_map = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points
        predictions["image_paths"] = image_names
        
        print(f"3D reconstruction complete. Processed {len(image_names)} images.")
        return predictions
    
    def find_objects_in_3d(self, predictions: Dict, spatial_prompt: str) -> List[Dict]:
        """Find objects in 3D space based on spatial prompt."""
        print(f"Processing spatial prompt: '{spatial_prompt}'")
        
        # Parse the spatial prompt
        parsed_prompt = self.spatial_processor.parse_spatial_prompt(spatial_prompt)
        print(f"Parsed prompt: {parsed_prompt}")
        
        # Get image features for each view
        image_paths = predictions["image_paths"]
        world_points = predictions["world_points_from_depth"]  # Shape: (num_views, H, W, 3)
        depth_confidence = predictions.get("depth_conf", None)
        
        candidate_regions = []
        
        for view_idx, img_path in enumerate(image_paths):
            # Load image and compute CLIP features
            image = Image.open(img_path).convert("RGB")
            image_input = self.spatial_processor.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.spatial_processor.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity with spatial prompt
            similarity_score = self.spatial_processor.compute_text_similarity(
                spatial_prompt, image_features
            )
            
            if similarity_score > 0.2:  # Threshold for relevance
                # Get 3D points for this view
                view_world_points = world_points[view_idx]  # Shape: (H, W, 3)
                view_confidence = depth_confidence[view_idx] if depth_confidence is not None else None
                
                candidate_regions.append({
                    'view_idx': view_idx,
                    'image_path': img_path,
                    'similarity_score': similarity_score,
                    'world_points': view_world_points,
                    'confidence': view_confidence,
                    'image_features': image_features.cpu().numpy()
                })
        
        # Sort by similarity score
        candidate_regions.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        print(f"Found {len(candidate_regions)} candidate regions")
        return candidate_regions
    
    def project_3d_to_views(self, candidate_regions: List[Dict], predictions: Dict) -> Dict[int, List[np.ndarray]]:
        """Project 3D object locations back to all 2D views."""
        print("Projecting 3D locations to all views...")
        
        view_projections = {}
        
        if not candidate_regions:
            return view_projections
        
        # Use the best candidate region as reference
        best_region = candidate_regions[0]
        reference_points = best_region['world_points']
        
        # Get high-confidence points from the reference
        if best_region['confidence'] is not None:
            high_conf_mask = best_region['confidence'] > 0.5
        else:
            high_conf_mask = np.ones_like(reference_points[:, :, 0], dtype=bool)
        
        # Extract high-confidence 3D points
        valid_points_3d = reference_points[high_conf_mask]
        
        if len(valid_points_3d) == 0:
            return view_projections
        
        # Project to all views
        extrinsics = predictions["extrinsic"]
        intrinsics = predictions["intrinsic"]
        
        for view_idx in range(len(predictions["image_paths"])):
            # Transform world points to camera coordinates
            extrinsic = extrinsics[view_idx]
            intrinsic = intrinsics[view_idx]
            
            # Convert to homogeneous coordinates
            points_3d_homo = np.hstack([valid_points_3d, np.ones((len(valid_points_3d), 1))])
            
            # Transform to camera coordinates
            cam_points = (extrinsic @ points_3d_homo.T).T
            cam_points = cam_points[:, :3]  # Remove homogeneous dimension
            
            # Project to image plane
            if cam_points.shape[0] > 0:
                # Filter points in front of camera
                valid_depth = cam_points[:, 2] > 0.1
                if np.any(valid_depth):
                    valid_cam_points = cam_points[valid_depth]
                    
                    # Project using intrinsic matrix
                    img_points = (intrinsic @ valid_cam_points.T).T
                    img_points = img_points[:, :2] / img_points[:, 2:3]  # Perspective division
                    
                    # Filter points within image bounds
                    H, W = reference_points.shape[:2]
                    valid_img = (
                        (img_points[:, 0] >= 0) & (img_points[:, 0] < W) &
                        (img_points[:, 1] >= 0) & (img_points[:, 1] < H)
                    )
                    
                    if np.any(valid_img):
                        final_points = img_points[valid_img]
                        view_projections[view_idx] = [final_points]
        
        print(f"Projected to {len(view_projections)} views")
        return view_projections
    
    def segment_with_sam2(self, image_path: str, points: np.ndarray) -> np.ndarray:
        """Use SAM2 to segment based on projected points."""
        if self.sam2_predictor is None:
            # Fallback: create simple circular masks around points
            image = cv2.imread(image_path)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            for point in points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(mask, (x, y), 20, 255, -1)
            
            return mask.astype(bool)
        
        # Use SAM2 for proper segmentation
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.sam2_predictor.set_image(image_rgb)
        
        # Use points as prompts for SAM2
        input_points = points.astype(np.float32)
        input_labels = np.ones(len(input_points), dtype=np.int32)  # All positive points
        
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        # Return the mask with highest score
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx]
    
    def run_segmentation(self, input_data: Union[str, List[str]], spatial_prompt: str) -> Dict:
        """Main segmentation pipeline."""
        print("Starting multi-view segmentation pipeline...")
        
        # Process input
        target_dir, image_paths = self.process_input(input_data)
        
        try:
            # Run 3D reconstruction
            predictions = self.run_3d_reconstruction(target_dir)
            
            # Find objects based on spatial prompt
            candidate_regions = self.find_objects_in_3d(predictions, spatial_prompt)
            
            # Project 3D locations to all views
            view_projections = self.project_3d_to_views(candidate_regions, predictions)
            
            # Generate segmentation masks for each view
            segmentation_results = {}
            
            for view_idx, img_path in enumerate(image_paths):
                if view_idx in view_projections:
                    points = view_projections[view_idx][0]
                    if len(points) > 0:
                        mask = self.segment_with_sam2(img_path, points)
                        segmentation_results[view_idx] = {
                            'image_path': img_path,
                            'mask': mask,
                            'points': points
                        }
            
            return {
                'target_dir': target_dir,
                'predictions': predictions,
                'candidate_regions': candidate_regions,
                'segmentation_results': segmentation_results,
                'spatial_prompt': spatial_prompt
            }
        
        finally:
            # Cleanup temporary directory
            import shutil
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
    
    def save_results(self, results: Dict, output_dir: str):
        """Save segmentation results to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual masks and overlays
        for view_idx, seg_result in results['segmentation_results'].items():
            # Load original image
            image = cv2.imread(seg_result['image_path'])
            mask = seg_result['mask']
            
            # Save mask
            mask_path = os.path.join(output_dir, f"mask_view_{view_idx:03d}.png")
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
            
            # Create overlay
            overlay = image.copy()
            overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
            
            # Draw points
            for point in seg_result['points']:
                x, y = int(point[0]), int(point[1])
                cv2.circle(overlay, (x, y), 5, (255, 0, 0), -1)
            
            overlay_path = os.path.join(output_dir, f"overlay_view_{view_idx:03d}.png")
            cv2.imwrite(overlay_path, overlay)
        
        # Save 3D point cloud with segmented regions
        world_points = results['predictions']['world_points_from_depth']
        
        # Create a combined point cloud
        all_points = []
        all_colors = []
        
        for view_idx, seg_result in results['segmentation_results'].items():
            view_points = world_points[view_idx]
            mask = seg_result['mask']
            
            # Get segmented 3D points
            segmented_points = view_points[mask]
            segmented_colors = np.full((len(segmented_points), 3), [255, 0, 0])  # Red for segmented
            
            all_points.append(segmented_points.reshape(-1, 3))
            all_colors.append(segmented_colors)
        
        if all_points:
            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors)
            
            # Save as PLY file
            ply_path = os.path.join(output_dir, "segmented_pointcloud.ply")
            self.save_ply(combined_points, combined_colors, ply_path)
        
        print(f"Results saved to {output_dir}")
    
    def save_ply(self, points: np.ndarray, colors: np.ndarray, filepath: str):
        """Save point cloud as PLY file."""
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for point, color in zip(points, colors):
                f.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")


if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = MultiViewSegmentationPipeline(device=device)
    
    # Example with images
    # image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
    # results = pipeline.run_segmentation(image_paths, "red chair on the left side")
    
    # Example with video
    # results = pipeline.run_segmentation("path/to/video.mp4", "person walking in the background")
    
    # pipeline.save_results(results, "output_segmentation/")
    
    print("Multi-view segmentation pipeline ready!")
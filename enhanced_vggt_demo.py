"""
Enhanced VGGT Demo with Multi-View Spatial Segmentation

This integrates the spatial segmentation capabilities with the existing VGGT demo,
allowing users to perform both 3D reconstruction and spatial segmentation in one interface.
"""

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import warnings
warnings.filterwarnings('ignore')

# Add paths for VGGT and other components
sys.path.append("vggt/")
sys.path.append("../")

# VGGT imports
try:
    from visual_util import predictions_to_glb
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    VGGT_AVAILABLE = True
except ImportError as e:
    print(f"VGGT imports failed: {e}")
    VGGT_AVAILABLE = False

# VQASynth imports
try:
    from vqasynth.embeddings import EmbeddingGenerator, TagFilter
    import clip
    VQASYNTH_AVAILABLE = True
except ImportError:
    print("VQASynth components not available. Spatial segmentation features will be limited.")
    VQASYNTH_AVAILABLE = False

# SAM2 imports
try:
    sys.path.append("../sam2/")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("SAM2 not available. Segmentation will use basic methods.")
    SAM2_AVAILABLE = False

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval()
model = model.to(device)

# Initialize spatial segmentation components
sam2_predictor = None
clip_model = None
clip_preprocess = None

if SAM2_AVAILABLE:
    try:
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        if os.path.exists(sam2_checkpoint):
            sam2_model = build_sam2("sam2.1_hiera_l.yaml", sam2_checkpoint, device=device)
            sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("✅ SAM2 initialized successfully")
    except Exception as e:
        print(f"SAM2 initialization failed: {e}")

if VQASYNTH_AVAILABLE:
    try:
        clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
        print("✅ CLIP initialized successfully")
    except Exception as e:
        print(f"CLIP initialization failed: {e}")


class SpatialSegmentationProcessor:
    """Handles spatial segmentation within the VGGT pipeline."""
    
    def __init__(self):
        self.spatial_keywords = {
            'left': ['left', 'left side', 'leftmost', 'to the left'],
            'right': ['right', 'right side', 'rightmost', 'to the right'],
            'above': ['above', 'on top', 'higher', 'upper', 'over'],
            'below': ['below', 'under', 'lower', 'beneath', 'underneath'],
            'front': ['front', 'in front', 'foreground', 'closer', 'nearer'],
            'back': ['back', 'behind', 'background', 'farther', 'further'],
            'center': ['center', 'middle', 'central', 'in the center'],
        }
        
        self.color_keywords = [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown',
            'black', 'white', 'gray', 'grey', 'silver', 'gold'
        ]
    
    def parse_prompt(self, prompt):
        """Parse spatial description."""
        prompt_lower = prompt.lower()
        return {
            'colors': [c for c in self.color_keywords if c in prompt_lower],
            'spatial_relations': [r for r, keywords in self.spatial_keywords.items() 
                                for k in keywords if k in prompt_lower],
            'full_prompt': prompt
        }
    
    def compute_similarity(self, prompt, image_path):
        """Compute CLIP similarity between prompt and image."""
        if clip_model is None:
            return 0.3  # Default similarity
        
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            image_input = clip_preprocess(image).unsqueeze(0).to(device)
            text_input = clip.tokenize([prompt]).to(device)
            
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                text_features = clip_model.encode_text(text_input)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
                return similarity.item()
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.3
    
    def segment_objects(self, predictions, spatial_prompt, target_dir):
        """Perform spatial segmentation on the reconstructed scene."""
        if not spatial_prompt.strip():
            return None
        
        print(f"Processing spatial prompt: '{spatial_prompt}'")
        
        # Get image paths
        image_dir = os.path.join(target_dir, "images")
        image_names = sorted(glob.glob(os.path.join(image_dir, "*")))
        
        segmentation_results = {}
        world_points = predictions.get("world_points_from_depth")
        
        for i, img_path in enumerate(image_names):
            # Compute similarity
            similarity = self.compute_similarity(spatial_prompt, img_path)
            
            if similarity > 0.2:  # Threshold for relevance
                # Generate segmentation mask
                mask = self.generate_mask(img_path, spatial_prompt, predictions, i)
                
                if mask is not None:
                    # Get 3D points for segmented regions
                    if world_points is not None and i < len(world_points):
                        view_3d_points = world_points[i][mask] if mask.shape == world_points[i].shape[:2] else None
                    else:
                        view_3d_points = None
                    
                    segmentation_results[i] = {
                        'image_path': img_path,
                        'mask': mask,
                        'similarity': similarity,
                        'world_points': view_3d_points
                    }
        
        return segmentation_results
    
    def generate_mask(self, image_path, prompt, predictions, view_idx):
        """Generate segmentation mask for an image."""
        try:
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            
            if sam2_predictor is not None:
                # Use SAM2 for segmentation
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                sam2_predictor.set_image(image_rgb)
                
                # Generate points based on spatial prompt and 3D info
                points = self.generate_prompt_points(image, prompt, predictions, view_idx)
                
                if len(points) > 0:
                    input_points = np.array(points, dtype=np.float32)
                    input_labels = np.ones(len(input_points), dtype=np.int32)
                    
                    masks, scores, _ = sam2_predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=True
                    )
                    
                    # Return best mask
                    best_idx = np.argmax(scores)
                    return masks[best_idx]
            
            # Fallback: simple mask based on color/position heuristics
            return self.generate_simple_mask(image, prompt)
            
        except Exception as e:
            print(f"Error generating mask: {e}")
            return None
    
    def generate_prompt_points(self, image, prompt, predictions, view_idx):
        """Generate points based on spatial prompt."""
        h, w = image.shape[:2]
        points = []
        
        parsed = self.parse_prompt(prompt)
        
        # Generate points based on spatial relations
        if 'left' in parsed['spatial_relations']:
            points.extend([[w*0.25, h*0.5], [w*0.15, h*0.3], [w*0.15, h*0.7]])
        if 'right' in parsed['spatial_relations']:
            points.extend([[w*0.75, h*0.5], [w*0.85, h*0.3], [w*0.85, h*0.7]])
        if 'center' in parsed['spatial_relations']:
            points.extend([[w*0.5, h*0.5], [w*0.4, h*0.4], [w*0.6, h*0.6]])
        if 'above' in parsed['spatial_relations']:
            points.extend([[w*0.5, h*0.25], [w*0.3, h*0.2], [w*0.7, h*0.2]])
        if 'below' in parsed['spatial_relations']:
            points.extend([[w*0.5, h*0.75], [w*0.3, h*0.8], [w*0.7, h*0.8]])
        
        # If no spatial relations, use center points
        if not points:
            points = [[w*0.5, h*0.5], [w*0.3, h*0.3], [w*0.7, h*0.7]]
        
        return points
    
    def generate_simple_mask(self, image, prompt):
        """Generate simple mask based on color heuristics."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        
        parsed = self.parse_prompt(prompt)
        
        # Simple color-based segmentation
        if 'red' in parsed['colors']:
            # Detect red regions
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
            red_mask = red_mask1 | red_mask2
            mask |= red_mask.astype(bool)
        
        if 'blue' in parsed['colors']:
            # Detect blue regions
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            mask |= blue_mask.astype(bool)
        
        if 'green' in parsed['colors']:
            # Detect green regions
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
            mask |= green_mask.astype(bool)
        
        # Apply spatial filtering based on relations
        if 'left' in parsed['spatial_relations']:
            mask[:, w//2:] = False  # Keep only left half
        if 'right' in parsed['spatial_relations']:
            mask[:, :w//2] = False  # Keep only right half
        if 'above' in parsed['spatial_relations']:
            mask[h//2:, :] = False  # Keep only upper half
        if 'below' in parsed['spatial_relations']:
            mask[:h//2, :] = False  # Keep only lower half
        
        return mask
    
    def create_segmentation_overlays(self, segmentation_results, target_dir):
        """Create overlay visualizations for segmentation results."""
        overlays = []
        
        for view_idx, result in segmentation_results.items():
            image = cv2.imread(result['image_path'])
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            overlay = image_rgb.copy()
            mask = result['mask']
            
            # Apply colored mask
            colored_mask = np.zeros_like(overlay)
            colored_mask[mask] = [255, 0, 0]  # Red overlay
            overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
            
            # Save overlay
            overlay_path = os.path.join(target_dir, f"segmentation_overlay_{view_idx:03d}.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            overlays.append(overlay_path)
        
        return overlays


# Initialize spatial processor
spatial_processor = SpatialSegmentationProcessor()


def run_model(target_dir, model) -> dict:
    """Run the VGGT model on images in the 'target_dir/images' folder and return predictions."""
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Using CPU mode.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions


def handle_uploads(input_video, input_images):
    """Create a new 'target_dir' + 'images' subfolder, and place user-uploaded images or extracted frames from video into it."""
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


def update_gallery_on_upload(input_video, input_images):
    """Whenever user uploads or changes files, immediately handle them and show in the gallery."""
    if not input_video and not input_images:
        return None, None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing.", []


def enhanced_gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
    spatial_prompt="",
    enable_segmentation=False,
):
    """Enhanced demo with spatial segmentation capabilities."""
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None, []

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running enhanced model pipeline...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    # Spatial segmentation
    segmentation_overlays = []
    if enable_segmentation and spatial_prompt.strip():
        print("Running spatial segmentation...")
        segmentation_results = spatial_processor.segment_objects(predictions, spatial_prompt, target_dir)
        
        if segmentation_results:
            segmentation_overlays = spatial_processor.create_segmentation_overlays(segmentation_results, target_dir)
            print(f"Created {len(segmentation_overlays)} segmentation overlays")
        else:
            print("No segmentation results found")

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    
    seg_msg = f" + {len(segmentation_overlays)} segmentation overlays" if segmentation_overlays else ""
    log_msg = f"Reconstruction Success ({len(all_files)} frames){seg_msg}. Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True), segmentation_overlays


# -------------------------------------------------------------------------
# UI Helper Functions
# -------------------------------------------------------------------------
def clear_fields():
    """Clears the 3D viewer, the stored target_dir, and empties the gallery."""
    return None


def update_log():
    """Display a quick log message while waiting."""
    return "Loading and Reconstructing..."


# -------------------------------------------------------------------------
# Enhanced Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    title="Enhanced VGGT with Spatial Segmentation",
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .segmentation-section {
        background: linear-gradient(45deg, #f0f9ff, #e0f2fe);
        border-radius: 10px;
        padding: 1rem;
        border: 2px solid #0ea5e9;
        margin: 1rem 0;
    }
    
    .example-prompt {
        background: #f8fafc;
        border-left: 4px solid #10b981;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-style: italic;
    }
    """,
) as demo:
    
    # State variables
    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")
    
    # Header
    gr.HTML("""
    <h1>🏛️ Enhanced VGGT with Spatial Segmentation</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
    <a href="#">Project Page</a>
    </p>
    
    <div style="font-size: 16px; line-height: 1.5;">
    <p><strong>New:</strong> Now with spatial segmentation capabilities! Upload images, reconstruct in 3D, and segment objects using natural language descriptions.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # File upload section
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )
            
            # Spatial Segmentation Section
            with gr.Group(elem_classes=["segmentation-section"]):
                gr.Markdown("### 🎯 Spatial Segmentation")
                
                enable_segmentation = gr.Checkbox(
                    label="Enable Spatial Segmentation", 
                    value=False,
                    info="Use natural language to segment objects across views"
                )
                
                spatial_prompt = gr.Textbox(
                    label="Spatial Description",
                    placeholder="e.g., 'red chair on the left side', 'person in the background'",
                    lines=2,
                    info="Describe what you want to segment using colors, positions, and objects"
                )
                
                # Example prompts
                gr.HTML("""
                <div class="example-prompt">
                    <strong>Examples:</strong><br>
                    • "red chair on the left side"<br>
                    • "blue car in the center"<br>
                    • "person walking in background"<br>
                    • "green plant above the table"
                </div>
                """)

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction + Spatial Segmentation**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.", 
                    elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("🔍 Reconstruct + Segment", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )

            # Segmentation Results
            with gr.Group():
                gr.Markdown("### 🎨 Segmentation Results")
                segmentation_gallery = gr.Gallery(
                    label="Segmentation Overlays",
                    columns=3,
                    height=300,
                    show_download_button=True,
                    visible=False
                )

            # VGGT Parameters
            with gr.Accordion("⚙️ Advanced Parameters", open=False):
                with gr.Row():
                    prediction_mode = gr.Radio(
                        ["Depthmap and Camera Branch", "Pointmap Branch"],
                        label="Prediction Mode",
                        value="Depthmap and Camera Branch",
                    )

                with gr.Row():
                    conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                    frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                    
                with gr.Row():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # Event handlers
    def process_reconstruction_and_segmentation(
        target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, 
        show_cam, mask_sky, prediction_mode, spatial_prompt, enable_segmentation
    ):
        """Process both reconstruction and segmentation."""
        glbfile, log_msg, dropdown, overlays = enhanced_gradio_demo(
            target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg,
            show_cam, mask_sky, prediction_mode, spatial_prompt, enable_segmentation
        )
        
        # Show/hide segmentation gallery based on results
        gallery_visible = len(overlays) > 0 if overlays else False
        
        return glbfile, log_msg, dropdown, overlays, gr.Gallery(visible=gallery_visible)

    # Upload handling
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output, segmentation_gallery],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output, segmentation_gallery],
    )

    # Main reconstruction button
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=process_reconstruction_and_segmentation,
        inputs=[
            target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg,
            show_cam, mask_sky, prediction_mode, spatial_prompt, enable_segmentation
        ],
        outputs=[reconstruction_output, log_output, frame_filter, segmentation_gallery, segmentation_gallery],
    )

    # Instructions
    with gr.Accordion("📖 Instructions", open=False):
        gr.Markdown("""
        **Enhanced VGGT with Spatial Segmentation:**
        
        1. **Upload**: Choose a video or multiple images of the same scene
        2. **Enable Segmentation**: Check the "Enable Spatial Segmentation" box
        3. **Describe**: Enter what you want to segment (e.g., "red chair on the left")
        4. **Process**: Click "Reconstruct + Segment" to run both 3D reconstruction and segmentation
        5. **View Results**: See the 3D model and segmentation overlays
        
        **Spatial Description Tips:**
        - Use colors: "red", "blue", "green", etc.
        - Use positions: "left", "right", "center", "above", "below"
        - Use objects: "chair", "person", "car", "table", etc.
        - Combine them: "blue car on the right side"
        
        **Requirements:**
        - Multiple views of the same scene work best
        - Clear, well-lit images improve results
        - Spatial descriptions should be specific and descriptive
        """)

    demo.queue(max_size=20).launch(show_error=True, share=True)
"""
Basic VQA Test - Simple image input and question answering
Test the core VQA functionality without safety layers or complex pipeline.
"""

import sys
import os
sys.path.append('/home/isr-lab3/James/VQASynth-UAV')

import torch
import numpy as np
import cv2
from PIL import Image
import time

def create_simple_test_image():
    """Create a simple test image with clear spatial features."""
    # Create 224x224 RGB image
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    print("🎨 Creating synthetic test image:")
    
    # Add some spatial features
    # Blue sky background (upper half)
    img[:112, :, :] = [135, 206, 235]  # Sky blue
    print("   - Blue sky in upper half")
    
    # Green ground (lower half)
    img[112:, :, :] = [34, 139, 34]   # Forest green
    print("   - Green ground in lower half")
    
    # Add a red obstacle on the right side
    img[80:140, 150:200, :] = [255, 0, 0]  # Red obstacle
    print("   - Red obstacle on right side (pixels 150-200)")
    
    # Add a yellow clear path marker in the center-left
    img[80:140, 50:100, :] = [255, 255, 0]  # Yellow clear path
    print("   - Yellow path marker in center-left (pixels 50-100)")
    
    # Add some texture/noise for realism
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    print(f"   - Added realistic noise")
    print(f"   - Final image shape: {img.shape}")
    
    return img

def create_camera_image():
    """Capture image from camera with improved detection."""
    camera_backends = [cv2.CAP_V4L2, cv2.CAP_ANY, cv2.CAP_GSTREAMER]
    camera_indices = [0, 1, 2, 3]
    
    print("🔍 Searching for available cameras...")
    
    for backend in camera_backends:
        for cam_id in camera_indices:
            try:
                print(f"   Trying camera {cam_id} with backend {backend}...")
                cap = cv2.VideoCapture(cam_id, backend)
                
                if not cap.isOpened():
                    cap.release()
                    continue
                
                # Set properties for better compatibility
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Warm up camera
                for _ in range(3):
                    ret, frame = cap.read()
                
                # Try to capture a frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"   Camera {cam_id} opened but no frame captured")
                    cap.release()
                    continue
                
                print(f"✅ Successfully captured from camera {cam_id}!")
                print(f"📷 Frame shape: {frame.shape}")
                
                # Show preview window
                preview = cv2.resize(frame, (320, 240))
                cv2.putText(preview, f"Camera {cam_id} - Press any key", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('Camera Preview', preview)
                print("📷 Camera preview shown - press any key to continue with VQA...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # Process frame
                frame_resized = cv2.resize(frame, (224, 224))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                cap.release()
                print("✅ Captured camera image for VQA analysis")
                return frame_rgb
                
            except Exception as e:
                print(f"   Error with camera {cam_id}: {e}")
                if 'cap' in locals():
                    cap.release()
                continue
    
    print("❌ No camera found")
    
    # Check if we can list video devices and provide helpful info
    try:
        import subprocess
        result = subprocess.run(['ls', '/dev/video*'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"📹 Video devices found: {result.stdout.strip()}")
            print("💡 Suggestions:")
            print("   - Try: sudo chmod 666 /dev/video*")
            print("   - Check if camera is being used by another application")
            print("   - Try: lsusb | grep -i camera")
        else:
            print("📹 No video devices found in /dev/")
            print("💡 Please connect a USB camera or webcam")
    except:
        pass
    
    return None

def image_to_tensor(image_np):
    """Convert numpy image to tensor."""
    # Convert to float [0, 1]
    img_tensor = torch.from_numpy(image_np).float() / 255.0
    # Change HWC to CHW
    img_tensor = img_tensor.permute(2, 0, 1)
    return img_tensor

def create_mock_vqa_model():
    """Create a simple mock VQA model that actually processes the image."""
    
    class BasicMockVQA:
        def __init__(self):
            self.responses = 0
            
        def __call__(self, image, question):
            """Analyze image and provide spatial answer."""
            self.responses += 1
            
            print(f"\n🔍 VQA Analysis #{self.responses}")
            print(f"📝 Question: {question}")
            
            # Convert tensor to numpy for analysis
            if torch.is_tensor(image):
                img_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = image
                
            print(f"📊 Image shape: {img_np.shape}")
            print(f"📈 Image stats: min={img_np.min():.3f}, max={img_np.max():.3f}, mean={img_np.mean():.3f}")
            
            # Detailed visual analysis
            h, w, c = img_np.shape
            
            # Analyze different regions
            left_region = img_np[:, :w//3, :]
            center_region = img_np[:, w//3:2*w//3, :]
            right_region = img_np[:, 2*w//3:, :]
            top_region = img_np[:h//2, :, :]
            bottom_region = img_np[h//2:, :, :]
            
            left_intensity = np.mean(left_region)
            center_intensity = np.mean(center_region)
            right_intensity = np.mean(right_region)
            top_intensity = np.mean(top_region)
            bottom_intensity = np.mean(bottom_region)
            
            # Color analysis
            red_channel = np.mean(img_np[:, :, 0])
            green_channel = np.mean(img_np[:, :, 1])
            blue_channel = np.mean(img_np[:, :, 2])
            
            print(f"🔍 Spatial Analysis:")
            print(f"   Left region intensity: {left_intensity:.3f}")
            print(f"   Center region intensity: {center_intensity:.3f}")
            print(f"   Right region intensity: {right_intensity:.3f}")
            print(f"   Top region intensity: {top_intensity:.3f}")
            print(f"   Bottom region intensity: {bottom_intensity:.3f}")
            print(f"🎨 Color Analysis:")
            print(f"   Red channel: {red_channel:.3f}")
            print(f"   Green channel: {green_channel:.3f}")
            print(f"   Blue channel: {blue_channel:.3f}")
            
            # Detailed scene description
            scene_elements = []
            
            # Sky/background detection
            if top_intensity > bottom_intensity + 0.1:
                if blue_channel > red_channel and blue_channel > green_channel:
                    scene_elements.append("blue sky visible in upper region")
                else:
                    scene_elements.append("bright background in upper area")
            
            # Ground/surface detection  
            if bottom_intensity > 0.3:
                if green_channel > red_channel and green_channel > blue_channel:
                    scene_elements.append("green ground or vegetation in lower area")
                elif bottom_intensity < 0.5:
                    scene_elements.append("dark surface or floor in lower region")
                else:
                    scene_elements.append("light colored surface in lower area")
            
            # Obstacle detection with details
            obstacles = []
            if right_intensity > left_intensity + 0.05:
                if red_channel > 0.6:
                    obstacles.append("red object or obstacle on the right side")
                else:
                    obstacles.append("bright object on the right side")
                    
            if left_intensity > right_intensity + 0.05:
                obstacles.append("object or obstacle on the left side")
                
            if center_intensity > 0.7:
                obstacles.append("central obstacle ahead")
            
            # Path detection
            clear_paths = []
            if center_intensity > left_intensity and center_intensity > right_intensity:
                if center_intensity < 0.8:  # Not too bright (obstacle)
                    clear_paths.append("open path straight ahead")
            
            if left_intensity < center_intensity - 0.1 and left_intensity < right_intensity - 0.1:
                clear_paths.append("potential path on the left")
                
            if right_intensity < center_intensity - 0.1 and right_intensity < left_intensity - 0.1:
                clear_paths.append("potential path on the right")
            
            # Generate comprehensive description
            if "describe" in question.lower() or "see" in question.lower():
                description_parts = ["I can see:"]
                
                if scene_elements:
                    description_parts.extend([f"- {element}" for element in scene_elements])
                
                if obstacles:
                    description_parts.append("Obstacles detected:")
                    description_parts.extend([f"- {obs}" for obs in obstacles])
                else:
                    description_parts.append("- No major obstacles detected")
                    
                if clear_paths:
                    description_parts.append("Navigation options:")
                    description_parts.extend([f"- {path}" for path in clear_paths])
                
                # Add lighting conditions
                if np.mean(img_np) > 0.7:
                    description_parts.append("- Well-lit environment")
                elif np.mean(img_np) < 0.3:
                    description_parts.append("- Low-light conditions")
                else:
                    description_parts.append("- Moderate lighting")
                
                response = "\n".join(description_parts)
                confidence = 0.85
                
            elif "safe" in question.lower() or "navigation" in question.lower():
                if clear_paths:
                    response = f"Navigation assessment: {clear_paths[0]}. "
                    if obstacles:
                        response += f"Caution: {obstacles[0]}."
                    direction = "forward" if "straight ahead" in clear_paths[0] else ("left" if "left" in clear_paths[0] else "right")
                    confidence = 0.8
                else:
                    response = "Navigation challenging - no clear safe path detected."
                    direction = "stop"
                    confidence = 0.4
                    
            elif "obstacle" in question.lower():
                if obstacles:
                    response = f"Yes, obstacles detected: {', '.join(obstacles)}"
                    confidence = 0.9
                else:
                    response = "No significant obstacles detected in the immediate area"
                    confidence = 0.8
                    direction = "forward"
                    
            else:
                response = "I can analyze this environment. Try asking about navigation, obstacles, or what I can see."
                confidence = 0.6
                direction = "unknown"
            
            result = {
                'answer': response,
                'confidence': confidence,
                'spatial_analysis': {
                    'left_intensity': left_intensity,
                    'center_intensity': center_intensity,
                    'right_intensity': right_intensity,
                    'top_intensity': top_intensity,
                    'bottom_intensity': bottom_intensity,
                    'color_analysis': {
                        'red': red_channel,
                        'green': green_channel, 
                        'blue': blue_channel
                    },
                    'scene_elements': scene_elements,
                    'obstacles': obstacles,
                    'clear_paths': clear_paths,
                    'recommended_direction': direction if 'direction' in locals() else 'assess_further'
                },
                'processing_time_ms': np.random.uniform(15, 40)  # Simulate more complex processing
            }
            
            print(f"💬 Answer: {response}")
            print(f"🎯 Confidence: {confidence:.2f}")
            
            return result
    
    return BasicMockVQA()

def test_basic_vqa():
    """Run basic VQA tests with REAL camera only."""
    print("🎯 Basic VQA Test - REAL CAMERA ONLY")
    print("=" * 50)
    
    # Create VQA model
    vqa_model = create_mock_vqa_model()
    
    # Test with real camera only
    print(f"\n🧪 Real Camera VQA Test")
    print("-" * 40)
    
    # Get camera image
    image_np = create_camera_image()
    
    if image_np is None:
        print("❌ No camera available. Please connect a camera and try again.")
        return
        
    # Convert to tensor
    image_tensor = image_to_tensor(image_np)
    
    # Test questions focused on real environment
    questions = [
        "Describe what you see in this real environment",
        "What objects or obstacles can you detect?",
        "What is the safe navigation path ahead?",
        "Can you identify any specific colors or shapes?",
        "What would you recommend for UAV navigation?"
    ]
    
    # Test each question
    for i, question in enumerate(questions, 1):
        print(f"\n📋 Question {i}: {question}")
        print("-" * 30)
        
        start_time = time.time()
        result = vqa_model(image_tensor, question)
        end_time = time.time()
        
        print(f"⏱️  Processing time: {(end_time - start_time) * 1000:.1f}ms")
        print("-" * 30)
    
    print(f"\n✅ Real camera VQA test completed!")
    print(f"📈 Total responses: {vqa_model.responses}")

def interactive_test():
    """Interactive mode - ask custom questions."""
    print(f"\n🎮 Interactive VQA Mode")
    print("Type 'exit' to quit, 'camera' for new camera image, 'synthetic' for test image")
    
    vqa_model = create_mock_vqa_model()
    current_image = None
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() == 'exit':
                break
            elif command.lower() == 'camera':
                current_image = create_camera_image()
                if current_image is not None:
                    current_image = image_to_tensor(current_image)
                    print("📷 Camera image loaded")
                continue
            elif command.lower() == 'synthetic':
                current_image = image_to_tensor(create_simple_test_image())
                print("🎨 Synthetic test image loaded")
                continue
            
            if current_image is None:
                print("❌ No image loaded. Use 'camera' or 'synthetic' first.")
                continue
                
            # Treat as question
            result = vqa_model(current_image, command)
            print(f"\n✅ Result: {result}")
            
        except KeyboardInterrupt:
            break
    
    print("\n👋 Interactive test ended")

if __name__ == "__main__":
    print("🚀 VQA Basic Test Suite")
    print("Choose test mode:")
    print("1. Automated tests")
    print("2. Interactive mode")
    
    try:
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == "2":
            interactive_test()
        else:
            test_basic_vqa()
            
    except KeyboardInterrupt:
        print("\n👋 Test cancelled")

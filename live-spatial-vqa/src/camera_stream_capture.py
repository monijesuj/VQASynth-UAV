#!/usr/bin/env python3
"""
Real-time Camera Stream Capture Module
Supports RealSense RGB camera streams with threading and buffering
"""
import cv2
import numpy as np
from threading import Thread, Lock
from queue import Queue
import time
from typing import Optional, Tuple
from PIL import Image

class CameraStreamCapture:
    """Threaded camera capture for real-time streaming"""
    
    def __init__(self, camera_type='realsense', width=640, height=480, fps=30):
        """
        Initialize camera stream capture
        
        Args:
            camera_type: 'realsense' or 'webcam'
            width: frame width
            height: frame height
            fps: frames per second
        """
        self.camera_type = camera_type
        self.width = width
        self.height = height
        self.fps = fps
        
        self.pipeline = None
        self.cap = None
        self.frame = None
        self.frame_lock = Lock()
        self.running = False
        self.thread = None
        
        # Frame buffer for batching
        self.frame_queue = Queue(maxsize=10)
        
        # Statistics
        self.frame_count = 0
        self.start_time = None
        
    def start(self) -> bool:
        """Start camera capture thread"""
        if self.running:
            print("⚠️  Camera already running")
            return True
        
        # Initialize camera
        if self.camera_type == 'realsense':
            success = self._init_realsense()
        else:
            success = self._init_webcam()
        
        if not success:
            return False
        
        self.running = True
        self.start_time = time.time()
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        # Wait for first frame to be captured
        max_wait = 2.0  # seconds
        wait_start = time.time()
        while self.frame is None and (time.time() - wait_start) < max_wait:
            time.sleep(0.1)
        
        if self.frame is None:
            print("⚠️  Warning: No frames captured yet")
        
        print(f"✅ Camera stream started: {self.width}x{self.height}@{self.fps}fps")
        return True
    
    def _init_realsense(self) -> bool:
        """Initialize Intel RealSense camera"""
        try:
            import pyrealsense2 as rs
            
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure RGB stream only
            config.enable_stream(rs.stream.color, self.width, self.height, 
                               rs.format.bgr8, self.fps)
            
            # Start streaming
            self.pipeline.start(config)
            
            # Warm up camera
            for _ in range(30):
                self.pipeline.wait_for_frames()
            
            print("✅ RealSense camera initialized")
            return True
            
        except ImportError:
            print("❌ pyrealsense2 not installed. Install with: pip install pyrealsense2")
            return False
        except Exception as e:
            print(f"❌ Failed to initialize RealSense: {e}")
            return False
    
    def _init_webcam(self) -> bool:
        """Initialize standard webcam"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.cap.isOpened():
                print("❌ Could not open webcam")
                return False
            
            # Test frame
            ret, _ = self.cap.read()
            if not ret:
                print("❌ Could not read from webcam")
                return False
            
            print("✅ Webcam initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize webcam: {e}")
            return False
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.running:
            frame = self._get_frame()
            
            if frame is not None:
                with self.frame_lock:
                    self.frame = frame
                    self.frame_count += 1
                
                # Add to queue if not full
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
            
            time.sleep(0.001)  # Small delay to prevent CPU hogging
    
    def _get_frame(self) -> Optional[np.ndarray]:
        """Get frame from camera"""
        try:
            if self.camera_type == 'realsense' and self.pipeline:
                import pyrealsense2 as rs
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    return None
                
                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())
                return frame
                
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                return frame if ret else None
                
        except Exception as e:
            print(f"❌ Error capturing frame: {e}")
            return None
    
    def read(self) -> Optional[np.ndarray]:
        """
        Read current frame (non-blocking)
        
        Returns:
            Current frame as numpy array (BGR format) or None
        """
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def read_pil(self) -> Optional[Image.Image]:
        """
        Read current frame as PIL Image (RGB format)
        
        Returns:
            Current frame as PIL Image or None
        """
        # Get frame with thread-safe copy
        with self.frame_lock:
            if self.frame is not None and isinstance(self.frame, np.ndarray):
                try:
                    # Make a proper copy to avoid threading issues
                    frame_copy = self.frame.copy()
                    
                    # Ensure frame is contiguous and proper dtype
                    if not frame_copy.flags.c_contiguous:
                        frame_copy = np.ascontiguousarray(frame_copy)
                    
                    if frame_copy.dtype != np.uint8:
                        frame_copy = frame_copy.astype(np.uint8)
                    
                    # Validate frame has proper dimensions
                    if len(frame_copy.shape) == 3 and frame_copy.shape[2] == 3:
                        # Convert BGR to RGB using numpy
                        frame_rgb = frame_copy[:, :, ::-1]  # Reverse color channels
                        return Image.fromarray(frame_rgb)
                except Exception as e:
                    # Frame might be invalid, return None
                    print(f"⚠️  read_pil error: {e}")
                    return None
        return None
    
    def get_fps(self) -> float:
        """Get actual FPS"""
        if self.start_time is None or self.frame_count == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Cleanup
        if self.camera_type == 'realsense' and self.pipeline:
            self.pipeline.stop()
        elif self.cap:
            self.cap.release()
        
        print(f"📊 Camera stopped. Captured {self.frame_count} frames at {self.get_fps():.1f} FPS")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


def test_camera():
    """Test camera stream capture"""
    print("🎥 Testing Camera Stream Capture")
    print("=" * 50)
    
    # Try RealSense first, fallback to webcam
    camera = CameraStreamCapture(camera_type='realsense', width=640, height=480, fps=30)
    
    if not camera.start():
        print("⚠️  RealSense not available, trying webcam...")
        camera = CameraStreamCapture(camera_type='webcam', width=640, height=480, fps=30)
        if not camera.start():
            print("❌ No camera available!")
            return
    
    print("\n📸 Press 'q' to quit, 's' to save frame")
    
    try:
        while True:
            frame = camera.read()
            
            if frame is not None:
                # Add FPS overlay
                fps = camera.get_fps()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Camera Stream', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and frame is not None:
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()

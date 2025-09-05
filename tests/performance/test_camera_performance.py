"""
Pure Camera Stream Performance Test
Tests VQA pipeline performance with camera-only input to establish clean baselines.
"""

import sys
import os
sys.path.append('/home/isr-lab3/James/VQASynth-UAV')

import torch
import numpy as np
import time
import threading
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from vqasynth.realtime_pipeline import RealtimePipeline
from vqasynth.safety_spatial_vqa import SafetyCriticalSpatialVQA
from vqasynth.realtime_optimizer import RealtimeOptimizer

class CameraStreamSimulator:
    """Simulate realistic camera stream for UAV navigation."""
    
    def __init__(self, fps: int = 30, resolution: tuple = (224, 224)):
        self.fps = fps
        self.resolution = resolution
        self.frame_interval = 1.0 / fps
        self.is_streaming = False
        self.frame_count = 0
        
    def generate_realistic_frame(self) -> torch.Tensor:
        """Generate realistic camera frame with spatial features."""
        
        # Simulate different scenarios
        scenarios = ['clear_path', 'obstacle_left', 'obstacle_right', 'narrow_passage', 'emergency']
        scenario = np.random.choice(scenarios, p=[0.4, 0.2, 0.2, 0.15, 0.05])
        
        # Base image with some structure
        image = torch.randn(3, self.resolution[0], self.resolution[1]) * 0.3 + 0.5
        
        # Add scenario-specific features
        if scenario == 'obstacle_left':
            # Add high-intensity region on left (obstacle)
            image[:, :, :self.resolution[1]//3] += 0.4
            
        elif scenario == 'obstacle_right':
            # Add high-intensity region on right (obstacle)
            image[:, :, 2*self.resolution[1]//3:] += 0.4
            
        elif scenario == 'narrow_passage':
            # Add obstacles on both sides
            image[:, :, :self.resolution[1]//4] += 0.3
            image[:, :, 3*self.resolution[1]//4:] += 0.3
            
        elif scenario == 'emergency':
            # Add large central obstacle
            h, w = self.resolution
            image[:, h//3:2*h//3, w//3:2*w//3] += 0.6
        
        # Normalize to [0, 1]
        image = torch.clamp(image, 0, 1)
        self.frame_count += 1
        
        return image, scenario
    
    def start_stream(self, callback_func, duration_seconds: float = 10):
        """Start camera stream simulation."""
        self.is_streaming = True
        start_time = time.time()
        
        print(f"Starting camera stream simulation ({self.fps} FPS, {duration_seconds}s)")
        
        while self.is_streaming and (time.time() - start_time) < duration_seconds:
            frame_start = time.time()
            
            # Generate frame
            image, scenario = self.generate_realistic_frame()
            
            # Call processing callback
            callback_func(image, scenario, self.frame_count)
            
            # Maintain FPS timing
            elapsed = time.time() - frame_start
            sleep_time = max(0, self.frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.is_streaming = False
        print(f"Camera stream simulation completed. Generated {self.frame_count} frames")

class RealCameraCapture:
    """Capture real camera stream for UAV navigation testing."""
    
    def __init__(self, camera_id: int = 0, fps: int = 30, resolution: tuple = (224, 224)):
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
        self.frame_interval = 1.0 / fps
        self.is_streaming = False
        self.frame_count = 0
        self.cap = None
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture with better error handling."""
        # Try multiple camera backends and indices
        camera_backends = [cv2.CAP_V4L2, cv2.CAP_ANY, cv2.CAP_GSTREAMER]
        camera_indices = [self.camera_id, 0, 1, 2]  # Try fallback indices
        
        for backend in camera_backends:
            for cam_id in camera_indices:
                try:
                    print(f"🔍 Trying camera {cam_id} with backend {backend}...")
                    self.cap = cv2.VideoCapture(cam_id, backend)
                    
                    if not self.cap.isOpened():
                        if self.cap:
                            self.cap.release()
                        continue
                    
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                    
                    # Test frame capture
                    ret, test_frame = self.cap.read()
                    if not ret or test_frame is None:
                        print(f"❌ Camera {cam_id} opened but no frames available")
                        self.cap.release()
                        continue
                        
                    print(f"✅ Camera {cam_id} initialized successfully!")
                    print(f"📷 Camera resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
                    print(f"🎯 Target resolution: {self.resolution[1]}x{self.resolution[0]}")
                    print(f"🔧 Backend: {backend}")
                    
                    # Update camera_id to the working one
                    self.camera_id = cam_id
                    return True
                    
                except Exception as e:
                    print(f"❌ Error with camera {cam_id}, backend {backend}: {e}")
                    if self.cap:
                        self.cap.release()
                    continue
            
        print(f"❌ Could not initialize any camera")
        print(f"💡 Suggestions:")
        print(f"   - Check camera permissions: sudo chmod 666 /dev/video*")
        print(f"   - Ensure camera not in use by another application")
        print(f"   - Try: lsusb | grep -i camera")
        return False
    
    def process_frame_for_vqa(self, frame) -> torch.Tensor:
        """Convert camera frame to VQA-ready tensor."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to target resolution
        frame_resized = cv2.resize(frame_rgb, self.resolution)
        
        # Convert to tensor and normalize to [0, 1]
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        
        # Change from HWC to CHW format
        frame_tensor = frame_tensor.permute(2, 0, 1)
        
        return frame_tensor
    
    def analyze_frame_complexity(self, frame_tensor: torch.Tensor) -> str:
        """Analyze frame content to determine scenario type."""
        frame_np = frame_tensor.cpu().numpy()
        
        # Calculate spatial complexity metrics
        gray_equivalent = np.mean(frame_np, axis=0)  # Average across channels
        
        # Edge detection for obstacle analysis
        edges = np.abs(np.gradient(gray_equivalent, axis=0)) + np.abs(np.gradient(gray_equivalent, axis=1))
        edge_density = np.mean(edges)
        
        # Analyze spatial distribution
        h, w = gray_equivalent.shape
        left_intensity = np.mean(gray_equivalent[:, :w//3])
        center_intensity = np.mean(gray_equivalent[:, w//3:2*w//3])
        right_intensity = np.mean(gray_equivalent[:, 2*w//3:])
        
        # Determine scenario based on analysis
        if edge_density > 0.15:  # High edge density
            if center_intensity > max(left_intensity, right_intensity) + 0.1:
                return 'central_obstacle'
            elif left_intensity > center_intensity + 0.1:
                return 'left_obstacle'
            elif right_intensity > center_intensity + 0.1:
                return 'right_obstacle'
            else:
                return 'complex_scene'
        else:
            return 'clear_path'
    
    def start_stream(self, callback_func, duration_seconds: float = 10):
        """Start real camera stream capture."""
        if not self.initialize_camera():
            return False
            
        self.is_streaming = True
        start_time = time.time()
        
        print(f"🎥 Starting REAL camera stream ({self.fps} FPS, {duration_seconds}s)")
        print(f"📹 Camera ID: {self.camera_id}")
        print("Press 'q' in camera window to stop early")
        
        try:
            while self.is_streaming and (time.time() - start_time) < duration_seconds:
                frame_start = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Failed to capture frame, stopping stream")
                    break
                
                # Convert frame for VQA processing
                frame_tensor = self.process_frame_for_vqa(frame)
                scenario = self.analyze_frame_complexity(frame_tensor)
                
                # Display frame (optional, for debugging)
                display_frame = cv2.resize(frame, (320, 240))
                cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Scenario: {scenario}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.imshow('UAV Camera Feed', display_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("👤 User requested stop")
                    break
                
                # Call processing callback
                callback_func(frame_tensor, scenario, self.frame_count)
                self.frame_count += 1
                
                # Maintain FPS timing
                elapsed = time.time() - frame_start
                sleep_time = max(0, self.frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.is_streaming = False
            
        print(f"🎬 Real camera stream completed. Captured {self.frame_count} frames")
        return True

class CameraPerformanceTester:
    """Test camera-only VQA pipeline performance."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        
        # Mock VQA model optimized for testing
        self.mock_model = self._create_optimized_mock_model()
        
        # Initialize pipeline components
        self.pipeline = RealtimePipeline(config_path, self.mock_model)
        
        # Performance tracking
        self.frame_times = []
        self.processing_times = []
        self.confidence_scores = []
        self.safety_decisions = []
        self.dropped_frames = 0
        self.processed_results = []
        
        # Test parameters
        self.test_scenarios = []
        
    def _create_optimized_mock_model(self):
        """Create optimized mock model for performance testing."""
        
        class OptimizedMockVQA:
            def __init__(self):
                self.eval_mode = True
                self.processing_times = []
            
            def train(self):
                self.eval_mode = False
            
            def eval(self):
                self.eval_mode = True
            
            def __call__(self, image, question):
                # Simulate realistic VQA processing time
                start_time = time.time()
                
                # Extract simple spatial features from image
                image_np = image.cpu().numpy() if torch.is_tensor(image) else image
                
                # Simulate processing based on image complexity
                complexity = np.std(image_np)  # Image complexity metric
                base_time = 0.008  # 8ms base processing
                variable_time = complexity * 0.015  # Variable based on complexity
                
                time.sleep(base_time + variable_time)
                
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Generate realistic spatial outputs
                center_intensity = np.mean(image_np[:, image_np.shape[1]//3:2*image_np.shape[1]//3, 
                                                    image_np.shape[2]//3:2*image_np.shape[2]//3])
                
                if center_intensity > 0.7:  # High intensity = obstacle
                    forward_distance = np.random.uniform(1.0, 3.0)
                    safe_direction = 'left' if np.random.random() > 0.5 else 'right'
                else:
                    forward_distance = np.random.uniform(5.0, 12.0)
                    safe_direction = 'forward'
                
                return {
                    'answer': f"Navigate {safe_direction}, obstacle distance {forward_distance:.1f}m",
                    'forward_distance': forward_distance,
                    'lateral_distance': np.random.uniform(-1.0, 1.0),
                    'vertical_distance': np.random.uniform(-0.5, 0.5),
                    'obstacle_detected': center_intensity > 0.6,
                    'safe_direction': safe_direction,
                    'processing_time': processing_time
                }
        
        return OptimizedMockVQA()
    
    def camera_frame_callback(self, image: torch.Tensor, scenario: str, frame_id: int):
        """Process camera frame through VQA pipeline."""
        frame_start_time = time.time()
        
        # Submit frame to pipeline
        pipeline_frame_id = self.pipeline.process_frame(
            image=image,
            question="What is the safe navigation path ahead?",
            sensor_data=None,  # Camera only
            priority=1 if scenario == 'emergency' else 0
        )
        
        if pipeline_frame_id >= 0:
            self.test_scenarios.append({
                'frame_id': frame_id,
                'pipeline_frame_id': pipeline_frame_id,
                'scenario': scenario,
                'timestamp': frame_start_time
            })
        else:
            self.dropped_frames += 1
    
    def collect_results(self, duration_seconds: float):
        """Collect processing results during test."""
        start_time = time.time()
        results_collected = 0
        
        while (time.time() - start_time) < duration_seconds + 2.0:  # Extra time for processing
            result = self.pipeline.get_result(timeout=0.1)
            
            if result:
                results_collected += 1
                
                # Extract performance metrics
                processing_time = result.get('processing_time_ms', 0) / 1000.0
                confidence = result.get('spatial_decision', {}).get('confidence', 0)
                safety_status = result.get('safety_status', 'UNKNOWN')
                
                self.processing_times.append(processing_time)
                self.confidence_scores.append(confidence)
                self.safety_decisions.append(safety_status)
                self.processed_results.append(result)
                
                # Print progress
                if results_collected % 10 == 0:
                    print(f"  Processed {results_collected} frames, "
                          f"avg time: {np.mean(self.processing_times[-10:]) * 1000:.1f}ms")
        
        print(f"Collected {results_collected} results")
    
    def run_performance_test(self, fps: int = 30, duration: float = 10.0, use_real_camera: bool = False, camera_id: int = 0):
        """Run comprehensive camera performance test."""
        
        print(f"🚀 Running Camera Performance Test")
        print(f"📊 Parameters: {fps} FPS, {duration}s duration")
        if use_real_camera:
            print(f"📹 Using REAL camera (ID: {camera_id})")
        else:
            print(f"🎭 Using SYNTHETIC camera simulation")
        print("=" * 60)
        
        # Reset metrics
        self.frame_times = []
        self.processing_times = []
        self.confidence_scores = []
        self.safety_decisions = []
        self.dropped_frames = 0
        self.processed_results = []
        self.test_scenarios = []
        
        print(f"🎥 CAMERA-ONLY PERFORMANCE TEST")
        print(f"📊 Parameters: {fps} FPS, {duration}s duration")
        print("=" * 50)
        
        # Start pipeline
        self.pipeline.start_pipeline()
        
        # Start result collection in background
        result_thread = threading.Thread(
            target=self.collect_results, 
            args=(duration,), 
            daemon=True
        )
        result_thread.start()
        
        # Choose camera source
        if use_real_camera:
            camera_source = RealCameraCapture(camera_id=camera_id, fps=fps)
        else:
            camera_source = CameraStreamSimulator(fps=fps)
        
        try:
            success = camera_source.start_stream(
                callback_func=self.camera_frame_callback,
                duration_seconds=duration
            )
            
            if not success and use_real_camera:
                print("❌ Real camera failed, falling back to simulation...")
                camera_source = CameraStreamSimulator(fps=fps)
                camera_source.start_stream(
                    callback_func=self.camera_frame_callback,
                    duration_seconds=duration
                )
            
            # Wait for processing to complete
            print("\n⏳ Waiting for processing to complete...")
            time.sleep(3.0)
            
        finally:
            # Stop pipeline
            self.pipeline.stop_pipeline()
            
            # Wait for result collection
            if result_thread.is_alive():
                result_thread.join(timeout=2.0)
        
        return self.analyze_performance()
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze camera performance test results."""
        
        print(f"\n📊 CAMERA PERFORMANCE ANALYSIS")
        print("=" * 40)
        
        if not self.processing_times:
            print("❌ No processing times recorded")
            return {}
        
        # Processing time analysis
        processing_times_ms = [t * 1000 for t in self.processing_times]
        
        analysis = {
            'frames_submitted': len(self.test_scenarios),
            'frames_processed': len(self.processing_times),
            'frames_dropped': self.dropped_frames,
            'drop_rate': self.dropped_frames / (len(self.test_scenarios) + self.dropped_frames) if (len(self.test_scenarios) + self.dropped_frames) > 0 else 0,
            
            # Timing metrics
            'avg_processing_time_ms': np.mean(processing_times_ms),
            'median_processing_time_ms': np.median(processing_times_ms),
            'min_processing_time_ms': np.min(processing_times_ms),
            'max_processing_time_ms': np.max(processing_times_ms),
            'std_processing_time_ms': np.std(processing_times_ms),
            'p95_processing_time_ms': np.percentile(processing_times_ms, 95),
            'p99_processing_time_ms': np.percentile(processing_times_ms, 99),
            
            # Real-time metrics
            'real_time_capable': np.percentile(processing_times_ms, 95) <= 100,
            'target_fps_achievable': np.mean(processing_times_ms) <= (1000/30),  # 30 FPS = 33.3ms max
            
            # Safety metrics
            'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
            'safety_decisions': {
                'SAFE': self.safety_decisions.count('SAFE'),
                'EMERGENCY': self.safety_decisions.count('EMERGENCY'),
                'UNKNOWN': self.safety_decisions.count('UNKNOWN')
            }
        }
        
        # Print detailed analysis
        print(f"📈 Processing Statistics:")
        print(f"  Frames submitted: {analysis['frames_submitted']}")
        print(f"  Frames processed: {analysis['frames_processed']}")
        print(f"  Frames dropped: {analysis['frames_dropped']} ({analysis['drop_rate']:.1%})")
        
        print(f"\n⏱️ Timing Analysis:")
        print(f"  Average: {analysis['avg_processing_time_ms']:.2f} ms")
        print(f"  Median: {analysis['median_processing_time_ms']:.2f} ms")
        print(f"  Min/Max: {analysis['min_processing_time_ms']:.2f} / {analysis['max_processing_time_ms']:.2f} ms")
        print(f"  95th percentile: {analysis['p95_processing_time_ms']:.2f} ms")
        print(f"  Standard deviation: {analysis['std_processing_time_ms']:.2f} ms")
        
        print(f"\n🎯 Real-Time Assessment:")
        print(f"  Real-time capable (<100ms p95): {'✅ YES' if analysis['real_time_capable'] else '❌ NO'}")
        print(f"  30 FPS achievable (<33ms avg): {'✅ YES' if analysis['target_fps_achievable'] else '❌ NO'}")
        
        print(f"\n🛡️ Safety Performance:")
        print(f"  Average confidence: {analysis['avg_confidence']:.3f}")
        print(f"  Safety decisions: {analysis['safety_decisions']}")
        
        # Performance recommendations
        self._generate_performance_recommendations(analysis)
        
        return analysis
    
    def _generate_performance_recommendations(self, analysis: Dict[str, Any]):
        """Generate performance optimization recommendations."""
        
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        
        recommendations = []
        
        if analysis['avg_processing_time_ms'] > 100:
            recommendations.append("🔧 CRITICAL: Reduce processing time by 50%+")
            recommendations.append("   - Reduce Monte Carlo samples from 5 to 3")
            recommendations.append("   - Implement model quantization for real VQA models")
            recommendations.append("   - Add parallel processing for VQA stages")
        
        elif analysis['avg_processing_time_ms'] > 50:
            recommendations.append("⚡ MEDIUM: Further speed optimization needed")
            recommendations.append("   - Fine-tune uncertainty estimation parameters")
            recommendations.append("   - Implement result caching for similar frames")
        
        if analysis['drop_rate'] > 0.05:
            recommendations.append("📉 Address frame dropping issues")
            recommendations.append("   - Increase queue sizes")
            recommendations.append("   - Implement frame prioritization")
        
        if analysis['avg_confidence'] < 0.3:
            recommendations.append("🎯 Calibrate confidence estimation")
            recommendations.append("   - Adjust uncertainty thresholds")
            recommendations.append("   - Implement confidence boosting for clear scenarios")
        
        if not recommendations:
            recommendations.append("🎉 Performance looks good! Ready for hardware testing")
        
        for rec in recommendations:
            print(f"  {rec}")
    
    def save_performance_results(self, analysis: Dict[str, Any], filename: str = "camera_performance_results.json"):
        """Save performance results to file."""
        import json
        
        # Add test configuration
        analysis['test_config'] = {
            'test_type': 'camera_only',
            'mock_model_type': 'optimized',
            'timestamp': time.time(),
            'monte_carlo_samples': 5
        }
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")

def run_camera_performance_test():
    """Main function to run camera performance tests."""
    print("🎯 VQA Camera Performance Testing Suite")
    print("=" * 50)
    
    config_path = '/home/isr-lab3/James/VQASynth-UAV/config/config_camera_test.yaml'
    
    # Get user preference for camera type
    print("\n📹 Camera Source Selection:")
    print("1. Real Camera (connected to computer)")
    print("2. Synthetic Simulation (for testing)")
    
    try:
        choice = input("\nSelect camera source (1/2) [default: 2]: ").strip()
        if choice == "1":
            use_real_camera = True
            try:
                camera_id = int(input("Enter camera ID (0 for default webcam): ").strip() or "0")
            except ValueError:
                camera_id = 0
                print("Invalid input, using camera ID 0")
        else:
            use_real_camera = False
            camera_id = 0
    except KeyboardInterrupt:
        print("\nUsing default synthetic simulation")
        use_real_camera = False
        camera_id = 0
    
    print(f"\n🎬 Camera Configuration:")
    print(f"   Type: {'Real Camera' if use_real_camera else 'Synthetic Simulation'}")
    if use_real_camera:
        print(f"   Camera ID: {camera_id}")
    
    # Test scenarios
    test_scenarios = [
        {'name': 'Low Rate Test (10 FPS)', 'fps': 10, 'duration': 10.0},
        {'name': 'Medium Rate Test (30 FPS)', 'fps': 30, 'duration': 5.0},
        {'name': 'High Rate Test (60 FPS)', 'fps': 60, 'duration': 3.0}
    ]
    
    all_results = {}
    
    for scenario in test_scenarios:
        print(f"\n🚀 Running {scenario['name']}")
        print("=" * 60)
        
        # Create new tester for each scenario
        tester = CameraPerformanceTester(config_path)
        
        try:
            results = tester.run_performance_test(
                fps=scenario['fps'],
                duration=scenario['duration'],
                use_real_camera=use_real_camera,
                camera_id=camera_id
            )
            
            all_results[scenario['name']] = results
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            
            # If real camera fails, offer fallback
            if use_real_camera:
                print(f"\n🔄 Real camera failed. Try synthetic simulation? (y/n): ", end="")
                try:
                    fallback = input().strip().lower()
                    if fallback in ['y', 'yes']:
                        print("Falling back to synthetic simulation...")
                        results = tester.run_performance_test(
                            fps=scenario['fps'],
                            duration=scenario['duration'],
                            use_real_camera=False,
                            camera_id=0
                        )
                        all_results[scenario['name']] = results
                except KeyboardInterrupt:
                    print("\nSkipping this test scenario")
    
    # Summary comparison
    if len(all_results) > 1:
        print(f"\n📊 COMPARATIVE ANALYSIS")
        print("=" * 30)
        
        for test_name, results in all_results.items():
            if results:
                print(f"{test_name}:")
                print(f"  Average time: {results['avg_processing_time_ms']:.1f}ms")
                print(f"  Real-time capable: {'✅' if results['real_time_capable'] else '❌'}")
                print(f"  Drop rate: {results['drop_rate']:.1%}")
    
    return all_results

if __name__ == "__main__":
    # Quick test mode for real camera
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--real-camera":
        print("🎯 Quick Real Camera Test Mode")
        config_path = '/home/isr-lab3/James/VQASynth-UAV/config/config_realtime_uav.yaml'
        tester = CameraPerformanceTester(config_path)
        
        camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        
        print(f"Testing real camera ID {camera_id} at 30 FPS for 10 seconds...")
        results = tester.run_performance_test(
            fps=30, 
            duration=10.0, 
            use_real_camera=True, 
            camera_id=camera_id
        )
    else:
        run_camera_performance_test()

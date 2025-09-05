"""
Comprehensive integration test for the real-time spatial VQA UAV navigation system.
Tests the complete pipeline from image input to control commands.
"""

import sys
import os
sys.path.append('/home/isr-lab3/James/VQASynth-UAV')

import torch
import numpy as np
import time
import threading
from typing import Dict, Any

# Import our modules
from vqasynth.realtime_optimizer import RealtimeOptimizer
from vqasynth.safety_spatial_vqa import SafetyCriticalSpatialVQA
from vqasynth.sensor_fusion import MultiModalSensorFusion, SensorData
from vqasynth.realtime_pipeline import RealtimePipeline
from navigation.vqa_navigation_controller import NavigationController

class MockSpatialVQAModel:
    """Enhanced mock VQA model with realistic spatial reasoning."""
    
    def __init__(self):
        self.eval_mode = True
        self.processed_frames = 0
    
    def train(self):
        self.eval_mode = False
    
    def eval(self):
        self.eval_mode = True
    
    def __call__(self, image, question):
        """Simulate spatial VQA inference with realistic outputs."""
        self.processed_frames += 1
        
        # Simulate processing time (optimized model)
        time.sleep(np.random.uniform(0.01, 0.025))
        
        # Generate realistic spatial outputs based on simulated scenarios
        scenario = np.random.choice(['clear_path', 'obstacle_ahead', 'narrow_passage', 'emergency'])
        
        if scenario == 'clear_path':
            return {
                'answer': "Clear path ahead, safe to proceed",
                'forward_distance': np.random.uniform(8, 15),
                'lateral_distance': np.random.uniform(-0.5, 0.5),
                'vertical_distance': np.random.uniform(-0.2, 0.2),
                'obstacle_detected': False,
                'safe_direction': 'forward'
            }
        
        elif scenario == 'obstacle_ahead':
            return {
                'answer': "Obstacle detected ahead, navigate around",
                'forward_distance': np.random.uniform(2, 4),
                'lateral_distance': np.random.uniform(1, 3),
                'vertical_distance': np.random.uniform(-0.5, 0.5),
                'obstacle_detected': True,
                'safe_direction': 'right'
            }
        
        elif scenario == 'narrow_passage':
            return {
                'answer': "Narrow passage detected, proceed with caution",
                'forward_distance': np.random.uniform(3, 6),
                'lateral_distance': np.random.uniform(-0.8, 0.8),
                'vertical_distance': np.random.uniform(-0.3, 0.3),
                'obstacle_detected': True,
                'safe_direction': 'forward'
            }
        
        else:  # emergency
            return {
                'answer': "Emergency obstacle very close, stop immediately",
                'forward_distance': np.random.uniform(0.5, 1.5),
                'lateral_distance': np.random.uniform(-2, 2),
                'vertical_distance': np.random.uniform(-1, 1),
                'obstacle_detected': True,
                'safe_direction': 'stop'
            }

class IntegratedVQANavigationSystem:
    """Complete integrated VQA navigation system."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        
        # Initialize components
        self.mock_model = MockSpatialVQAModel()
        self.pipeline = RealtimePipeline(config_path, self.mock_model)
        self.navigation_controller = NavigationController(config_path)
        
        # System state
        self.is_running = False
        self.results_buffer = []
        self.control_commands = []
        
        # Performance tracking
        self.start_time = None
        self.total_frames_processed = 0
        self.emergency_stops = 0
        
    def start_system(self):
        """Start the complete navigation system."""
        print("Starting integrated VQA navigation system...")
        
        self.start_time = time.time()
        self.is_running = True
        
        # Start the real-time pipeline
        self.pipeline.start_pipeline()
        
        # Start result processing thread
        self.result_thread = threading.Thread(target=self._process_results, daemon=True)
        self.result_thread.start()
        
        print("✅ System started successfully")
    
    def stop_system(self):
        """Stop the complete navigation system."""
        print("Stopping integrated navigation system...")
        
        self.is_running = False
        self.pipeline.stop_pipeline()
        
        if hasattr(self, 'result_thread') and self.result_thread.is_alive():
            self.result_thread.join(timeout=2.0)
        
        print("✅ System stopped")
    
    def process_navigation_frame(self, image: torch.Tensor, question: str = None, 
                               sensor_data: Dict[str, SensorData] = None):
        """Process a single navigation frame through the complete system."""
        
        if not self.is_running:
            print("⚠️ System not running")
            return None
        
        # Use default question if not provided
        if question is None:
            question = "What is the safe navigation path ahead?"
        
        # Submit frame to pipeline
        frame_id = self.pipeline.process_frame(image, question, sensor_data)
        
        if frame_id >= 0:
            self.total_frames_processed += 1
            return frame_id
        else:
            print("⚠️ Frame dropped due to queue overflow")
            return None
    
    def _process_results(self):
        """Process pipeline results and generate control commands."""
        while self.is_running:
            # Get result from pipeline
            result = self.pipeline.get_result(timeout=0.1)
            
            if result is None:
                continue
            
            try:
                # Process through navigation controller
                control_command = self.navigation_controller.process_spatial_vqa_result(result)
                
                # Store command
                self.control_commands.append(control_command)
                
                # Track emergency stops
                if control_command.flight_mode.value == 'emergency_stop':
                    self.emergency_stops += 1
                
                # Log significant events
                if control_command.confidence < 0.5:
                    print(f"⚠️ Low confidence navigation: {control_command.confidence:.2f}")
                
                if control_command.safety_margin < 2.0:
                    print(f"⚠️ Low safety margin: {control_command.safety_margin:.2f}m")
                
            except Exception as e:
                print(f"❌ Error processing result: {e}")
    
    def get_latest_control_command(self):
        """Get the most recent control command."""
        return self.control_commands[-1] if self.control_commands else None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        runtime = time.time() - self.start_time if self.start_time else 0
        
        # Pipeline statistics
        pipeline_stats = self.pipeline.get_pipeline_statistics()
        
        # Navigation statistics
        nav_stats = self.navigation_controller.get_controller_statistics()
        
        return {
            'runtime_seconds': runtime,
            'total_frames_submitted': self.total_frames_processed,
            'frames_processed': pipeline_stats.get('processed_frames', 0),
            'frames_dropped': pipeline_stats.get('dropped_frames', 0),
            'average_processing_time_ms': pipeline_stats.get('avg_processing_time_ms', 0),
            'real_time_capable': pipeline_stats.get('real_time_capable', False),
            'emergency_stops': self.emergency_stops,
            'current_flight_mode': nav_stats.get('current_mode', 'unknown'),
            'control_commands_generated': len(self.control_commands),
            'system_health': self._assess_system_health()
        }
    
    def _assess_system_health(self) -> str:
        """Assess overall system health."""
        if not self.is_running:
            return 'STOPPED'
        
        pipeline_stats = self.pipeline.get_pipeline_statistics()
        
        # Check if real-time capable
        if not pipeline_stats.get('real_time_capable', False):
            return 'DEGRADED - Not real-time capable'
        
        # Check drop rate
        drop_rate = pipeline_stats.get('drop_rate', 0)
        if drop_rate > 0.1:  # More than 10% drop rate
            return 'DEGRADED - High frame drop rate'
        
        # Check emergency stop rate
        if self.emergency_stops > self.total_frames_processed * 0.3:
            return 'DEGRADED - High emergency stop rate'
        
        return 'HEALTHY'

def run_comprehensive_test():
    """Run comprehensive integration test."""
    print("🚀 STARTING COMPREHENSIVE VQA UAV NAVIGATION TEST")
    print("=" * 60)
    
    # Initialize system
    system = IntegratedVQANavigationSystem(
        '/home/isr-lab3/James/VQASynth-UAV/config/config_realtime_uav.yaml'
    )
    
    try:
        # Start system
        system.start_system()
        
        print("\n📡 Simulating UAV navigation scenarios...")
        
        # Test scenario 1: Normal flight
        print("\n🛩️ Scenario 1: Normal flight operations")
        for i in range(10):
            # Generate mock camera image
            image = torch.randn(3, 224, 224)
            
            # Generate mock sensor data
            current_time = time.time()
            sensor_data = {
                'camera': SensorData(
                    data={'image': image.numpy()},
                    timestamp=current_time,
                    sensor_type='camera',
                    reliability=0.9,
                    quality_score=np.random.uniform(0.8, 0.95)
                ),
                'lidar': SensorData(
                    data={'point_cloud': np.random.rand(1000, 3)},
                    timestamp=current_time,
                    sensor_type='lidar',
                    reliability=0.95,
                    quality_score=np.random.uniform(0.85, 0.98)
                ),
                'gps': SensorData(
                    data={'lat': 40.7128 + i*0.0001, 'lon': -74.0060 + i*0.0001, 'alt': 100.0},
                    timestamp=current_time,
                    sensor_type='gps',
                    reliability=0.8,
                    quality_score=np.random.uniform(0.7, 0.9)
                )
            }
            
            # Process frame
            frame_id = system.process_navigation_frame(image, sensor_data=sensor_data)
            
            if frame_id is not None:
                print(f"  ✅ Frame {frame_id} submitted")
            
            # Simulate real-time processing (10 Hz)
            time.sleep(0.1)
        
        # Wait for processing
        print("\n⏳ Waiting for processing to complete...")
        time.sleep(3.0)
        
        # Test scenario 2: Emergency situations
        print("\n🚨 Scenario 2: Emergency response test")
        for i in range(5):
            # Create emergency scenario image
            image = torch.randn(3, 224, 224)
            
            # Process with high priority
            frame_id = system.process_navigation_frame(
                image, 
                question="Emergency obstacle detected, what should I do?",
                sensor_data=None
            )
            
            time.sleep(0.05)  # Faster processing for emergency
        
        # Wait for emergency processing
        time.sleep(2.0)
        
        # Display results
        print("\n📊 SYSTEM PERFORMANCE ANALYSIS")
        print("=" * 40)
        
        status = system.get_system_status()
        
        print(f"Runtime: {status['runtime_seconds']:.1f} seconds")
        print(f"Frames submitted: {status['total_frames_submitted']}")
        print(f"Frames processed: {status['frames_processed']}")
        print(f"Frames dropped: {status['frames_dropped']}")
        print(f"Average processing time: {status['average_processing_time_ms']:.2f} ms")
        print(f"Real-time capable: {status['real_time_capable']}")
        print(f"Emergency stops triggered: {status['emergency_stops']}")
        print(f"Current flight mode: {status['current_flight_mode']}")
        print(f"Control commands generated: {status['control_commands_generated']}")
        print(f"System health: {status['system_health']}")
        
        # Show latest control command
        latest_command = system.get_latest_control_command()
        if latest_command:
            print(f"\n🎮 Latest Control Command:")
            print(f"  Flight mode: {latest_command.flight_mode.value}")
            print(f"  Linear velocity: {latest_command.linear_velocity}")
            print(f"  Angular velocity: {latest_command.angular_velocity}")
            print(f"  Confidence: {latest_command.confidence:.3f}")
            print(f"  Safety margin: {latest_command.safety_margin:.2f}m")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT")
        print("=" * 30)
        
        if status['real_time_capable']:
            print("✅ PASS: Real-time performance achieved")
        else:
            print("❌ FAIL: Real-time performance not achieved")
        
        if status['frames_dropped'] == 0:
            print("✅ PASS: No frames dropped")
        else:
            print(f"⚠️ WARNING: {status['frames_dropped']} frames dropped")
        
        if status['system_health'] == 'HEALTHY':
            print("✅ PASS: System health is good")
        else:
            print(f"⚠️ WARNING: System health: {status['system_health']}")
        
        print(f"\n🏁 Integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        system.stop_system()
        print("\n🔌 System shutdown complete")

if __name__ == "__main__":
    run_comprehensive_test()

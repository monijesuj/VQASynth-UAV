"""
Real-time processing pipeline for UAV spatial VQA.
Implements parallel processing, memory management, and hardware optimization.
"""

import torch
import torch.multiprocessing as mp
import queue
import threading
import time
import numpy as np
from typing import Dict, Any, Optional, Callable
import yaml
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass

from vqasynth.realtime_optimizer import RealtimeOptimizer
from vqasynth.safety_spatial_vqa import SafetyCriticalSpatialVQA
from vqasynth.sensor_fusion import MultiModalSensorFusion, SensorData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineFrame:
    """Container for pipeline frame data."""
    frame_id: int
    timestamp: float
    image: torch.Tensor
    question: str
    sensor_data: Dict[str, SensorData]
    priority: int = 0  # Higher priority = process first

class RealtimePipeline:
    """Real-time spatial VQA pipeline for UAV deployment."""
    
    def __init__(self, config_path: str, base_model):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.realtime_config = self.config['realtime_config']
        self.max_inference_time = self.realtime_config['max_inference_time_ms'] / 1000.0
        
        # Initialize components
        self.optimizer = RealtimeOptimizer(config_path)
        self.safety_vqa = SafetyCriticalSpatialVQA(base_model, config_path)
        self.sensor_fusion = MultiModalSensorFusion(config_path)
        
        # Optimize the base model
        self.optimized_model = self.optimizer.optimize_model(base_model)
        
        # Pipeline queues
        self.input_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_queue = queue.PriorityQueue(maxsize=10)
        
        # Pipeline state
        self.is_running = False
        self.frame_counter = 0
        self.pipeline_threads = []
        
        # Performance monitoring
        self.frame_times = []
        self.dropped_frames = 0
        self.processed_frames = 0
        
        # Pipeline stages
        self.pipeline_stages = {
            'preprocessing': self._preprocess_stage,
            'sensor_fusion': self._sensor_fusion_stage,
            'vqa_inference': self._vqa_inference_stage,
            'safety_check': self._safety_check_stage,
            'postprocessing': self._postprocess_stage
        }
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def start_pipeline(self):
        """Start the real-time processing pipeline."""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        self.is_running = True
        logger.info("Starting real-time VQA pipeline...")
        
        # Start pipeline threads
        self.pipeline_threads = [
            threading.Thread(target=self._input_processor, daemon=True),
            threading.Thread(target=self._parallel_processor, daemon=True),
            threading.Thread(target=self._output_processor, daemon=True),
            threading.Thread(target=self._monitoring_thread, daemon=True)
        ]
        
        for thread in self.pipeline_threads:
            thread.start()
        
        logger.info("Pipeline started successfully")
    
    def stop_pipeline(self):
        """Stop the real-time processing pipeline."""
        self.is_running = False
        logger.info("Stopping pipeline...")
        
        # Wait for threads to finish
        for thread in self.pipeline_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        self.executor.shutdown(wait=True)
        logger.info("Pipeline stopped")
    
    def process_frame(self, image: torch.Tensor, question: str, 
                     sensor_data: Dict[str, SensorData] = None, 
                     priority: int = 0) -> int:
        """Submit a frame for processing."""
        if not self.is_running:
            logger.error("Pipeline is not running")
            return -1
        
        frame = PipelineFrame(
            frame_id=self.frame_counter,
            timestamp=time.time(),
            image=image,
            question=question,
            sensor_data=sensor_data or {},
            priority=priority
        )
        
        try:
            self.input_queue.put_nowait(frame)
            self.frame_counter += 1
            return frame.frame_id
        except queue.Full:
            self.dropped_frames += 1
            logger.warning(f"Input queue full, dropping frame {self.frame_counter}")
            return -1
    
    def get_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get processing result if available."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _input_processor(self):
        """Process input frames and queue them for parallel processing."""
        while self.is_running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                
                # Add to priority queue for processing
                priority_item = (-frame.priority, frame.timestamp, frame)
                self.processing_queue.put(priority_item)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Input processor error: {e}")
    
    def _parallel_processor(self):
        """Main parallel processing loop."""
        while self.is_running:
            try:
                # Get frame from priority queue
                priority_item = self.processing_queue.get(timeout=0.1)
                _, _, frame = priority_item
                
                # Check frame age - drop if too old
                frame_age = time.time() - frame.timestamp
                if frame_age > self.max_inference_time:
                    self.dropped_frames += 1
                    logger.warning(f"Dropping aged frame {frame.frame_id} (age: {frame_age*1000:.1f}ms)")
                    continue
                
                # Process frame through pipeline stages
                future = self.executor.submit(self._process_frame_parallel, frame)
                
                # Handle result asynchronously
                def handle_result(fut: Future):
                    try:
                        result = fut.result()
                        if result:
                            self.result_queue.put_nowait(result)
                            self.processed_frames += 1
                    except queue.Full:
                        logger.warning("Result queue full, dropping result")
                    except Exception as e:
                        logger.error(f"Processing error: {e}")
                
                future.add_done_callback(handle_result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Parallel processor error: {e}")
    
    def _process_frame_parallel(self, frame: PipelineFrame) -> Optional[Dict[str, Any]]:
        """Process a single frame through all pipeline stages."""
        start_time = time.time()
        
        try:
            # Stage 1: Preprocessing
            preprocessed_data = self._preprocess_stage(frame)
            
            # Stage 2: Sensor Fusion (parallel with preprocessing if possible)
            if frame.sensor_data:
                fusion_result = self._sensor_fusion_stage(frame.sensor_data)
            else:
                fusion_result = None
            
            # Stage 3: VQA Inference
            vqa_result = self._vqa_inference_stage(preprocessed_data)
            
            # Stage 4: Safety Check
            safety_result = self._safety_check_stage(vqa_result, fusion_result)
            
            # Stage 5: Postprocessing
            final_result = self._postprocess_stage(safety_result, frame)
            
            processing_time = time.time() - start_time
            self.frame_times.append(processing_time)
            
            # Add timing information
            final_result.update({
                'frame_id': frame.frame_id,
                'processing_time_ms': processing_time * 1000,
                'frame_age_ms': (time.time() - frame.timestamp) * 1000,
                'timestamp': frame.timestamp
            })
            
            return final_result
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return None
    
    def _preprocess_stage(self, frame: PipelineFrame) -> Dict[str, Any]:
        """Preprocess input data for VQA inference."""
        # Normalize image
        if frame.image.max() > 1.0:
            image = frame.image.float() / 255.0
        else:
            image = frame.image.float()
        
        # Ensure correct dimensions [B, C, H, W]
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        return {
            'image': image,
            'question': frame.question,
            'timestamp': frame.timestamp
        }
    
    def _sensor_fusion_stage(self, sensor_data: Dict[str, SensorData]) -> Dict[str, Any]:
        """Fuse multi-modal sensor data."""
        camera_data = sensor_data.get('camera')
        lidar_data = sensor_data.get('lidar')
        gps_data = sensor_data.get('gps')
        imu_data = sensor_data.get('imu')
        
        return self.sensor_fusion.fuse_spatial_data(
            camera_data=camera_data,
            lidar_data=lidar_data,
            gps_data=gps_data,
            imu_data=imu_data
        )
    
    def _vqa_inference_stage(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform VQA inference with uncertainty quantification."""
        return self.safety_vqa.predict_with_uncertainty(
            preprocessed_data['image'],
            preprocessed_data['question']
        )
    
    def _safety_check_stage(self, vqa_result: Dict[str, Any], 
                           fusion_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform safety checks and generate navigation commands."""
        # Enhance VQA result with sensor fusion data
        if fusion_result and fusion_result.get('fused_spatial_data'):
            vqa_result['sensor_fusion'] = fusion_result
            
            # Adjust confidence based on sensor consistency
            consistency_bonus = fusion_result['consistency_score'] * 0.1
            vqa_result['confidence'] = min(1.0, vqa_result['confidence'] + consistency_bonus)
        
        # Generate navigation command
        nav_command = self.safety_vqa.get_navigation_command(vqa_result)
        
        # Emergency stop check
        emergency_response = self.safety_vqa.emergency_stop_trigger(vqa_result)
        
        return {
            'vqa_result': vqa_result,
            'navigation_command': nav_command,
            'emergency_response': emergency_response,
            'safety_status': 'SAFE' if emergency_response['action'] == 'CONTINUE' else 'EMERGENCY'
        }
    
    def _postprocess_stage(self, safety_result: Dict[str, Any], 
                          frame: PipelineFrame) -> Dict[str, Any]:
        """Postprocess results for output."""
        return {
            'spatial_decision': safety_result['vqa_result'],
            'navigation_command': safety_result['navigation_command'],
            'safety_status': safety_result['safety_status'],
            'emergency_info': safety_result['emergency_response'],
            'frame_metadata': {
                'frame_id': frame.frame_id,
                'original_timestamp': frame.timestamp,
                'question': frame.question
            }
        }
    
    def _output_processor(self):
        """Process and deliver output results."""
        while self.is_running:
            try:
                result = self.result_queue.get(timeout=0.1)
                
                # Here you would send results to the navigation system
                # For now, we just log the result
                logger.debug(f"Result ready for frame {result.get('frame_metadata', {}).get('frame_id', 'unknown')}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Output processor error: {e}")
    
    def _monitoring_thread(self):
        """Monitor pipeline performance."""
        while self.is_running:
            time.sleep(1.0)  # Monitor every second
            
            if self.frame_times:
                avg_time = np.mean(self.frame_times[-10:])  # Last 10 frames
                
                if avg_time > self.max_inference_time:
                    logger.warning(f"Pipeline running slow: {avg_time*1000:.1f}ms (target: {self.max_inference_time*1000:.1f}ms)")
                
                # Clear old timing data
                if len(self.frame_times) > 100:
                    self.frame_times = self.frame_times[-50:]
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance statistics."""
        if not self.frame_times:
            return {"status": "no_data"}
        
        recent_times = self.frame_times[-50:]  # Last 50 frames
        
        return {
            'processed_frames': self.processed_frames,
            'dropped_frames': self.dropped_frames,
            'drop_rate': self.dropped_frames / (self.processed_frames + self.dropped_frames) if (self.processed_frames + self.dropped_frames) > 0 else 0,
            'avg_processing_time_ms': np.mean(recent_times) * 1000,
            'max_processing_time_ms': np.max(recent_times) * 1000,
            'min_processing_time_ms': np.min(recent_times) * 1000,
            'std_processing_time_ms': np.std(recent_times) * 1000,
            'real_time_capable': np.percentile(recent_times, 95) <= self.max_inference_time,
            'queue_sizes': {
                'input': self.input_queue.qsize(),
                'processing': self.processing_queue.qsize(),
                'result': self.result_queue.qsize()
            },
            'pipeline_status': 'running' if self.is_running else 'stopped'
        }
    
    def __del__(self):
        """Cleanup pipeline resources."""
        try:
            if hasattr(self, 'is_running') and self.is_running:
                self.stop_pipeline()
        except Exception:
            pass  # Ignore cleanup errors


def create_realtime_pipeline_test():
    """Create test script for real-time pipeline."""
    test_script = '''
import sys
import os
sys.path.append('/home/isr-lab3/James/VQASynth-UAV')

import torch
import time
import numpy as np
from vqasynth.realtime_pipeline import RealtimePipeline
from vqasynth.sensor_fusion import SensorData

class MockVQAModel:
    """Mock VQA model for testing."""
    
    def train(self):
        pass
    
    def eval(self):
        pass
    
    def __call__(self, image, question):
        # Simulate processing time
        time.sleep(np.random.uniform(0.01, 0.03))
        
        return {
            'answer': f"Navigate forward {np.random.uniform(3, 8):.1f} meters",
            'forward_distance': np.random.uniform(3, 8),
            'lateral_distance': np.random.uniform(-1, 1),
            'vertical_distance': np.random.uniform(-0.5, 0.5),
            'obstacle_detected': np.random.random() > 0.8
        }

def test_realtime_pipeline():
    """Test the real-time processing pipeline."""
    print("Testing Real-Time VQA Pipeline...")
    
    # Create mock model and pipeline
    mock_model = MockVQAModel()
    pipeline = RealtimePipeline(
        '/home/isr-lab3/James/VQASynth-UAV/config/config_realtime_uav.yaml',
        mock_model
    )
    
    # Start pipeline
    pipeline.start_pipeline()
    
    print("Pipeline started. Processing test frames...")
    
    # Simulate real-time frame processing
    for i in range(20):
        # Create mock frame data
        image = torch.randn(3, 224, 224)
        question = f"What is the safe navigation path in frame {i}?"
        
        # Create mock sensor data
        current_time = time.time()
        sensor_data = {
            'camera': SensorData(
                data={'image': image},
                timestamp=current_time,
                sensor_type='camera',
                reliability=0.9,
                quality_score=np.random.uniform(0.7, 0.95)
            ),
            'lidar': SensorData(
                data={'point_cloud': np.random.rand(1000, 3)},
                timestamp=current_time,
                sensor_type='lidar',
                reliability=0.95,
                quality_score=np.random.uniform(0.8, 0.98)
            )
        }
        
        # Submit frame for processing
        frame_id = pipeline.process_frame(image, question, sensor_data, priority=i%3)
        
        if frame_id >= 0:
            print(f"Submitted frame {frame_id}")
        
        # Check for results
        result = pipeline.get_result(timeout=0.01)
        if result:
            print(f"  Result: {result['safety_status']} - {result['navigation_command']['action']}")
        
        # Simulate real-time processing interval
        time.sleep(0.05)  # 20 FPS
    
    # Wait for remaining results
    print("\\nWaiting for remaining results...")
    time.sleep(2.0)
    
    # Check remaining results
    while True:
        result = pipeline.get_result(timeout=0.1)
        if result is None:
            break
        print(f"Final result: {result['safety_status']} - Frame {result['frame_metadata']['frame_id']}")
    
    # Get performance statistics
    stats = pipeline.get_pipeline_statistics()
    print("\\n=== PIPELINE PERFORMANCE ===")
    print(f"Processed frames: {stats['processed_frames']}")
    print(f"Dropped frames: {stats['dropped_frames']}")
    print(f"Drop rate: {stats['drop_rate']:.2%}")
    print(f"Average processing time: {stats['avg_processing_time_ms']:.2f} ms")
    print(f"Real-time capable: {stats['real_time_capable']}")
    print(f"Queue sizes: {stats['queue_sizes']}")
    
    # Stop pipeline
    pipeline.stop_pipeline()
    print("\\nPipeline test completed.")

if __name__ == "__main__":
    test_realtime_pipeline()
'''
    
    with open('/home/isr-lab3/James/VQASynth-UAV/test_realtime_pipeline.py', 'w') as f:
        f.write(test_script)

if __name__ == "__main__":
    create_realtime_pipeline_test()

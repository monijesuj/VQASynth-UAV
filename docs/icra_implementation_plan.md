# ICRA Implementation Plan: Real-Time Spatial VQA for Autonomous UAV Navigation
## 7-Day Step-by-Step Implementation Guide

---

## **Day 1: Real-Time Architecture Optimization**

### **Step 1.1: Analyze Current Pipeline Performance**
```bash
# Profile existing VQASynth pipeline
python -m cProfile -o profile_results.prof run_pipeline.py
python analyze_performance.py --profile profile_results.prof
```

**Tasks:**
- Identify bottlenecks in current VQASynth stages
- Measure inference time for each component
- Analyze memory usage patterns
- Document baseline performance metrics

### **Step 1.2: Create Real-Time Configuration**
```yaml
# config/config_realtime_uav.yaml
directories:
  output_dir: /home/isr-lab3/James/vqasynth_output

realtime_config:
  max_inference_time_ms: 100  # Target: sub-100ms
  memory_limit_gb: 4         # Typical UAV compute constraint
  batch_size: 1              # Real-time single frame processing
  
optimization:
  use_tensorrt: true         # GPU optimization
  use_quantization: true     # INT8 for speed
  parallel_stages: true     # Pipeline parallelization
  
uav_constraints:
  max_power_watts: 25       # Typical UAV power budget
  operating_temp_max: 70    # Thermal constraints
  vibration_tolerance: high # UAV environment
```

### **Step 1.3: Implement Memory-Efficient Processing**
Create: `vqasynth/realtime_optimizer.py`
```python
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
import tensorrt as trt

class RealtimeOptimizer:
    def __init__(self, config):
        self.config = config
        self.memory_budget = config['memory_limit_gb'] * 1e9
        
    def optimize_model(self, model):
        """Optimize model for real-time UAV deployment"""
        # Dynamic quantization for speed
        quantized_model = quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        
        # TensorRT optimization if available
        if self.config.get('use_tensorrt', False):
            quantized_model = self._tensorrt_optimize(quantized_model)
            
        return quantized_model
    
    def setup_parallel_pipeline(self):
        """Setup parallel processing for pipeline stages"""
        from concurrent.futures import ThreadPoolExecutor
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        return self.executor
```

---

## **Day 2: Safety-Critical Decision Framework**

### **Step 2.1: Implement Uncertainty-Aware Spatial Reasoning**
Create: `vqasynth/safety_spatial_vqa.py`
```python
import torch
import torch.nn.functional as F
import numpy as np

class SafetyCriticalSpatialVQA:
    def __init__(self, base_model, safety_config):
        self.base_model = base_model
        self.safety_config = safety_config
        self.min_confidence = safety_config['min_confidence_threshold']
        self.safety_margin = safety_config['spatial_safety_margin_meters']
        
    def predict_with_uncertainty(self, image, question):
        """Generate spatial answer with uncertainty bounds"""
        
        # Multiple forward passes for uncertainty estimation
        predictions = []
        with torch.no_grad():
            for _ in range(10):  # Monte Carlo dropout
                pred = self.base_model(image, question)
                predictions.append(pred)
        
        # Calculate mean and uncertainty
        mean_pred = torch.stack(predictions).mean(dim=0)
        uncertainty = torch.stack(predictions).std(dim=0)
        
        # Safety check
        confidence = 1.0 - uncertainty.mean().item()
        is_safe = confidence >= self.min_confidence
        
        return {
            'answer': mean_pred,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'is_safe_decision': is_safe,
            'safety_margin': self.calculate_safety_margin(mean_pred)
        }
    
    def calculate_safety_margin(self, spatial_answer):
        """Calculate safe distance margin for navigation"""
        # Extract spatial information from answer
        if 'distance' in spatial_answer:
            distance = float(spatial_answer['distance'])
            return max(0, distance - self.safety_margin)
        return 0
        
    def emergency_stop_trigger(self, spatial_decision):
        """Trigger emergency stop if spatial decision is unsafe"""
        if not spatial_decision['is_safe_decision']:
            return {
                'action': 'EMERGENCY_STOP',
                'reason': f'Low confidence: {spatial_decision["confidence"]:.2f}',
                'recommendation': 'Switch to manual control or SLAM backup'
            }
        return {'action': 'CONTINUE'}
```

### **Step 2.2: Multi-Modal Sensor Fusion**
Create: `vqasynth/sensor_fusion.py`
```python
class MultiModalSensorFusion:
    def __init__(self):
        self.sensors = {
            'camera': {'weight': 0.7, 'reliability': 0.9},
            'lidar': {'weight': 0.2, 'reliability': 0.95},
            'gps': {'weight': 0.1, 'reliability': 0.8}
        }
    
    def fuse_spatial_data(self, camera_data, lidar_data=None, gps_data=None):
        """Fuse multi-modal sensor data for spatial reasoning"""
        
        # Dynamic weighting based on sensor availability and reliability
        available_sensors = []
        if camera_data is not None:
            available_sensors.append(('camera', camera_data))
        if lidar_data is not None:
            available_sensors.append(('lidar', lidar_data))
        if gps_data is not None:
            available_sensors.append(('gps', gps_data))
        
        # Weighted fusion
        fused_data = self._weighted_fusion(available_sensors)
        
        # Cross-modal validation
        consistency_score = self._validate_consistency(available_sensors)
        
        return {
            'fused_spatial_data': fused_data,
            'consistency_score': consistency_score,
            'active_sensors': [s[0] for s in available_sensors]
        }
```

---

## **Day 3: Hardware Integration & Edge Deployment**

### **Step 3.1: UAV Hardware Optimization**
Create: `deployment/uav_hardware_config.py`
```python
import psutil
import GPUtil

class UAVHardwareManager:
    def __init__(self):
        self.cpu_cores = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_available = len(GPUtil.getGPUs()) > 0
        
    def optimize_for_hardware(self):
        """Configure processing based on available hardware"""
        config = {
            'batch_size': 1,  # Real-time processing
            'num_workers': min(4, self.cpu_cores - 1),  # Leave CPU for flight control
            'use_gpu': self.gpu_available,
            'memory_fraction': 0.6  # Reserve memory for other systems
        }
        
        # Thermal throttling protection
        if self.get_cpu_temp() > 70:
            config['num_workers'] = max(1, config['num_workers'] - 1)
            
        return config
    
    def monitor_system_health(self):
        """Monitor system health during operation"""
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_temp': self.get_cpu_temp(),
            'gpu_temp': self.get_gpu_temp() if self.gpu_available else None
        }
```

### **Step 3.2: Real-Time Processing Pipeline**
Create: `vqasynth/realtime_pipeline.py`
```python
import queue
import threading
import time
from collections import deque

class RealtimeSpatialVQAPipeline:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.frame_queue = queue.Queue(maxsize=3)  # Small buffer for real-time
        self.result_queue = queue.Queue()
        self.processing_times = deque(maxlen=100)  # Performance monitoring
        
    def start_pipeline(self):
        """Start real-time processing pipeline"""
        # Processing thread
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def add_frame(self, frame, question):
        """Add frame for processing (non-blocking)"""
        try:
            self.frame_queue.put_nowait((frame, question, time.time()))
        except queue.Full:
            # Drop oldest frame if queue is full (real-time constraint)
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((frame, question, time.time()))
            except queue.Empty:
                pass
    
    def _process_frames(self):
        """Main processing loop"""
        while True:
            try:
                frame, question, timestamp = self.frame_queue.get(timeout=1.0)
                
                start_time = time.time()
                result = self.model.predict_with_uncertainty(frame, question)
                processing_time = (time.time() - start_time) * 1000  # ms
                
                self.processing_times.append(processing_time)
                
                # Add timestamp and performance info
                result.update({
                    'timestamp': timestamp,
                    'processing_time_ms': processing_time,
                    'frame_age_ms': (time.time() - timestamp) * 1000
                })
                
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
```

---

## **Day 4: Navigation Integration & Path Planning**

### **Step 4.1: VQA-to-Control Interface**
Create: `navigation/vqa_navigation_controller.py`
```python
import numpy as np
from mavros_msgs.msg import OverrideRCIn
from geometry_msgs.msg import Twist

class VQANavigationController:
    def __init__(self, safety_config):
        self.safety_config = safety_config
        self.max_velocity = safety_config['max_velocity_ms']
        self.safety_distance = safety_config['min_obstacle_distance_m']
        
    def spatial_vqa_to_control(self, vqa_result, current_pose):
        """Convert spatial VQA result to control commands"""
        
        # Safety check first
        if not vqa_result['is_safe_decision']:
            return self._emergency_stop()
        
        # Extract spatial information
        spatial_info = self._extract_spatial_info(vqa_result['answer'])
        
        # Generate control command
        control_cmd = self._generate_control_command(spatial_info, current_pose)
        
        # Apply safety constraints
        safe_cmd = self._apply_safety_constraints(control_cmd, vqa_result)
        
        return safe_cmd
    
    def _extract_spatial_info(self, vqa_answer):
        """Extract actionable spatial information from VQA answer"""
        # Parse natural language answer to spatial coordinates
        # Example: "The drone should move 3 meters forward and 2 meters left"
        return {
            'forward_distance': 3.0,
            'lateral_distance': -2.0,  # Negative for left
            'vertical_distance': 0.0,
            'obstacle_distance': 5.2
        }
    
    def _generate_control_command(self, spatial_info, current_pose):
        """Generate velocity commands from spatial information"""
        cmd = Twist()
        
        # Forward/backward velocity
        if spatial_info['obstacle_distance'] > self.safety_distance:
            cmd.linear.x = min(self.max_velocity, 
                             spatial_info['forward_distance'] * 0.5)
        
        # Lateral velocity
        cmd.linear.y = spatial_info['lateral_distance'] * 0.3
        
        # Vertical velocity
        cmd.linear.z = spatial_info['vertical_distance'] * 0.2
        
        return cmd
```

### **Step 4.2: Dynamic Path Planning**
Create: `navigation/dynamic_path_planner.py`
```python
import numpy as np
from scipy.spatial.distance import cdist

class DynamicPathPlanner:
    def __init__(self, config):
        self.config = config
        self.waypoint_buffer = []
        self.obstacle_map = {}
        
    def update_path_from_vqa(self, vqa_results, goal_position):
        """Update path plan based on spatial VQA understanding"""
        
        # Extract obstacles from VQA results
        obstacles = self._extract_obstacles_from_vqa(vqa_results)
        
        # Update obstacle map
        self._update_obstacle_map(obstacles)
        
        # Replan path if needed
        if self._path_blocked(self.waypoint_buffer, obstacles):
            new_path = self._replan_path(goal_position, obstacles)
            self.waypoint_buffer = new_path
            
        return self.waypoint_buffer
    
    def _extract_obstacles_from_vqa(self, vqa_results):
        """Extract obstacle positions from VQA spatial reasoning"""
        obstacles = []
        
        for result in vqa_results:
            if 'obstacle' in result['answer'].lower():
                # Parse obstacle location from natural language
                obstacle_pos = self._parse_obstacle_position(result['answer'])
                if obstacle_pos:
                    obstacles.append({
                        'position': obstacle_pos,
                        'confidence': result['confidence'],
                        'radius': 2.0  # Safety margin
                    })
        
        return obstacles
```

---

## **Day 5: Flight Testing & Hardware Validation**

### **Step 5.1: Simulation Testing Setup**
```bash
# Install simulation environment
sudo apt-get install ros-noetic-gazebo-ros-pkgs
pip install gym-gazebo airsim-python

# Create simulation test scripts
mkdir -p simulation_tests/
```

Create: `simulation_tests/test_spatial_vqa_navigation.py`
```python
import airsim
import numpy as np
import cv2
import time

class SpatialVQASimulationTest:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
    def test_obstacle_avoidance(self):
        """Test spatial VQA obstacle avoidance in simulation"""
        
        # Take off
        self.client.takeoffAsync().join()
        
        test_scenarios = [
            {'name': 'narrow_gap', 'position': [10, 0, -5]},
            {'name': 'tree_avoidance', 'position': [20, 10, -5]},
            {'name': 'building_navigation', 'position': [30, 0, -10]}
        ]
        
        results = []
        for scenario in test_scenarios:
            result = self._run_scenario(scenario)
            results.append(result)
            
        return results
    
    def _run_scenario(self, scenario):
        """Run individual test scenario"""
        start_time = time.time()
        
        # Get camera image
        response = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])[0]
        
        # Convert to OpenCV format
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        # Run spatial VQA
        question = f"Can the drone safely navigate to position {scenario['position']}?"
        vqa_result = self.spatial_vqa_model.predict_with_uncertainty(img_rgb, question)
        
        # Execute navigation if safe
        if vqa_result['is_safe_decision']:
            self.client.moveToPositionAsync(*scenario['position'], 2.0).join()
            success = True
        else:
            success = False
            
        return {
            'scenario': scenario['name'],
            'success': success,
            'confidence': vqa_result['confidence'],
            'processing_time': time.time() - start_time
        }
```

### **Step 5.2: Real Hardware Integration**
Create: `hardware/uav_integration.py`
```python
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class UAVHardwareIntegration:
    def __init__(self, vqa_pipeline):
        rospy.init_node('spatial_vqa_navigation')
        
        self.vqa_pipeline = vqa_pipeline
        self.bridge = CvBridge()
        
        # ROS subscribers and publishers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Start VQA pipeline
        self.vqa_pipeline.start_pipeline()
        
    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Add spatial navigation question
            question = "What obstacles are visible and how should the drone navigate?"
            
            # Add frame to processing pipeline
            self.vqa_pipeline.add_frame(cv_image, question)
            
            # Get latest result if available
            try:
                result = self.vqa_pipeline.result_queue.get_nowait()
                control_cmd = self.generate_control_from_vqa(result)
                self.cmd_pub.publish(control_cmd)
            except:
                pass  # No new results available
                
        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")
```

---

## **Day 6: Performance Benchmarking & Safety Analysis**

### **Step 6.1: Performance Metrics Collection**
Create: `evaluation/performance_benchmarks.py`
```python
import time
import numpy as np
import matplotlib.pyplot as plt

class PerformanceBenchmark:
    def __init__(self):
        self.metrics = {
            'inference_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'gpu_usage': [],
            'accuracy_scores': [],
            'safety_decisions': []
        }
        
    def run_benchmark_suite(self, vqa_pipeline, test_dataset):
        """Run comprehensive performance benchmark"""
        
        print("Starting performance benchmark...")
        
        for i, (image, question, ground_truth) in enumerate(test_dataset):
            start_time = time.time()
            
            # Memory usage before
            mem_before = self._get_memory_usage()
            
            # Run inference
            result = vqa_pipeline.predict_with_uncertainty(image, question)
            
            # Timing and memory
            inference_time = time.time() - start_time
            mem_after = self._get_memory_usage()
            
            # Collect metrics
            self.metrics['inference_times'].append(inference_time * 1000)  # ms
            self.metrics['memory_usage'].append(mem_after - mem_before)
            self.metrics['accuracy_scores'].append(self._calculate_accuracy(result, ground_truth))
            self.metrics['safety_decisions'].append(result['is_safe_decision'])
            
            if i % 100 == 0:
                print(f"Processed {i} samples...")
        
        return self._generate_report()
    
    def _generate_report(self):
        """Generate performance report"""
        report = {
            'avg_inference_time_ms': np.mean(self.metrics['inference_times']),
            'max_inference_time_ms': np.max(self.metrics['inference_times']),
            '95th_percentile_ms': np.percentile(self.metrics['inference_times'], 95),
            'real_time_capable': np.percentile(self.metrics['inference_times'], 95) < 100,
            'avg_accuracy': np.mean(self.metrics['accuracy_scores']),
            'safety_decision_rate': np.mean(self.metrics['safety_decisions']),
            'memory_efficiency': np.mean(self.metrics['memory_usage'])
        }
        
        print("Performance Report:")
        print(f"Average inference time: {report['avg_inference_time_ms']:.2f} ms")
        print(f"95th percentile: {report['95th_percentile_ms']:.2f} ms")
        print(f"Real-time capable: {report['real_time_capable']}")
        print(f"Average accuracy: {report['avg_accuracy']:.3f}")
        
        return report
```

### **Step 6.2: Safety Analysis Framework**
Create: `safety/safety_analysis.py`
```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

class SafetyAnalysis:
    def __init__(self):
        self.safety_logs = []
        self.failure_modes = {
            'low_confidence': 0,
            'sensor_failure': 0,
            'processing_timeout': 0,
            'inconsistent_decisions': 0
        }
        
    def analyze_safety_performance(self, test_results):
        """Comprehensive safety analysis"""
        
        # Safety decision accuracy
        safety_accuracy = self._calculate_safety_accuracy(test_results)
        
        # Failure mode analysis
        failure_analysis = self._analyze_failure_modes(test_results)
        
        # Risk assessment
        risk_levels = self._assess_risk_levels(test_results)
        
        return {
            'safety_accuracy': safety_accuracy,
            'failure_modes': failure_analysis,
            'risk_assessment': risk_levels,
            'recommendations': self._generate_safety_recommendations()
        }
    
    def _calculate_safety_accuracy(self, test_results):
        """Calculate safety decision accuracy"""
        true_safe = []
        pred_safe = []
        
        for result in test_results:
            true_safe.append(result['ground_truth_safe'])
            pred_safe.append(result['predicted_safe'])
        
        # Confusion matrix for safety decisions
        cm = confusion_matrix(true_safe, pred_safe)
        
        return {
            'confusion_matrix': cm,
            'accuracy': np.trace(cm) / np.sum(cm),
            'false_positive_rate': cm[0,1] / (cm[0,0] + cm[0,1]),  # Unsafe classified as safe
            'false_negative_rate': cm[1,0] / (cm[1,0] + cm[1,1])   # Safe classified as unsafe
        }
```

---

## **Day 7: Results Analysis & ICRA Paper Preparation**

### **Step 7.1: Comprehensive Results Analysis**
Create: `analysis/icra_results_analysis.py`
```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class ICRAResultsAnalysis:
    def __init__(self, performance_data, safety_data, flight_test_data):
        self.performance_data = performance_data
        self.safety_data = safety_data
        self.flight_test_data = flight_test_data
        
    def generate_icra_figures(self):
        """Generate figures for ICRA paper"""
        
        # Figure 1: Real-time performance analysis
        self._plot_realtime_performance()
        
        # Figure 2: Safety decision accuracy
        self._plot_safety_analysis()
        
        # Figure 3: Flight test results
        self._plot_flight_test_results()
        
        # Figure 4: System architecture diagram
        self._create_system_diagram()
        
    def _plot_realtime_performance(self):
        """Plot real-time performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Inference time distribution
        axes[0,0].hist(self.performance_data['inference_times'], bins=50)
        axes[0,0].axvline(x=100, color='red', linestyle='--', label='Real-time threshold')
        axes[0,0].set_xlabel('Inference Time (ms)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Inference Time Distribution')
        axes[0,0].legend()
        
        # Memory usage over time
        axes[0,1].plot(self.performance_data['memory_usage'])
        axes[0,1].set_xlabel('Time Steps')
        axes[0,1].set_ylabel('Memory Usage (MB)')
        axes[0,1].set_title('Memory Usage Over Time')
        
        # Accuracy vs. inference time trade-off
        axes[1,0].scatter(self.performance_data['inference_times'], 
                         self.performance_data['accuracy_scores'], alpha=0.6)
        axes[1,0].set_xlabel('Inference Time (ms)')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_title('Accuracy vs Speed Trade-off')
        
        # System resource utilization
        resource_data = ['CPU', 'Memory', 'GPU']
        utilization = [65, 45, 80]  # Example percentages
        axes[1,1].bar(resource_data, utilization)
        axes[1,1].set_ylabel('Utilization (%)')
        axes[1,1].set_title('System Resource Utilization')
        
        plt.tight_layout()
        plt.savefig('figures/realtime_performance.png', dpi=300)
        
    def generate_icra_paper_outline(self):
        """Generate ICRA paper structure and key points"""
        
        paper_outline = """
        # Real-Time Spatial VQA for Autonomous UAV Navigation in Dense Environments
        ## ICRA 2025 Submission
        
        ### Abstract (200 words)
        - Problem: Autonomous UAV navigation in dense environments requires real-time spatial reasoning
        - Approach: Real-time spatial VQA with uncertainty quantification and safety guarantees
        - Results: Sub-100ms inference, 95%+ safety accuracy, successful flight tests
        - Impact: Enables safe autonomous navigation in complex environments
        
        ### 1. Introduction
        - Motivation: Dense environment navigation challenges
        - Problem statement: Real-time spatial understanding for UAV safety
        - Contributions:
          1. Real-time spatial VQA architecture for UAV navigation
          2. Safety-critical decision framework with uncertainty bounds  
          3. Multi-modal sensor fusion for robust perception
          4. Hardware validation with flight test demonstrations
        
        ### 2. Related Work
        - Visual Question Answering in robotics
        - UAV navigation and obstacle avoidance
        - Real-time perception for autonomous systems
        - Safety-critical AI for robotics
        
        ### 3. System Architecture
        - Real-time VQA pipeline design
        - Safety-critical decision framework
        - Multi-modal sensor integration
        - Hardware-software co-optimization
        
        ### 4. Real-Time Implementation
        - Performance optimization strategies
        - Memory-efficient processing
        - Parallel pipeline architecture
        - Hardware deployment considerations
        
        ### 5. Safety Analysis Framework
        - Uncertainty quantification methods
        - Failure mode analysis
        - Emergency response mechanisms
        - Conservative decision making
        
        ### 6. Experimental Validation
        - Simulation testing results
        - Hardware flight test demonstrations
        - Performance benchmarking
        - Safety analysis results
        
        ### 7. Results and Discussion
        - Real-time performance: X.X ms average inference
        - Safety accuracy: XX.X% correct safety decisions
        - Flight test success: XX/XX successful autonomous flights
        - Comparison with baseline methods
        
        ### 8. Conclusion and Future Work
        - Summary of contributions
        - Impact on autonomous robotics
        - Future research directions
        - Deployment considerations
        """
        
        return paper_outline
```

---

## **Implementation Commands Summary:**

### **Setup Commands:**
```bash
# Day 1: Setup
mkdir -p config deployment navigation simulation_tests hardware evaluation safety analysis figures
pip install tensorrt-python torch-quantization psutil GPUtil airsim

# Day 2-3: Core implementation
python -m vqasynth.realtime_optimizer --config config/config_realtime_uav.yaml
python -m vqasynth.safety_spatial_vqa --test-mode

# Day 4-5: Integration and testing
python simulation_tests/test_spatial_vqa_navigation.py
roslaunch hardware/uav_integration.launch  # If using ROS

# Day 6-7: Analysis
python evaluation/performance_benchmarks.py --dataset test_data/
python safety/safety_analysis.py --results results/
python analysis/icra_results_analysis.py --generate-figures
```

### **Expected Deliverables:**
- ✅ Real-time spatial VQA system (sub-100ms inference)
- ✅ Safety-critical decision framework
- ✅ Hardware integration and flight test validation
- ✅ Comprehensive performance and safety analysis
- ✅ ICRA paper-ready results and figures

This implementation plan provides a complete roadmap from current VQASynth codebase to a full ICRA-ready real-time spatial VQA navigation system with hardware validation.

Would you like me to start implementing any specific component, or would you prefer to begin with Day 1 setup and optimization?

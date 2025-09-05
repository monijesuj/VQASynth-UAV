"""
Multi-modal sensor fusion module for robust spatial VQA.
Combines camera, LiDAR, GPS, and IMU data for enhanced spatial understanding.
"""

import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Any
import yaml
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensorData:
    """Container for sensor data with metadata."""
    data: Any
    timestamp: float
    sensor_type: str
    reliability: float
    quality_score: float = 1.0

class MultiModalSensorFusion:
    """Multi-modal sensor fusion for robust spatial VQA."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.sensor_config = config['sensor_fusion']
        self.sensors = {
            'camera': {
                'weight': self.sensor_config['camera']['weight'],
                'reliability': self.sensor_config['camera']['reliability']
            },
            'lidar': {
                'weight': self.sensor_config['lidar']['weight'],
                'reliability': self.sensor_config['lidar']['reliability']
            },
            'gps': {
                'weight': self.sensor_config['gps']['weight'],
                'reliability': self.sensor_config['gps']['reliability']
            }
        }
        
        # Sensor failure detection
        self.sensor_failures = {}
        self.sensor_history = {}
        
        # Fusion parameters
        self.max_sensor_age = 0.1  # 100ms max age for sensor data
        self.min_sensors_required = 1  # Minimum sensors needed
        
    def fuse_spatial_data(self, 
                         camera_data: Optional[SensorData] = None,
                         lidar_data: Optional[SensorData] = None,
                         gps_data: Optional[SensorData] = None,
                         imu_data: Optional[SensorData] = None) -> Dict[str, Any]:
        """Fuse multi-modal sensor data for spatial reasoning."""
        
        start_time = time.time()
        
        # Collect available sensors
        available_sensors = []
        current_time = time.time()
        
        for sensor_name, data in [
            ('camera', camera_data),
            ('lidar', lidar_data),
            ('gps', gps_data),
            ('imu', imu_data)
        ]:
            if data is not None and self._is_sensor_data_valid(data, current_time):
                available_sensors.append((sensor_name, data))
                self._update_sensor_history(sensor_name, data)
        
        if len(available_sensors) < self.min_sensors_required:
            logger.warning(f"Insufficient sensors available: {len(available_sensors)}")
            return self._generate_fallback_result()
        
        # Dynamic weighting based on sensor availability and reliability
        weights = self._calculate_dynamic_weights(available_sensors)
        
        # Weighted fusion
        fused_data = self._weighted_fusion(available_sensors, weights)
        
        # Cross-modal validation
        consistency_score = self._validate_consistency(available_sensors)
        
        # Environmental adaptation
        env_factors = self._assess_environmental_factors(available_sensors)
        
        fusion_time = time.time() - start_time
        
        return {
            'fused_spatial_data': fused_data,
            'consistency_score': consistency_score,
            'active_sensors': [s[0] for s in available_sensors],
            'sensor_weights': weights,
            'environmental_factors': env_factors,
            'fusion_time_ms': fusion_time * 1000,
            'sensor_health': self._get_sensor_health_status(),
            'quality_metrics': self._calculate_quality_metrics(available_sensors)
        }
    
    def _is_sensor_data_valid(self, sensor_data: SensorData, current_time: float) -> bool:
        """Check if sensor data is valid and recent."""
        # Check age
        age = current_time - sensor_data.timestamp
        if age > self.max_sensor_age:
            return False
        
        # Check quality score
        if sensor_data.quality_score < 0.5:
            return False
        
        # Check for sensor failure
        if self.sensor_failures.get(sensor_data.sensor_type, False):
            return False
        
        return True
    
    def _update_sensor_history(self, sensor_name: str, data: SensorData):
        """Update sensor history for failure detection."""
        if sensor_name not in self.sensor_history:
            self.sensor_history[sensor_name] = []
        
        self.sensor_history[sensor_name].append({
            'timestamp': data.timestamp,
            'quality_score': data.quality_score,
            'reliability': data.reliability
        })
        
        # Keep only recent history (last 100 readings)
        if len(self.sensor_history[sensor_name]) > 100:
            self.sensor_history[sensor_name] = self.sensor_history[sensor_name][-100:]
    
    def _calculate_dynamic_weights(self, available_sensors: List[Tuple[str, SensorData]]) -> Dict[str, float]:
        """Calculate dynamic weights based on sensor reliability and quality."""
        weights = {}
        total_weight = 0
        
        for sensor_name, sensor_data in available_sensors:
            if sensor_name in self.sensors:
                base_weight = self.sensors[sensor_name]['weight']
                reliability = self.sensors[sensor_name]['reliability']
                quality = sensor_data.quality_score
                
                # Adjust weight based on quality and reliability
                adjusted_weight = base_weight * reliability * quality
                weights[sensor_name] = adjusted_weight
                total_weight += adjusted_weight
        
        # Normalize weights
        if total_weight > 0:
            for sensor_name in weights:
                weights[sensor_name] /= total_weight
        
        return weights
    
    def _weighted_fusion(self, available_sensors: List[Tuple[str, SensorData]], 
                        weights: Dict[str, float]) -> Dict[str, Any]:
        """Perform weighted fusion of sensor data."""
        fused_result = {
            'spatial_coordinates': np.zeros(3),  # [x, y, z]
            'obstacle_distances': {},
            'confidence': 0.0,
            'heading': 0.0,
            'velocity': np.zeros(3)
        }
        
        total_confidence = 0.0
        
        for sensor_name, sensor_data in available_sensors:
            weight = weights.get(sensor_name, 0.0)
            
            if sensor_name == 'camera':
                camera_spatial = self._extract_camera_spatial_info(sensor_data.data)
                fused_result['spatial_coordinates'] += weight * camera_spatial['position']
                fused_result['obstacle_distances'].update(camera_spatial['obstacles'])
                total_confidence += weight * camera_spatial['confidence']
                
            elif sensor_name == 'lidar':
                lidar_spatial = self._extract_lidar_spatial_info(sensor_data.data)
                fused_result['spatial_coordinates'] += weight * lidar_spatial['position']
                fused_result['obstacle_distances'].update(lidar_spatial['obstacles'])
                total_confidence += weight * lidar_spatial['confidence']
                
            elif sensor_name == 'gps':
                gps_spatial = self._extract_gps_spatial_info(sensor_data.data)
                fused_result['spatial_coordinates'] += weight * gps_spatial['position']
                total_confidence += weight * gps_spatial['confidence']
                
            elif sensor_name == 'imu':
                imu_spatial = self._extract_imu_spatial_info(sensor_data.data)
                fused_result['heading'] += weight * imu_spatial['heading']
                fused_result['velocity'] += weight * imu_spatial['velocity']
        
        fused_result['confidence'] = total_confidence
        return fused_result
    
    def _extract_camera_spatial_info(self, camera_data: Any) -> Dict[str, Any]:
        """Extract spatial information from camera data."""
        # Mock implementation - replace with actual camera processing
        return {
            'position': np.array([0.0, 0.0, 0.0]),
            'obstacles': {'front': 5.0, 'left': 3.0, 'right': 3.0},
            'confidence': 0.8
        }
    
    def _extract_lidar_spatial_info(self, lidar_data: Any) -> Dict[str, Any]:
        """Extract spatial information from LiDAR data."""
        # Mock implementation - replace with actual LiDAR processing
        return {
            'position': np.array([0.0, 0.0, 0.0]),
            'obstacles': {'front': 4.8, 'left': 2.9, 'right': 3.1, 'back': 10.0},
            'confidence': 0.95
        }
    
    def _extract_gps_spatial_info(self, gps_data: Any) -> Dict[str, Any]:
        """Extract spatial information from GPS data."""
        # Mock implementation - replace with actual GPS processing
        return {
            'position': np.array([0.0, 0.0, 0.0]),
            'confidence': 0.7
        }
    
    def _extract_imu_spatial_info(self, imu_data: Any) -> Dict[str, Any]:
        """Extract spatial information from IMU data."""
        # Mock implementation - replace with actual IMU processing
        return {
            'heading': 0.0,
            'velocity': np.array([1.0, 0.0, 0.0]),
            'confidence': 0.9
        }
    
    def _validate_consistency(self, available_sensors: List[Tuple[str, SensorData]]) -> float:
        """Validate consistency across multiple sensors."""
        if len(available_sensors) < 2:
            return 1.0  # Single sensor, assume consistent
        
        # Extract spatial data from each sensor
        spatial_data = []
        for sensor_name, sensor_data in available_sensors:
            if sensor_name == 'camera':
                spatial_data.append(self._extract_camera_spatial_info(sensor_data.data))
            elif sensor_name == 'lidar':
                spatial_data.append(self._extract_lidar_spatial_info(sensor_data.data))
        
        if len(spatial_data) < 2:
            return 1.0
        
        # Compare obstacle distances between sensors
        consistency_scores = []
        for obstacle_type in ['front', 'left', 'right']:
            distances = []
            for data in spatial_data:
                if obstacle_type in data['obstacles']:
                    distances.append(data['obstacles'][obstacle_type])
            
            if len(distances) >= 2:
                std_dev = np.std(distances)
                mean_dist = np.mean(distances)
                # Normalize by mean distance (higher std_dev relative to mean = less consistent)
                if mean_dist > 0:
                    consistency = max(0, 1 - (std_dev / mean_dist))
                    consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _assess_environmental_factors(self, available_sensors: List[Tuple[str, SensorData]]) -> Dict[str, Any]:
        """Assess environmental factors affecting sensor performance."""
        factors = {
            'lighting_condition': 'good',  # good, poor, night
            'weather_condition': 'clear',  # clear, rain, fog, snow
            'vibration_level': 'low',      # low, medium, high
            'interference_level': 'none'   # none, low, medium, high
        }
        
        # Assess based on sensor quality scores
        avg_quality = np.mean([data.quality_score for _, data in available_sensors])
        
        if avg_quality < 0.3:
            factors['weather_condition'] = 'severe'
        elif avg_quality < 0.6:
            factors['weather_condition'] = 'poor'
        
        # Check for camera-specific issues
        camera_sensors = [data for name, data in available_sensors if name == 'camera']
        if camera_sensors and camera_sensors[0].quality_score < 0.4:
            factors['lighting_condition'] = 'poor'
        
        return factors
    
    def _get_sensor_health_status(self) -> Dict[str, str]:
        """Get health status of all sensors."""
        health_status = {}
        
        for sensor_name in self.sensors:
            if sensor_name in self.sensor_failures and self.sensor_failures[sensor_name]:
                health_status[sensor_name] = 'FAILED'
            elif sensor_name in self.sensor_history:
                recent_quality = [h['quality_score'] for h in self.sensor_history[sensor_name][-10:]]
                avg_quality = np.mean(recent_quality) if recent_quality else 1.0
                
                if avg_quality > 0.8:
                    health_status[sensor_name] = 'EXCELLENT'
                elif avg_quality > 0.6:
                    health_status[sensor_name] = 'GOOD'
                elif avg_quality > 0.4:
                    health_status[sensor_name] = 'DEGRADED'
                else:
                    health_status[sensor_name] = 'POOR'
            else:
                health_status[sensor_name] = 'UNKNOWN'
        
        return health_status
    
    def _calculate_quality_metrics(self, available_sensors: List[Tuple[str, SensorData]]) -> Dict[str, float]:
        """Calculate overall quality metrics for sensor fusion."""
        if not available_sensors:
            return {'overall_quality': 0.0, 'data_freshness': 0.0, 'sensor_diversity': 0.0}
        
        current_time = time.time()
        
        # Overall quality (weighted average of sensor quality scores)
        total_weight = 0
        weighted_quality = 0
        for sensor_name, sensor_data in available_sensors:
            weight = self.sensors.get(sensor_name, {}).get('weight', 0.1)
            weighted_quality += weight * sensor_data.quality_score
            total_weight += weight
        
        overall_quality = weighted_quality / total_weight if total_weight > 0 else 0.0
        
        # Data freshness (how recent is the data)
        ages = [current_time - sensor_data.timestamp for _, sensor_data in available_sensors]
        avg_age = np.mean(ages)
        data_freshness = max(0, 1 - (avg_age / self.max_sensor_age))
        
        # Sensor diversity (how many different sensor types)
        unique_sensors = len(set(sensor_name for sensor_name, _ in available_sensors))
        max_sensors = len(self.sensors)
        sensor_diversity = unique_sensors / max_sensors
        
        return {
            'overall_quality': overall_quality,
            'data_freshness': data_freshness,
            'sensor_diversity': sensor_diversity,
            'fusion_confidence': (overall_quality + data_freshness + sensor_diversity) / 3
        }
    
    def _generate_fallback_result(self) -> Dict[str, Any]:
        """Generate fallback result when insufficient sensors are available."""
        return {
            'fused_spatial_data': {
                'spatial_coordinates': np.zeros(3),
                'obstacle_distances': {},
                'confidence': 0.0,
                'heading': 0.0,
                'velocity': np.zeros(3)
            },
            'consistency_score': 0.0,
            'active_sensors': [],
            'sensor_weights': {},
            'environmental_factors': {'status': 'insufficient_data'},
            'fusion_time_ms': 0.0,
            'sensor_health': self._get_sensor_health_status(),
            'quality_metrics': {'overall_quality': 0.0, 'data_freshness': 0.0, 'sensor_diversity': 0.0},
            'fallback_mode': True
        }
    
    def detect_sensor_failure(self, sensor_name: str, threshold: float = 0.3) -> bool:
        """Detect sensor failure based on historical quality scores."""
        if sensor_name not in self.sensor_history:
            return False
        
        recent_scores = [h['quality_score'] for h in self.sensor_history[sensor_name][-20:]]
        
        if len(recent_scores) < 5:
            return False
        
        avg_recent_quality = np.mean(recent_scores)
        
        if avg_recent_quality < threshold:
            self.sensor_failures[sensor_name] = True
            logger.warning(f"Sensor failure detected: {sensor_name} (quality: {avg_recent_quality:.2f})")
            return True
        
        return False
    
    def reset_sensor_failure(self, sensor_name: str):
        """Reset sensor failure status."""
        if sensor_name in self.sensor_failures:
            del self.sensor_failures[sensor_name]
            logger.info(f"Sensor failure status reset: {sensor_name}")
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive sensor fusion statistics."""
        return {
            'active_sensors': list(self.sensors.keys()),
            'failed_sensors': list(self.sensor_failures.keys()),
            'sensor_history_length': {name: len(history) for name, history in self.sensor_history.items()},
            'average_sensor_quality': {
                name: np.mean([h['quality_score'] for h in history[-10:]]) if history else 0.0
                for name, history in self.sensor_history.items()
            },
            'sensor_weights': self.sensors
        }


def create_sensor_fusion_test():
    """Create test script for sensor fusion."""
    test_script = '''
import sys
import os
sys.path.append('/home/isr-lab3/James/VQASynth-UAV')

import numpy as np
import time
from vqasynth.sensor_fusion import MultiModalSensorFusion, SensorData

def test_sensor_fusion():
    """Test multi-modal sensor fusion."""
    print("Testing Multi-Modal Sensor Fusion...")
    
    fusion_system = MultiModalSensorFusion(
        '/home/isr-lab3/James/VQASynth-UAV/config/config_realtime_uav.yaml'
    )
    
    # Simulate sensor data
    current_time = time.time()
    
    camera_data = SensorData(
        data={'image': np.random.rand(224, 224, 3)},
        timestamp=current_time,
        sensor_type='camera',
        reliability=0.9,
        quality_score=0.85
    )
    
    lidar_data = SensorData(
        data={'point_cloud': np.random.rand(1000, 3)},
        timestamp=current_time,
        sensor_type='lidar',
        reliability=0.95,
        quality_score=0.92
    )
    
    gps_data = SensorData(
        data={'lat': 40.7128, 'lon': -74.0060, 'alt': 100.0},
        timestamp=current_time,
        sensor_type='gps',
        reliability=0.8,
        quality_score=0.75
    )
    
    print("\\nFusing sensor data...")
    result = fusion_system.fuse_spatial_data(
        camera_data=camera_data,
        lidar_data=lidar_data,
        gps_data=gps_data
    )
    
    print(f"Active sensors: {result['active_sensors']}")
    print(f"Consistency score: {result['consistency_score']:.3f}")
    print(f"Fusion time: {result['fusion_time_ms']:.2f} ms")
    print(f"Sensor weights: {result['sensor_weights']}")
    print(f"Quality metrics: {result['quality_metrics']}")
    print(f"Environmental factors: {result['environmental_factors']}")
    
    # Test sensor failure detection
    print("\\nTesting sensor failure detection...")
    
    # Simulate degraded sensor
    for i in range(25):
        degraded_data = SensorData(
            data={'image': np.random.rand(224, 224, 3)},
            timestamp=time.time(),
            sensor_type='camera',
            reliability=0.9,
            quality_score=0.2  # Poor quality
        )
        
        fusion_system.fuse_spatial_data(camera_data=degraded_data)
        
        if fusion_system.detect_sensor_failure('camera'):
            print("Camera sensor failure detected!")
            break
    
    # Get statistics
    stats = fusion_system.get_fusion_statistics()
    print(f"\\nFusion statistics: {stats}")

if __name__ == "__main__":
    test_sensor_fusion()
'''
    
    with open('/home/isr-lab3/James/VQASynth-UAV/test_sensor_fusion.py', 'w') as f:
        f.write(test_script)

if __name__ == "__main__":
    create_sensor_fusion_test()

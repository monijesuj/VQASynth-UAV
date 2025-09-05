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
    
    print("\nFusing sensor data...")
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
    print("\nTesting sensor failure detection...")
    
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
    print(f"\nFusion statistics: {stats}")

if __name__ == "__main__":
    test_sensor_fusion()

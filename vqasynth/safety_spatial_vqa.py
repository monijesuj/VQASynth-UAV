"""
Safety-critical spatial VQA module for autonomous UAV navigation.
Implements uncertainty-aware spatial reasoning with conservative decision making.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafetyCriticalSpatialVQA:
    """Safety-critical spatial VQA system with uncertainty quantification."""
    
    def __init__(self, base_model, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.base_model = base_model
        self.safety_config = config['safety_config']
        self.navigation_config = config['navigation_config']
        
        self.min_confidence = self.safety_config['min_confidence_threshold']
        self.safety_margin = self.safety_config['spatial_safety_margin_meters']
        self.emergency_stop_enabled = self.safety_config['emergency_stop_enabled']
        
        # Safety monitoring
        self.safety_decisions = []
        self.emergency_stops = 0
        self.total_decisions = 0
        
        # Confidence thresholds for different actions (based on config)
        base_threshold = self.min_confidence
        self.confidence_thresholds = {
            'normal_navigation': min(0.9, base_threshold + 0.4),
            'conservative_navigation': min(0.7, base_threshold + 0.2), 
            'human_intervention': base_threshold,
            'emergency_stop': 0.0
        }
    
    def predict_with_uncertainty(self, image: torch.Tensor, question: str) -> Dict[str, Any]:
        """Generate spatial answer with uncertainty bounds using Monte Carlo dropout."""
        start_time = time.time()
        
        # Enable dropout for uncertainty estimation
        self.base_model.train()  # Enable dropout
        
        # Multiple forward passes for uncertainty estimation
        predictions = []
        with torch.no_grad():
            for _ in range(5):  # Reduced from 10 to 5 for speed optimization
                pred = self.base_model(image, question)
                predictions.append(pred)
        
        # Calculate mean and uncertainty
        if isinstance(predictions[0], dict):
            # Handle dictionary predictions
            mean_pred = self._aggregate_dict_predictions(predictions)
            uncertainty = self._calculate_dict_uncertainty(predictions)
        else:
            # Handle tensor predictions
            mean_pred = torch.stack(predictions).mean(dim=0)
            uncertainty = torch.stack(predictions).std(dim=0)
        
        # Calculate confidence
        if isinstance(uncertainty, dict):
            confidence = 1.0 - np.mean([v.mean().item() if torch.is_tensor(v) else v 
                                       for v in uncertainty.values()])
        else:
            confidence = 1.0 - uncertainty.mean().item()
        
        # Safety evaluation
        is_safe = confidence >= self.min_confidence
        safety_margin = self.calculate_safety_margin(mean_pred)
        
        # Decision classification
        decision_type = self._classify_decision(confidence)
        
        # Log decision
        self.total_decisions += 1
        self.safety_decisions.append({
            'confidence': confidence,
            'is_safe': is_safe,
            'decision_type': decision_type,
            'timestamp': time.time()
        })
        
        inference_time = time.time() - start_time
        
        result = {
            'answer': mean_pred,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'is_safe_decision': is_safe,
            'safety_margin': safety_margin,
            'decision_type': decision_type,
            'inference_time_ms': inference_time * 1000,
            'spatial_info': self._extract_spatial_info(mean_pred)
        }
        
        # Set model back to eval mode
        self.base_model.eval()
        
        return result
    
    def _aggregate_dict_predictions(self, predictions: List[Dict]) -> Dict:
        """Aggregate dictionary predictions across multiple samples."""
        if not predictions:
            return {}
        
        aggregated = {}
        for key in predictions[0].keys():
            values = [pred[key] for pred in predictions]
            if torch.is_tensor(values[0]):
                aggregated[key] = torch.stack(values).mean(dim=0)
            elif isinstance(values[0], (int, float)):
                aggregated[key] = np.mean(values)
            elif isinstance(values[0], str):
                # For string values, take the most common one or the first one
                aggregated[key] = values[0]  # Simple approach: take first
            else:
                # For other types, try to average if possible, otherwise take first
                try:
                    aggregated[key] = np.mean(values)
                except:
                    aggregated[key] = values[0]
        
        return aggregated
    
    def _calculate_dict_uncertainty(self, predictions: List[Dict]) -> Dict:
        """Calculate uncertainty for dictionary predictions."""
        if not predictions:
            return {}
        
        uncertainty = {}
        for key in predictions[0].keys():
            values = [pred[key] for pred in predictions]
            if torch.is_tensor(values[0]):
                uncertainty[key] = torch.stack(values).std(dim=0)
            elif isinstance(values[0], (int, float)):
                uncertainty[key] = np.std(values)
            elif isinstance(values[0], str):
                # For string values, uncertainty is based on consistency
                unique_values = len(set(values))
                uncertainty[key] = 1.0 - (1.0 / unique_values) if unique_values > 1 else 0.0
            else:
                # For other types, try to calculate std if possible, otherwise set to 0
                try:
                    uncertainty[key] = np.std(values)
                except:
                    uncertainty[key] = 0.0
        
        return uncertainty
    
    def _classify_decision(self, confidence: float) -> str:
        """Classify decision type based on confidence level."""
        if confidence >= self.confidence_thresholds['normal_navigation']:
            return 'normal_navigation'
        elif confidence >= self.confidence_thresholds['conservative_navigation']:
            return 'conservative_navigation'
        elif confidence >= self.confidence_thresholds['human_intervention']:
            return 'human_intervention'
        else:
            return 'emergency_stop'
    
    def calculate_safety_margin(self, spatial_answer: Any) -> float:
        """Calculate safe distance margin for navigation."""
        try:
            # Extract spatial information from answer
            if isinstance(spatial_answer, dict):
                if 'distance' in spatial_answer:
                    distance = float(spatial_answer['distance'])
                    return max(0, distance - self.safety_margin)
                elif 'obstacle_distance' in spatial_answer:
                    distance = float(spatial_answer['obstacle_distance'])
                    return max(0, distance - self.safety_margin)
            
            # For other types, assume minimum safety margin
            return self.safety_margin
            
        except (ValueError, KeyError, TypeError):
            logger.warning("Could not extract distance information, using default safety margin")
            return self.safety_margin
    
    def _extract_spatial_info(self, prediction: Any) -> Dict[str, Any]:
        """Extract spatial information from prediction."""
        spatial_info = {
            'forward_distance': 0.0,
            'lateral_distance': 0.0,
            'vertical_distance': 0.0,
            'obstacle_detected': False,
            'safe_direction': None
        }
        
        if isinstance(prediction, dict):
            # Extract known spatial keys
            for key in ['forward_distance', 'lateral_distance', 'vertical_distance']:
                if key in prediction:
                    spatial_info[key] = float(prediction[key])
            
            if 'obstacle_detected' in prediction:
                spatial_info['obstacle_detected'] = bool(prediction['obstacle_detected'])
            
            if 'safe_direction' in prediction:
                spatial_info['safe_direction'] = prediction['safe_direction']
        
        return spatial_info
    
    def emergency_stop_trigger(self, spatial_decision: Dict[str, Any]) -> Dict[str, str]:
        """Trigger emergency stop if spatial decision is unsafe."""
        if not spatial_decision['is_safe_decision'] or spatial_decision['decision_type'] == 'emergency_stop':
            self.emergency_stops += 1
            
            logger.critical(f"EMERGENCY STOP TRIGGERED! Confidence: {spatial_decision['confidence']:.2f}")
            
            return {
                'action': 'EMERGENCY_STOP',
                'reason': f'Low confidence: {spatial_decision["confidence"]:.2f}',
                'recommendation': f'Switch to {self.safety_config.get("backup_navigation", "manual")} control',
                'safety_margin': spatial_decision['safety_margin'],
                'timestamp': time.time()
            }
        
        return {'action': 'CONTINUE'}
    
    def get_navigation_command(self, spatial_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Generate navigation command based on spatial decision and confidence."""
        decision_type = spatial_decision['decision_type']
        spatial_info = spatial_decision['spatial_info']
        
        # Check for emergency stop first
        emergency_response = self.emergency_stop_trigger(spatial_decision)
        if emergency_response['action'] == 'EMERGENCY_STOP':
            return {
                'linear_velocity': [0.0, 0.0, 0.0],
                'angular_velocity': [0.0, 0.0, 0.0],
                'action': 'EMERGENCY_STOP',
                'emergency_info': emergency_response
            }
        
        # Calculate velocities based on decision type
        max_vel = self.navigation_config['max_velocity']
        
        if decision_type == 'normal_navigation':
            velocity_scale = 1.0
        elif decision_type == 'conservative_navigation':
            velocity_scale = 0.5
        elif decision_type == 'human_intervention':
            velocity_scale = 0.2
        else:
            velocity_scale = 0.0
        
        # Calculate safe velocities
        forward_vel = min(max_vel * velocity_scale, 
                         spatial_info['forward_distance'] * 0.5)
        lateral_vel = spatial_info['lateral_distance'] * 0.3 * velocity_scale
        vertical_vel = spatial_info['vertical_distance'] * 0.2 * velocity_scale
        
        return {
            'linear_velocity': [forward_vel, lateral_vel, vertical_vel],
            'angular_velocity': [0.0, 0.0, 0.0],  # Simple linear motion for now
            'action': 'NAVIGATE',
            'decision_type': decision_type,
            'confidence': spatial_decision['confidence'],
            'safety_margin': spatial_decision['safety_margin']
        }
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get comprehensive safety statistics."""
        if not self.safety_decisions:
            return {"status": "no_data"}
        
        confidences = [d['confidence'] for d in self.safety_decisions]
        decision_types = [d['decision_type'] for d in self.safety_decisions]
        
        # Count decision types
        decision_counts = {}
        for dt in decision_types:
            decision_counts[dt] = decision_counts.get(dt, 0) + 1
        
        safe_decisions = sum(1 for d in self.safety_decisions if d['is_safe'])
        
        return {
            'total_decisions': self.total_decisions,
            'safe_decisions': safe_decisions,
            'safety_rate': safe_decisions / self.total_decisions if self.total_decisions > 0 else 0,
            'emergency_stops': self.emergency_stops,
            'emergency_stop_rate': self.emergency_stops / self.total_decisions if self.total_decisions > 0 else 0,
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'decision_type_distribution': decision_counts,
            'recent_decisions': self.safety_decisions[-10:]  # Last 10 decisions
        }
    
    def validate_spatial_consistency(self, current_decision: Dict, previous_decisions: List[Dict]) -> bool:
        """Validate spatial consistency across multiple frames."""
        if not previous_decisions:
            return True
        
        # Check for dramatic changes in spatial understanding
        current_spatial = current_decision['spatial_info']
        prev_spatial = previous_decisions[-1]['spatial_info']
        
        # Check forward distance consistency (shouldn't change drastically)
        forward_change = abs(current_spatial['forward_distance'] - prev_spatial['forward_distance'])
        
        # Allow for reasonable movement (UAV could be moving)
        max_reasonable_change = 2.0  # meters per frame
        
        return forward_change <= max_reasonable_change
    
    def reset_safety_stats(self):
        """Reset safety statistics for new session."""
        self.safety_decisions = []
        self.emergency_stops = 0
        self.total_decisions = 0
        logger.info("Safety statistics reset")


class SpatialReasoningValidator:
    """Validate spatial reasoning outputs for safety."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.navigation_config = config['navigation_config']
        self.min_obstacle_distance = self.navigation_config['min_obstacle_distance']
    
    def validate_spatial_answer(self, spatial_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate spatial reasoning answer for safety compliance."""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'safety_level': 'SAFE'
        }
        
        # Check obstacle distance
        if spatial_info.get('forward_distance', float('inf')) < self.min_obstacle_distance:
            validation_result['warnings'].append(
                f"Forward obstacle too close: {spatial_info['forward_distance']:.2f}m < {self.min_obstacle_distance}m"
            )
            validation_result['safety_level'] = 'WARNING'
        
        # Check for negative distances (invalid)
        for key in ['forward_distance', 'lateral_distance', 'vertical_distance']:
            if spatial_info.get(key, 0) < 0:
                validation_result['errors'].append(f"Invalid negative distance: {key} = {spatial_info[key]}")
                validation_result['is_valid'] = False
                validation_result['safety_level'] = 'UNSAFE'
        
        # Check for unreasonable distances (sensor range limits)
        max_sensor_range = 50.0  # meters
        for key in ['forward_distance', 'lateral_distance']:
            if spatial_info.get(key, 0) > max_sensor_range:
                validation_result['warnings'].append(
                    f"Distance beyond sensor range: {key} = {spatial_info[key]:.2f}m > {max_sensor_range}m"
                )
        
        return validation_result


def create_safety_test_script():
    """Create a test script for safety-critical spatial VQA."""
    test_script = '''
import sys
import os
sys.path.append('/home/isr-lab3/James/VQASynth-UAV')

import torch
import numpy as np
from vqasynth.safety_spatial_vqa import SafetyCriticalSpatialVQA, SpatialReasoningValidator

class MockSpatialVQAModel:
    """Mock model for testing safety framework."""
    
    def __init__(self):
        self.eval_mode = True
    
    def train(self):
        self.eval_mode = False
    
    def eval(self):
        self.eval_mode = True
    
    def __call__(self, image, question):
        # Simulate varying confidence levels
        confidence_sim = np.random.random()
        
        return {
            'answer': f"Navigate forward {np.random.uniform(1, 10):.1f} meters",
            'forward_distance': np.random.uniform(1, 10),
            'lateral_distance': np.random.uniform(-2, 2),
            'vertical_distance': np.random.uniform(-1, 1),
            'obstacle_detected': np.random.random() > 0.7,
            'confidence_sim': confidence_sim
        }

def test_safety_framework():
    """Test the safety-critical spatial VQA framework."""
    print("Testing Safety-Critical Spatial VQA Framework...")
    
    # Create mock model and safety system
    mock_model = MockSpatialVQAModel()
    safety_vqa = SafetyCriticalSpatialVQA(
        mock_model, 
        '/home/isr-lab3/James/VQASynth-UAV/config/config_realtime_uav.yaml'
    )
    
    validator = SpatialReasoningValidator(
        '/home/isr-lab3/James/VQASynth-UAV/config/config_realtime_uav.yaml'
    )
    
    # Simulate multiple navigation decisions
    image = torch.randn(1, 3, 224, 224)  # Mock image
    question = "What is the safe navigation path ahead?"
    
    print("\\nSimulating navigation decisions...")
    for i in range(10):
        decision = safety_vqa.predict_with_uncertainty(image, question)
        
        print(f"\\nDecision {i+1}:")
        print(f"  Confidence: {decision['confidence']:.3f}")
        print(f"  Decision Type: {decision['decision_type']}")
        print(f"  Safety Margin: {decision['safety_margin']:.2f}m")
        print(f"  Is Safe: {decision['is_safe_decision']}")
        
        # Validate spatial answer
        validation = validator.validate_spatial_answer(decision['spatial_info'])
        print(f"  Validation: {validation['safety_level']}")
        
        # Generate navigation command
        nav_command = safety_vqa.get_navigation_command(decision)
        print(f"  Action: {nav_command['action']}")
        
        if nav_command['action'] == 'EMERGENCY_STOP':
            print(f"  EMERGENCY: {nav_command['emergency_info']['reason']}")
    
    # Print safety statistics
    stats = safety_vqa.get_safety_statistics()
    print("\\n=== SAFETY STATISTICS ===")
    print(f"Total decisions: {stats['total_decisions']}")
    print(f"Safety rate: {stats['safety_rate']:.2%}")
    print(f"Emergency stops: {stats['emergency_stops']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"Decision distribution: {stats['decision_type_distribution']}")

if __name__ == "__main__":
    test_safety_framework()
'''
    
    with open('/home/isr-lab3/James/VQASynth-UAV/test_safety_vqa.py', 'w') as f:
        f.write(test_script)

if __name__ == "__main__":
    create_safety_test_script()

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
    
    print("\nSimulating navigation decisions...")
    for i in range(10):
        decision = safety_vqa.predict_with_uncertainty(image, question)
        
        print(f"\nDecision {i+1}:")
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
    print("\n=== SAFETY STATISTICS ===")
    print(f"Total decisions: {stats['total_decisions']}")
    print(f"Safety rate: {stats['safety_rate']:.2%}")
    print(f"Emergency stops: {stats['emergency_stops']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"Decision distribution: {stats['decision_type_distribution']}")

if __name__ == "__main__":
    test_safety_framework()

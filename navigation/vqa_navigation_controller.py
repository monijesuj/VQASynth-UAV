"""
VQA-to-Control Navigation Controller for autonomous UAV navigation.
Converts spatial VQA decisions into flight control commands.
"""

import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
import yaml
import logging
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlightMode(Enum):
    """Flight modes for UAV navigation."""
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    EMERGENCY_STOP = "emergency_stop"
    CONSERVATIVE = "conservative"
    HOVER = "hover"

@dataclass
class ControlCommand:
    """UAV control command structure."""
    linear_velocity: List[float]  # [x, y, z] in m/s
    angular_velocity: List[float]  # [roll, pitch, yaw] in rad/s
    flight_mode: FlightMode
    timestamp: float
    confidence: float
    safety_margin: float

class VQANavigationController:
    """Convert VQA spatial decisions to UAV control commands."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.navigation_config = config['navigation_config']
        self.safety_config = config['safety_config']
        
        # Navigation parameters
        self.max_velocity = self.navigation_config['max_velocity']
        self.min_obstacle_distance = self.navigation_config['min_obstacle_distance']
        self.planning_horizon = self.navigation_config['planning_horizon']
        self.update_frequency = self.navigation_config['update_frequency']
        
        # Safety parameters
        self.safety_margin = self.safety_config['spatial_safety_margin_meters']
        self.min_confidence = self.safety_config['min_confidence_threshold']
        
        # Control state
        self.current_mode = FlightMode.HOVER
        self.last_command = None
        self.emergency_stop_active = False
        
        # Command history for smoothing
        self.command_history = []
        self.max_history_length = 10
        
        # Velocity limits per flight mode
        self.velocity_limits = {
            FlightMode.AUTONOMOUS: 1.0,
            FlightMode.CONSERVATIVE: 0.5,
            FlightMode.EMERGENCY_STOP: 0.0,
            FlightMode.HOVER: 0.1,
            FlightMode.MANUAL: 1.0
        }
        
        # Control thread for continuous operation
        self.control_thread = None
        self.is_running = False
    
    def process_vqa_decision(self, spatial_decision: Dict[str, Any]) -> ControlCommand:
        """Convert spatial VQA decision to control command."""
        
        # Extract spatial information
        spatial_info = spatial_decision.get('spatial_info', {})
        confidence = spatial_decision.get('confidence', 0.0)
        decision_type = spatial_decision.get('decision_type', 'emergency_stop')
        safety_margin = spatial_decision.get('safety_margin', 0.0)
        
        # Determine flight mode based on decision
        flight_mode = self._determine_flight_mode(decision_type, confidence)
        
        # Calculate base velocities from spatial information
        linear_vel = self._calculate_linear_velocity(spatial_info, flight_mode)
        angular_vel = self._calculate_angular_velocity(spatial_info, flight_mode)
        
        # Apply safety constraints
        linear_vel = self._apply_safety_constraints(linear_vel, spatial_info)
        
        # Apply velocity smoothing
        linear_vel = self._apply_velocity_smoothing(linear_vel)
        
        # Create control command
        command = ControlCommand(
            linear_velocity=linear_vel,
            angular_velocity=angular_vel,
            flight_mode=flight_mode,
            timestamp=time.time(),
            confidence=confidence,
            safety_margin=safety_margin
        )
        
        # Update state
        self.current_mode = flight_mode
        self.last_command = command
        self._add_to_history(command)
        
        return command
    
    def _determine_flight_mode(self, decision_type: str, confidence: float) -> FlightMode:
        """Determine appropriate flight mode based on VQA decision."""
        
        if decision_type == 'emergency_stop' or confidence < 0.3:
            self.emergency_stop_active = True
            return FlightMode.EMERGENCY_STOP
        
        elif decision_type == 'human_intervention':
            return FlightMode.HOVER  # Wait for human input
        
        elif decision_type == 'conservative_navigation' or confidence < 0.8:
            return FlightMode.CONSERVATIVE
        
        elif decision_type == 'normal_navigation' and confidence >= 0.8:
            self.emergency_stop_active = False
            return FlightMode.AUTONOMOUS
        
        else:
            return FlightMode.HOVER
    
    def _calculate_linear_velocity(self, spatial_info: Dict[str, Any], 
                                 flight_mode: FlightMode) -> List[float]:
        """Calculate linear velocity commands from spatial information."""
        
        # Get velocity scale factor based on flight mode
        velocity_scale = self.velocity_limits[flight_mode]
        
        if flight_mode == FlightMode.EMERGENCY_STOP:
            return [0.0, 0.0, 0.0]
        
        # Extract distances
        forward_distance = spatial_info.get('forward_distance', 0.0)
        lateral_distance = spatial_info.get('lateral_distance', 0.0)
        vertical_distance = spatial_info.get('vertical_distance', 0.0)
        
        # Calculate forward velocity (X-axis)
        if forward_distance > self.min_obstacle_distance:
            # Proportional control with safety margin
            forward_vel = min(
                self.max_velocity * velocity_scale,
                (forward_distance - self.safety_margin) * 0.3
            )
            forward_vel = max(0.0, forward_vel)  # No reverse
        else:
            forward_vel = 0.0  # Stop if obstacle too close
        
        # Calculate lateral velocity (Y-axis)
        lateral_vel = np.clip(
            lateral_distance * 0.2 * velocity_scale,
            -self.max_velocity * velocity_scale,
            self.max_velocity * velocity_scale
        )
        
        # Calculate vertical velocity (Z-axis)
        vertical_vel = np.clip(
            vertical_distance * 0.15 * velocity_scale,
            -self.max_velocity * velocity_scale * 0.5,  # Slower vertical movement
            self.max_velocity * velocity_scale * 0.5
        )
        
        return [forward_vel, lateral_vel, vertical_vel]
    
    def _calculate_angular_velocity(self, spatial_info: Dict[str, Any], 
                                  flight_mode: FlightMode) -> List[float]:
        """Calculate angular velocity commands from spatial information."""
        
        if flight_mode == FlightMode.EMERGENCY_STOP:
            return [0.0, 0.0, 0.0]
        
        # Simple yaw control based on lateral distance
        lateral_distance = spatial_info.get('lateral_distance', 0.0)
        
        # Convert lateral distance to yaw rate
        yaw_rate = np.clip(lateral_distance * 0.1, -0.5, 0.5)  # Max 0.5 rad/s
        
        # Keep roll and pitch at zero for stability
        roll_rate = 0.0
        pitch_rate = 0.0
        
        return [roll_rate, pitch_rate, yaw_rate]
    
    def _apply_safety_constraints(self, linear_vel: List[float], 
                                spatial_info: Dict[str, Any]) -> List[float]:
        """Apply safety constraints to velocity commands."""
        
        # Check for obstacles in each direction
        forward_distance = spatial_info.get('forward_distance', float('inf'))
        
        # Emergency braking if obstacle too close
        if forward_distance < self.safety_margin:
            linear_vel[0] = 0.0  # Stop forward motion
            logger.warning(f"Emergency braking: obstacle at {forward_distance:.2f}m")
        
        # Limit velocity based on obstacle distance
        if forward_distance < self.min_obstacle_distance:
            scale_factor = max(0.1, forward_distance / self.min_obstacle_distance)
            linear_vel = [v * scale_factor for v in linear_vel]
        
        # Apply absolute velocity limits
        for i in range(3):
            linear_vel[i] = np.clip(linear_vel[i], -self.max_velocity, self.max_velocity)
        
        return linear_vel
    
    def _apply_velocity_smoothing(self, linear_vel: List[float]) -> List[float]:
        """Apply velocity smoothing to reduce jitter."""
        
        if not self.command_history:
            return linear_vel
        
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing factor
        last_vel = self.command_history[-1].linear_velocity
        
        smoothed_vel = []
        for i in range(3):
            smoothed = alpha * linear_vel[i] + (1 - alpha) * last_vel[i]
            smoothed_vel.append(smoothed)
        
        return smoothed_vel
    
    def _add_to_history(self, command: ControlCommand):
        """Add command to history for smoothing."""
        self.command_history.append(command)
        
        # Limit history length
        if len(self.command_history) > self.max_history_length:
            self.command_history = self.command_history[-self.max_history_length:]
    
    def emergency_stop(self) -> ControlCommand:
        """Execute immediate emergency stop."""
        logger.critical("EMERGENCY STOP ACTIVATED")
        
        self.emergency_stop_active = True
        self.current_mode = FlightMode.EMERGENCY_STOP
        
        command = ControlCommand(
            linear_velocity=[0.0, 0.0, 0.0],
            angular_velocity=[0.0, 0.0, 0.0],
            flight_mode=FlightMode.EMERGENCY_STOP,
            timestamp=time.time(),
            confidence=0.0,
            safety_margin=0.0
        )
        
        self.last_command = command
        return command
    
    def hover_command(self) -> ControlCommand:
        """Generate hover command."""
        command = ControlCommand(
            linear_velocity=[0.0, 0.0, 0.0],
            angular_velocity=[0.0, 0.0, 0.0],
            flight_mode=FlightMode.HOVER,
            timestamp=time.time(),
            confidence=1.0,
            safety_margin=self.safety_margin
        )
        
        return command
    
    def reset_emergency_stop(self):
        """Reset emergency stop state."""
        if self.emergency_stop_active:
            self.emergency_stop_active = False
            self.current_mode = FlightMode.HOVER
            logger.info("Emergency stop reset - switching to HOVER mode")
    
    def get_navigation_status(self) -> Dict[str, Any]:
        """Get current navigation status."""
        return {
            'current_mode': self.current_mode.value,
            'emergency_stop_active': self.emergency_stop_active,
            'last_command_timestamp': self.last_command.timestamp if self.last_command else 0,
            'command_history_length': len(self.command_history),
            'is_running': self.is_running
        }
    
    def validate_command(self, command: ControlCommand) -> Tuple[bool, List[str]]:
        """Validate control command for safety."""
        issues = []
        
        # Check velocity limits
        for i, vel in enumerate(command.linear_velocity):
            if abs(vel) > self.max_velocity:
                issues.append(f"Linear velocity {i} exceeds limit: {vel:.2f} > {self.max_velocity}")
        
        for i, vel in enumerate(command.angular_velocity):
            if abs(vel) > 1.0:  # 1 rad/s angular velocity limit
                issues.append(f"Angular velocity {i} exceeds limit: {vel:.2f} > 1.0")
        
        # Check confidence
        if command.confidence < 0.3 and command.flight_mode == FlightMode.AUTONOMOUS:
            issues.append(f"Low confidence for autonomous mode: {command.confidence:.2f}")
        
        # Check safety margin
        if command.safety_margin < 1.0 and command.linear_velocity[0] > 0:
            issues.append(f"Moving forward with insufficient safety margin: {command.safety_margin:.2f}m")
        
        return len(issues) == 0, issues
    
    def start_autonomous_navigation(self):
        """Start autonomous navigation mode."""
        if not self.emergency_stop_active:
            self.current_mode = FlightMode.AUTONOMOUS
            logger.info("Autonomous navigation started")
        else:
            logger.warning("Cannot start autonomous navigation: emergency stop active")
    
    def stop_autonomous_navigation(self):
        """Stop autonomous navigation and switch to hover."""
        self.current_mode = FlightMode.HOVER
        logger.info("Autonomous navigation stopped - switching to HOVER")


class NavigationController:
    """High-level navigation controller integrating VQA decisions."""
    
    def __init__(self, config_path: str):
        self.vqa_controller = VQANavigationController(config_path)
        self.command_queue = []
        self.max_queue_size = 10
        
    def process_spatial_vqa_result(self, vqa_result: Dict[str, Any]) -> ControlCommand:
        """Process spatial VQA result and generate control command."""
        
        # Extract navigation command from VQA result
        nav_command = vqa_result.get('navigation_command', {})
        
        if nav_command.get('action') == 'EMERGENCY_STOP':
            return self.vqa_controller.emergency_stop()
        
        # Process the spatial decision
        spatial_decision = vqa_result.get('spatial_decision', {})
        command = self.vqa_controller.process_vqa_decision(spatial_decision)
        
        # Validate command
        is_valid, issues = self.vqa_controller.validate_command(command)
        
        if not is_valid:
            logger.warning(f"Invalid command generated: {issues}")
            # Return hover command as fallback
            command = self.vqa_controller.hover_command()
        
        # Add to command queue
        self._add_command_to_queue(command)
        
        return command
    
    def _add_command_to_queue(self, command: ControlCommand):
        """Add command to execution queue."""
        self.command_queue.append(command)
        
        # Limit queue size
        if len(self.command_queue) > self.max_queue_size:
            self.command_queue = self.command_queue[-self.max_queue_size:]
    
    def get_latest_command(self) -> Optional[ControlCommand]:
        """Get the latest control command."""
        return self.command_queue[-1] if self.command_queue else None
    
    def get_controller_statistics(self) -> Dict[str, Any]:
        """Get navigation controller statistics."""
        stats = self.vqa_controller.get_navigation_status()
        stats.update({
            'command_queue_size': len(self.command_queue),
            'total_commands_processed': len(self.command_queue)
        })
        return stats


def create_navigation_test():
    """Create test script for navigation controller."""
    test_script = '''
import sys
import os
sys.path.append('/home/isr-lab3/James/VQASynth-UAV')

import numpy as np
import time
from navigation.vqa_navigation_controller import NavigationController

def test_navigation_controller():
    """Test the VQA navigation controller."""
    print("Testing VQA Navigation Controller...")
    
    controller = NavigationController(
        '/home/isr-lab3/James/VQASynth-UAV/config/config_realtime_uav.yaml'
    )
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Normal Navigation',
            'spatial_decision': {
                'confidence': 0.9,
                'decision_type': 'normal_navigation',
                'safety_margin': 3.0,
                'spatial_info': {
                    'forward_distance': 8.0,
                    'lateral_distance': 0.5,
                    'vertical_distance': 0.0,
                    'obstacle_detected': False
                }
            },
            'navigation_command': {'action': 'NAVIGATE'}
        },
        {
            'name': 'Conservative Navigation',
            'spatial_decision': {
                'confidence': 0.75,
                'decision_type': 'conservative_navigation',
                'safety_margin': 2.5,
                'spatial_info': {
                    'forward_distance': 4.0,
                    'lateral_distance': -1.0,
                    'vertical_distance': 0.5,
                    'obstacle_detected': True
                }
            },
            'navigation_command': {'action': 'NAVIGATE'}
        },
        {
            'name': 'Emergency Stop',
            'spatial_decision': {
                'confidence': 0.2,
                'decision_type': 'emergency_stop',
                'safety_margin': 0.5,
                'spatial_info': {
                    'forward_distance': 1.0,
                    'lateral_distance': 0.0,
                    'vertical_distance': 0.0,
                    'obstacle_detected': True
                }
            },
            'navigation_command': {'action': 'EMERGENCY_STOP'}
        }
    ]
    
    print("\\nTesting different navigation scenarios...")
    
    for scenario in scenarios:
        print(f"\\n--- {scenario['name']} ---")
        
        vqa_result = {
            'spatial_decision': scenario['spatial_decision'],
            'navigation_command': scenario['navigation_command']
        }
        
        command = controller.process_spatial_vqa_result(vqa_result)
        
        print(f"Flight Mode: {command.flight_mode.value}")
        print(f"Linear Velocity: [{command.linear_velocity[0]:.2f}, {command.linear_velocity[1]:.2f}, {command.linear_velocity[2]:.2f}] m/s")
        print(f"Angular Velocity: [{command.angular_velocity[0]:.2f}, {command.angular_velocity[1]:.2f}, {command.angular_velocity[2]:.2f}] rad/s")
        print(f"Confidence: {command.confidence:.2f}")
        print(f"Safety Margin: {command.safety_margin:.2f}m")
    
    # Test controller statistics
    stats = controller.get_controller_statistics()
    print(f"\\n=== CONTROLLER STATISTICS ===")
    print(f"Current mode: {stats['current_mode']}")
    print(f"Emergency stop active: {stats['emergency_stop_active']}")
    print(f"Commands processed: {stats['total_commands_processed']}")
    
    print("\\nNavigation controller test completed.")

if __name__ == "__main__":
    test_navigation_controller()
'''
    
    with open('/home/isr-lab3/James/VQASynth-UAV/test_navigation_controller.py', 'w') as f:
        f.write(test_script)

if __name__ == "__main__":
    create_navigation_test()

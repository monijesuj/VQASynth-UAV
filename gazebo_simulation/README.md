# Gazebo Spatial Navigation Simulation 🚁

**Complete simulation environment for testing spatial reasoning-based drone navigation using ROS Noetic + ArduPilot SITL + Gazebo**

## 📋 Overview

This simulation environment integrates:
- **Gazebo**: Realistic 3D physics simulation
- **ArduPilot SITL**: Flight controller simulation
- **ROS Noetic**: Robot Operating System framework  
- **SpaceOm/SpaceThinker**: Spatial reasoning VLMs
- **MAVROS**: ROS-ArduPilot communication bridge

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gazebo World  │    │  ArduPilot SITL │    │ Spatial VLM     │
│   • Obstacles   ├────┤  • Flight Ctrl  ├────┤ • SpaceOm       │
│   • Sensors     │    │  • Physics      │    │ • SpaceThinker  │
│   • Physics     │    │  • Navigation   │    │ • Analysis      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                        ▲                        ▲
         │                        │                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ROS Noetic    │    │     MAVROS      │    │  Navigation     │
│   • Topics      ├────┤  • Bridge       ├────┤  Controller     │
│   • Services    │    │  • Messages     │    │  • Missions     │
│   • Transforms  │    │  • Commands     │    │  • Safety       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Run setup script (installs dependencies)
cd gazebo_simulation
./setup_simulation.sh

# Source environment
source ~/.bashrc
```

### 2. Launch Simulation
```bash
# Launch complete simulation stack
./launch_spatial_simulation.sh
```

This opens multiple terminals:
- **Gazebo**: 3D simulation environment
- **ArduPilot SITL**: Flight controller
- **MAVROS**: ROS communication bridge
- **Spatial Reasoning**: AI navigation node
- **Navigation Controller**: Mission execution

### 3. Test Navigation
```bash
# In Navigation Controller terminal:
🎮 Command: mission corridor_navigation
🎮 Command: goto 5 5 2  
🎮 Command: status
```

## 📁 File Structure

```
gazebo_simulation/
├── worlds/
│   └── spatial_navigation_world.world    # Main Gazebo world
├── models/
│   └── spatial_reasoning_drone/          # Drone model with sensors
│       ├── model.config
│       └── model.sdf
├── launch/
│   └── spatial_navigation.launch        # ROS launch file
├── config/
│   └── navigation_config.yaml          # Configuration parameters  
├── spatial_reasoning_node.py           # Core spatial reasoning node
├── navigation_controller.py            # Mission controller
├── test_navigation_scenarios.py        # Test scenarios
├── setup_simulation.sh                # Environment setup
├── launch_spatial_simulation.sh       # Launch script
└── README.md                          # This file
```

## 🌍 Simulation Environment

### World Features
- **Warehouse building** with walls and roof
- **Navigation obstacles**: Boxes, pillars, corridors  
- **Outdoor elements**: Trees, power lines
- **Realistic physics** and lighting
- **Multi-level navigation** challenges

### Drone Capabilities  
- **RGB Camera**: 1920x1080 front-facing for spatial analysis
- **Depth Camera**: 640x480 for distance measurement
- **Down Camera**: Ground distance estimation
- **IMU**: Attitude and acceleration sensing
- **GPS**: Position reference

## 🧠 Spatial Reasoning Integration

### Core Features
- **Real-time spatial analysis** using SpaceOm/SpaceThinker
- **Multi-query reasoning**: Distance, obstacles, clearance
- **Safety monitoring**: Automatic collision avoidance  
- **Adaptive navigation**: Path planning based on AI insights

### Spatial Queries
```python
queries = [
    "What obstacles are directly ahead within 5 meters?",
    "Is there enough clearance to fly forward safely?", 
    "What is the distance to the nearest obstacle?",
    "Are there any overhead obstacles like wires?",
    "What is the safest direction to navigate?"
]
```

## 🎯 Test Scenarios

### 1. Corridor Navigation
Navigate through 6-meter wide corridor between walls
```bash
python3 test_navigation_scenarios.py corridor_navigation
```

### 2. Obstacle Avoidance  
Avoid boxes, pillars, and other static obstacles
```bash
python3 test_navigation_scenarios.py obstacle_avoidance
```

### 3. Height Variation
Navigate at different altitudes (1m to 5m)
```bash
python3 test_navigation_scenarios.py height_variation
```

### 4. Emergency Stop
Test safety system response times
```bash
python3 test_navigation_scenarios.py emergency_stop
```

### 5. Spatial Accuracy
Validate AI spatial reasoning accuracy
```bash
python3 test_navigation_scenarios.py spatial_reasoning_accuracy
```

## 🎮 Interactive Control

### Navigation Commands
```bash
# Go to specific waypoint
🎮 Command: goto 5 5 2

# Execute pre-defined mission
🎮 Command: mission corridor_navigation

# Check system status  
🎮 Command: status

# View spatial analysis
🎮 Command: analysis

# Emergency abort
🎮 Command: abort
```

### ROS Topics
```bash
# Camera feeds
rostopic echo /drone/rgb_camera/image_raw
rostopic echo /drone/depth_camera/depth/image_raw

# Spatial analysis results
rostopic echo /spatial_reasoning/analysis
rostopic echo /spatial_reasoning/safety_status

# Navigation commands
rostopic pub /spatial_navigation/goto_waypoint geometry_msgs/Point "x: 5.0 y: 5.0 z: 2.0"
```

### ROS Services  
```bash
# Navigate to waypoint
rosservice call /spatial_navigation/goto_waypoint "x: 5.0 y: 5.0 z: 2.0"

# Emergency stop
rosservice call /spatial_navigation/emergency_stop
```

## 🛠️ Configuration

### Navigation Parameters (`config/navigation_config.yaml`)
```yaml
safety:
  min_obstacle_distance: 2.0  # meters
  max_velocity: 2.0          # m/s
  emergency_stop_distance: 1.0

navigation:
  waypoint_tolerance: 0.5    # meters
  default_altitude: 2.0      # meters
  
spatial_reasoning:
  model_type: "SpaceOm"      # or "SpaceThinker"
  confidence_threshold: 0.7
```

## 📊 Performance Monitoring

### Metrics Tracked
- **Navigation accuracy**: Waypoint reaching success rate
- **Collision avoidance**: Zero-collision missions
- **Spatial reasoning**: Query response accuracy and speed
- **Safety response**: Emergency stop reaction time

### Test Report Generation
```bash
# Run full test suite
python3 test_navigation_scenarios.py

# Generates: spatial_navigation_test_report.json
```

## 🔧 Troubleshooting

### Common Issues

#### 1. ArduPilot Connection Failed
```bash
# Check ArduPilot SITL is running
ps aux | grep sim_vehicle

# Restart ArduPilot
cd ~/ardupilot
sim_vehicle.py -v ArduCopter -f gazebo-iris --console
```

#### 2. Gazebo Plugin Not Found
```bash
# Check plugin path
echo $GAZEBO_PLUGIN_PATH

# Add to ~/.bashrc
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:$HOME/gazebo_ardupilot_plugin/build
```

#### 3. MAVROS Connection Issues
```bash
# Check MAVROS status
rostopic echo /mavros/state

# Restart MAVROS
roslaunch mavros apm.launch fcu_url:=udp://:14550@127.0.0.1:14551
```

#### 4. Spatial Model Loading Failed
```bash
# Check model files exist
ls -la ../SpaceOm ../SpaceThinker-Qwen2.5VL-3B

# Check GPU memory
nvidia-smi

# Test models separately  
cd .. && python simple_spatial_tester.py
```

### Debug Commands
```bash
# Check ROS nodes
rosnode list

# Monitor topics
rostopic list
rostopic hz /drone/rgb_camera/image_raw

# Check transforms
rosrun tf2_tools view_frames.py

# Gazebo verbose output
gazebo --verbose worlds/spatial_navigation_world.world
```

## 🔮 Future Enhancements

### Planned Features
- **Multi-drone swarms**: Coordinated navigation
- **Dynamic obstacles**: Moving targets and threats  
- **Weather simulation**: Wind, rain effects
- **Advanced missions**: Search and rescue scenarios
- **Real hardware**: Deploy to physical drones

### Integration Opportunities
- **ROS2 migration**: Modern robotics framework
- **Isaac Sim**: NVIDIA's photorealistic simulation
- **PX4 support**: Alternative flight stack
- **Cloud deployment**: Scalable simulation infrastructure

## 📚 Dependencies

### Required Packages
```bash
# ROS Noetic
sudo apt install ros-noetic-desktop-full
sudo apt install ros-noetic-mavros ros-noetic-mavros-extras

# ArduPilot SITL  
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot && Tools/environment_install/install-prereqs-ubuntu.sh -y

# Gazebo ArduPilot Plugin
git clone https://github.com/khancyr/ardupilot_gazebo.git

# Python Dependencies
pip3 install torch transformers opencv-python pillow
```

### System Requirements
- **OS**: Ubuntu 20.04 LTS
- **GPU**: CUDA-capable (8GB+ VRAM recommended)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space

## 📄 License & Citations

This simulation environment builds upon:
- **Gazebo**: Open-source robotics simulator
- **ArduPilot**: Open-source autopilot software  
- **ROS**: Robot Operating System
- **SpaceOm/SpaceThinker**: Spatial reasoning models

### Citation
```bibtex
@software{spatial_navigation_sim,
  title={Gazebo Spatial Navigation Simulation},
  author={VQASynth Navigation Team},
  year={2025},
  url={https://github.com/your-repo/spatial-navigation-sim}
}
```

---

**🚁 Ready to test spatial reasoning navigation? Start with `./setup_simulation.sh`!**
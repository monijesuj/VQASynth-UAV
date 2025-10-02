#!/bin/bash

# Launch script for complete spatial navigation simulation
# Combines Gazebo + ArduPilot SITL + ROS Noetic + Spatial Reasoning

echo "🚁 Launching Spatial Navigation Simulation"
echo "=========================================="

# Check if setup was run
if [ ! -d "$HOME/ardupilot" ]; then
    echo "❌ ArduPilot not found. Please run setup_simulation.sh first"
    exit 1
fi

# Source ROS environment
source /opt/ros/noetic/setup.bash
if [ -f "$HOME/catkin_ws/devel/setup.bash" ]; then
    source $HOME/catkin_ws/devel/setup.bash
fi

# Set Gazebo paths
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(pwd)/models
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:$(pwd)/worlds
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:$HOME/gazebo_ardupilot_plugin/build

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f gazebo || true
pkill -f ardupilot || true
pkill -f sim_vehicle || true
pkill -f mavproxy || true
sleep 2

# Launch Gazebo with our world
echo "🌍 Starting Gazebo simulation..."
gnome-terminal --tab --title="Gazebo" -- bash -c "
    cd $(pwd)
    export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(pwd)/models
    export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:$(pwd)/worlds
    export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:$HOME/gazebo_ardupilot_plugin/build
    roslaunch gazebo_ros empty_world.launch world_name:=$(pwd)/worlds/spatial_navigation_world.world gui:=true paused:=false verbose:=true
    exec bash
"

# Wait for Gazebo to start
echo "⏳ Waiting for Gazebo to initialize..."
sleep 5

# Start ArduPilot SITL
echo "🛸 Starting ArduPilot SITL..."
gnome-terminal --tab --title="ArduPilot SITL" -- bash -c "
    cd $HOME/ardupilot
    export PATH=\$PATH:$HOME/ardupilot/Tools/autotest:$HOME/ardupilot/build/sitl/bin
    sim_vehicle.py -v ArduCopter -f gazebo-iris --map --console --out=udp:127.0.0.1:14550
    exec bash
"

# Wait for ArduPilot to start
echo "⏳ Waiting for ArduPilot to initialize..."
sleep 10

# Spawn drone in Gazebo
echo "🚁 Spawning drone in Gazebo..."
gnome-terminal --tab --title="Drone Spawn" -- bash -c "
    source /opt/ros/noetic/setup.bash
    rosrun gazebo_ros spawn_model -file $(pwd)/models/spatial_reasoning_drone/model.sdf -sdf -model spatial_drone -x 0 -y 0 -z 1
    exec bash
"

# Start MAVROS
echo "📡 Starting MAVROS connection..."
gnome-terminal --tab --title="MAVROS" -- bash -c "
    source /opt/ros/noetic/setup.bash
    roslaunch mavros apm.launch fcu_url:=udp://:14550@127.0.0.1:14551
    exec bash
"

# Wait for MAVROS to connect
sleep 5

# Start spatial reasoning node
echo "🧠 Starting spatial reasoning node..."
gnome-terminal --tab --title="Spatial Reasoning" -- bash -c "
    source /opt/ros/noetic/setup.bash
    cd $(pwd)
    python3 spatial_reasoning_node.py
    exec bash
"

# Start navigation controller
echo "🎮 Starting navigation controller..."
gnome-terminal --tab --title="Navigation Controller" -- bash -c "
    source /opt/ros/noetic/setup.bash
    cd $(pwd)
    python3 navigation_controller.py
    exec bash
"

echo ""
echo "✅ Simulation launched successfully!"
echo ""
echo "📋 Control Instructions:"
echo "   • Gazebo GUI: Visualize the simulation world"
echo "   • ArduPilot Console: Manual flight control"  
echo "   • MAVROS: ROS-ArduPilot bridge"
echo "   • Spatial Reasoning: AI-powered navigation decisions"
echo "   • Navigation Controller: High-level mission commands"
echo ""
echo "🎯 Test Commands (in Navigation Controller terminal):"
echo "   • rosservice call /spatial_navigation/goto_waypoint '{x: 5, y: 5, z: 2}'"
echo "   • rostopic echo /spatial_reasoning/analysis"
echo "   • rostopic echo /drone/rgb_camera/image_raw"
echo ""
echo "🛑 To stop simulation: Ctrl+C in all terminals"
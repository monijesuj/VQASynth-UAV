#!/bin/bash

# Setup script for Gazebo + ArduPilot + ROS Noetic Spatial Navigation Simulation

echo "🚁 Setting up Spatial Navigation Simulation Environment"
echo "=================================================="

# Check if running on Ubuntu 20.04
if ! lsb_release -a 2>/dev/null | grep -q "20.04"; then
    echo "⚠️  Warning: This script is designed for Ubuntu 20.04"
fi

# Check ROS Noetic installation
if ! source /opt/ros/noetic/setup.bash 2>/dev/null; then
    echo "❌ ROS Noetic not found. Installing..."
    sudo apt update
    sudo apt install -y ros-noetic-desktop-full
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
fi

# Install required ROS packages
echo "📦 Installing ROS packages..."
sudo apt install -y \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-mavros \
    ros-noetic-mavros-extras \
    ros-noetic-geographic-msgs \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-camera-calibration-parsers \
    ros-noetic-camera-info-manager \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-tf2-sensor-msgs

# Install GeographicLib datasets for MAVROS
echo "🌍 Installing GeographicLib datasets..."
sudo /opt/ros/noetic/lib/mavros/install_geographiclib_datasets.sh

# Check ArduPilot installation
if [ ! -d "$HOME/ardupilot" ]; then
    echo "🛸 Installing ArduPilot..."
    cd $HOME
    git clone https://github.com/ArduPilot/ardupilot.git
    cd ardupilot
    git checkout Copter-4.3
    git submodule update --init --recursive
    
    # Install dependencies
    Tools/environment_install/install-prereqs-ubuntu.sh -y
    
    # Build SITL
    ./waf configure --board sitl
    ./waf copter
    
    echo "export PATH=\$PATH:\$HOME/ardupilot/Tools/autotest" >> ~/.bashrc
    echo "export PATH=\$PATH:\$HOME/ardupilot/build/sitl/bin" >> ~/.bashrc
fi

# Check Gazebo ArduPilot Plugin
if [ ! -d "$HOME/gazebo_ardupilot_plugin" ]; then
    echo "🔌 Installing Gazebo ArduPilot Plugin..."
    cd $HOME
    git clone https://github.com/khancyr/ardupilot_gazebo.git gazebo_ardupilot_plugin
    cd gazebo_ardupilot_plugin
    mkdir build
    cd build
    # Fix CMake version compatibility
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    make -j4
    
    # Add to Gazebo plugin path
    echo "export GAZEBO_PLUGIN_PATH=\$GAZEBO_PLUGIN_PATH:\$HOME/gazebo_ardupilot_plugin/build" >> ~/.bashrc
fi

# Set up Gazebo model paths
GAZEBO_MODELS_PATH="$HOME/.gazebo/models"
mkdir -p $GAZEBO_MODELS_PATH

# Copy our custom models
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp -r "$CURRENT_DIR/../models/"* "$GAZEBO_MODELS_PATH/"

echo "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:$CURRENT_DIR/../models" >> ~/.bashrc
echo "export GAZEBO_RESOURCE_PATH=\$GAZEBO_RESOURCE_PATH:$CURRENT_DIR/../worlds" >> ~/.bashrc

# Create workspace if it doesn't exist
if [ ! -d "$HOME/catkin_ws" ]; then
    echo "🏗️  Creating catkin workspace..."
    mkdir -p $HOME/catkin_ws/src
    cd $HOME/catkin_ws
    catkin_make
    echo "source $HOME/catkin_ws/devel/setup.bash" >> ~/.bashrc
fi

# Install Python dependencies for spatial reasoning
echo "🐍 Installing Python dependencies..."
pip3 install --user \
    torch \
    torchvision \
    transformers \
    opencv-python \
    pillow \
    numpy \
    matplotlib

# Note: rospy, sensor-msgs, geometry-msgs, cv-bridge are ROS packages, not pip packages
echo "📦 ROS Python packages (rospy, sensor-msgs, geometry-msgs, cv-bridge) are included with ROS installation"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the simulation:"
echo "   1. Source the environment: source ~/.bashrc"
echo "   2. Launch simulation: ./launch_spatial_simulation.sh"
echo ""
echo "📋 Manual steps if needed:"
echo "   - Restart terminal or run: source ~/.bashrc"
echo "   - Test ArduPilot: sim_vehicle.py -v ArduCopter"
echo "   - Test Gazebo: gazebo --verbose"
echo ""
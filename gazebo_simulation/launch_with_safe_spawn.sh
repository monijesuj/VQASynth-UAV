#!/bin/bash

# Alternative launch script with safe drone spawn position
# This shows how to manually control the spawn position if needed

echo "🚁 Launching Spatial Navigation Simulation with Safe Spawn"
echo "======================================================="

# Set safe spawn position (away from warehouse)
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(pwd)/models
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:$(pwd)/worlds

# Launch Gazebo with our world (without drone included in world file)
echo "🌍 Starting Gazebo simulation..."
gazebo --verbose worlds/spatial_navigation_world_no_drone.world &
GAZEBO_PID=$!

# Wait for Gazebo to start
sleep 5

# Spawn drone at safe position using gazebo service
echo "🚁 Spawning drone at safe position (-25, -25, 1.0)..."
rosrun gazebo_ros spawn_model -file models/spatial_reasoning_drone/model.sdf -sdf -model spatial_reasoning_drone -x -25 -y -25 -z 1.0

echo ""
echo "✅ Simulation launched successfully!"
echo ""
echo "🚀 Next steps:"
echo "   1. In another terminal: cd ~/ardupilot/ArduCopter"
echo "   2. Run: ../Tools/autotest/sim_vehicle.py -f gazebo-iris --console --map"
echo "   3. In SITL console, set home position: param set SIM_OPOS_LAT=-35.363261 SIM_OPOS_LNG=149.165230"
echo "   4. Launch spatial reasoning: roslaunch spatial_reasoning_sim spatial_reasoning.launch"
echo ""
echo "📍 Drone spawn position: (-25, -25, 1.0)"
echo "📍 Warehouse center: (0, 0, 0)"
echo "📍 Safe distance: ~35 meters from warehouse"
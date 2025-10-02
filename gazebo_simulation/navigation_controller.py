#!/usr/bin/env python3
"""
Navigation Controller for Spatial Reasoning Drone
High-level mission planning and execution
"""

import rospy
import time
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import String, Bool
import json

class NavigationController:
    def __init__(self):
        rospy.init_node('navigation_controller', anonymous=True)
        rospy.loginfo("🎮 Starting Navigation Controller")
        
        # Navigation state
        self.current_mission = None
        self.mission_active = False
        self.safety_status = True
        self.current_analysis = ""
        
        # Subscribers
        self.safety_sub = rospy.Subscriber('/spatial_reasoning/safety_status', 
                                         Bool, self.safety_callback)
        self.analysis_sub = rospy.Subscriber('/spatial_reasoning/analysis', 
                                           String, self.analysis_callback)
        
        # Publishers  
        self.waypoint_pub = rospy.Publisher('/spatial_navigation/goto_waypoint', 
                                          Point, queue_size=1)
        
        rospy.loginfo("🚀 Navigation Controller ready!")
        
    def safety_callback(self, msg):
        """Update safety status from spatial reasoning"""
        self.safety_status = msg.data
        if not self.safety_status:
            rospy.logwarn("⚠️ Safety alert from spatial reasoning system")
    
    def analysis_callback(self, msg):
        """Update spatial analysis results"""
        self.current_analysis = msg.data
        rospy.logdebug("📊 Spatial analysis updated")
    
    def execute_test_mission(self, mission_type="corridor_navigation"):
        """Execute predefined test missions"""
        rospy.loginfo(f"🎯 Starting {mission_type} mission")
        
        if mission_type == "corridor_navigation":
            waypoints = [
                Point(x=0, y=0, z=2),    # Takeoff position
                Point(x=0, y=3, z=2),    # Enter corridor
                Point(x=0, y=6, z=2),    # Middle of corridor  
                Point(x=0, y=9, z=2),    # Exit corridor
                Point(x=0, y=0, z=2)     # Return home
            ]
        
        elif mission_type == "obstacle_avoidance":
            waypoints = [
                Point(x=0, y=0, z=2),    # Start
                Point(x=-5, y=0, z=2),   # Approach box obstacle
                Point(x=-5, y=3, z=2),   # Avoid around obstacle
                Point(x=5, y=3, z=2),    # Approach pillar
                Point(x=5, y=7, z=2),    # Avoid pillar
                Point(x=0, y=0, z=2)     # Return home
            ]
        
        elif mission_type == "height_variation":
            waypoints = [
                Point(x=0, y=0, z=1),    # Low altitude
                Point(x=5, y=0, z=3),    # Medium altitude
                Point(x=10, y=0, z=5),   # High altitude  
                Point(x=5, y=0, z=2),    # Return to medium
                Point(x=0, y=0, z=2)     # Land
            ]
        
        else:
            rospy.logerr(f"Unknown mission type: {mission_type}")
            return
        
        self.execute_waypoint_sequence(waypoints)
    
    def execute_waypoint_sequence(self, waypoints, wait_time=10):
        """Execute a sequence of waypoints with spatial reasoning"""
        self.mission_active = True
        
        for i, waypoint in enumerate(waypoints):
            if not self.mission_active:
                rospy.logwarn("Mission aborted")
                break
                
            rospy.loginfo(f"🎯 Waypoint {i+1}/{len(waypoints)}: ({waypoint.x:.1f}, {waypoint.y:.1f}, {waypoint.z:.1f})")
            
            # Send waypoint command (you would use the service call in real implementation)
            self.waypoint_pub.publish(waypoint)
            
            # Wait for waypoint completion with safety monitoring
            start_time = time.time()
            while time.time() - start_time < wait_time:
                if not self.safety_status:
                    rospy.logwarn("⚠️ Safety violation detected - pausing mission")
                    self.wait_for_safety_clearance()
                
                rospy.sleep(1)
                
            rospy.loginfo(f"✅ Waypoint {i+1} completed")
            
        self.mission_active = False
        rospy.loginfo("🏁 Mission completed")
    
    def wait_for_safety_clearance(self):
        """Wait until safety status is clear"""
        rospy.loginfo("⏳ Waiting for safety clearance...")
        while not self.safety_status and not rospy.is_shutdown():
            rospy.sleep(1)
        rospy.loginfo("✅ Safety clearance received - resuming mission")
    
    def emergency_abort(self):
        """Abort current mission immediately"""
        rospy.logwarn("🛑 Emergency mission abort")
        self.mission_active = False
        # Send emergency stop command
        # (implement service call to spatial_reasoning_node)
    
    def interactive_control(self):
        """Interactive control mode for manual testing"""
        rospy.loginfo("🕹️ Entering interactive control mode")
        rospy.loginfo("Commands:")
        rospy.loginfo("  'goto x y z' - Navigate to waypoint")
        rospy.loginfo("  'mission <type>' - Execute test mission")
        rospy.loginfo("  'status' - Show current status")
        rospy.loginfo("  'analysis' - Show spatial analysis")
        rospy.loginfo("  'abort' - Emergency abort")
        rospy.loginfo("  'quit' - Exit")
        
        while not rospy.is_shutdown():
            try:
                cmd = input("\n🎮 Command: ").strip().lower()
                
                if cmd == 'quit':
                    break
                elif cmd == 'status':
                    rospy.loginfo(f"Mission active: {self.mission_active}")
                    rospy.loginfo(f"Safety status: {self.safety_status}")
                elif cmd == 'analysis':
                    rospy.loginfo("📊 Current spatial analysis:")
                    print(self.current_analysis)
                elif cmd == 'abort':
                    self.emergency_abort()
                elif cmd.startswith('goto'):
                    parts = cmd.split()
                    if len(parts) == 4:
                        x, y, z = map(float, parts[1:4])
                        waypoint = Point(x=x, y=y, z=z)
                        rospy.loginfo(f"Going to waypoint: ({x}, {y}, {z})")
                        self.waypoint_pub.publish(waypoint)
                    else:
                        rospy.logwarn("Usage: goto x y z")
                elif cmd.startswith('mission'):
                    parts = cmd.split()
                    if len(parts) == 2:
                        mission_type = parts[1]
                        self.execute_test_mission(mission_type)
                    else:
                        rospy.loginfo("Available missions: corridor_navigation, obstacle_avoidance, height_variation")
                else:
                    rospy.logwarn("Unknown command")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                rospy.logerr(f"Error in interactive control: {e}")
    
if __name__ == '__main__':
    try:
        controller = NavigationController()
        
        # Check if we should run a specific mission from command line
        import sys
        if len(sys.argv) > 1:
            mission_type = sys.argv[1]
            rospy.loginfo(f"Running automated mission: {mission_type}")
            controller.execute_test_mission(mission_type)
        else:
            # Enter interactive mode
            controller.interactive_control()
            
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation controller shutdown")
    except Exception as e:
        rospy.logerr(f"Fatal error in navigation controller: {e}")
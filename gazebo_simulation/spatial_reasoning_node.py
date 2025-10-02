#!/usr/bin/env python3
"""
ROS Noetic Spatial Reasoning Node for Drone Navigation
Integrates SpaceOm/SpaceThinker VLMs with Gazebo simulation
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, Imu
from geometry_msgs.msg import Twist, Point, PoseStamped
from std_msgs.msg import String, Bool
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs

import sys
import os
sys.path.append('/home/isr-lab3/James/VQASynth-UAV')

from simple_spatial_tester import SimpleSpatialTester
import torch
from PIL import Image as PILImage

class SpatialReasoningNavigator:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('spatial_reasoning_navigator', anonymous=True)
        rospy.loginfo("🧠 Starting Spatial Reasoning Navigator")
        
        # Initialize spatial reasoning model
        self.spatial_tester = SimpleSpatialTester()
        rospy.loginfo("Loading spatial reasoning models...")
        if self.spatial_tester.load_model():
            rospy.loginfo("✅ Spatial reasoning models loaded successfully")
        else:
            rospy.logerr("❌ Failed to load spatial reasoning models")
            return
        
        # ROS components
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Current sensor data
        self.current_image = None
        self.current_depth = None
        self.current_pose = None
        self.current_state = None
        
        # Navigation parameters
        self.safe_distance = 2.0  # meters
        self.max_velocity = 1.0   # m/s
        self.navigation_active = False
        self.current_waypoint = None
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', 
                                         Twist, queue_size=1)
        self.spatial_analysis_pub = rospy.Publisher('/spatial_reasoning/analysis', 
                                                  String, queue_size=1)
        self.safety_status_pub = rospy.Publisher('/spatial_reasoning/safety_status', 
                                               Bool, queue_size=1)
        self.setpoint_pub = rospy.Publisher('/mavros/setpoint_position/local', 
                                          PoseStamped, queue_size=1)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/drone/rgb_camera/image_raw', 
                                        Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/drone/depth_camera/depth/image_raw', 
                                        Image, self.depth_callback)
        self.pose_sub = rospy.Subscriber('/mavros/local_position/pose', 
                                       PoseStamped, self.pose_callback)
        self.state_sub = rospy.Subscriber('/mavros/state', 
                                        State, self.state_callback)
        
        # Services
        self.waypoint_srv = rospy.Service('/spatial_navigation/goto_waypoint', 
                                        self.goto_waypoint_callback)
        self.emergency_stop_srv = rospy.Service('/spatial_navigation/emergency_stop', 
                                              self.emergency_stop_callback)
        
        # MAVROS services
        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        
        # Main control loop
        self.rate = rospy.Rate(10)  # 10 Hz
        rospy.loginfo("🚁 Spatial Reasoning Navigator ready!")
        
    def image_callback(self, msg):
        """Process RGB camera feed for spatial reasoning"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Trigger spatial analysis if navigation is active
            if self.navigation_active and self.current_waypoint:
                self.perform_spatial_analysis()
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def depth_callback(self, msg):
        """Process depth camera data"""
        try:
            self.current_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            rospy.logerr(f"Error processing depth: {e}")
    
    def pose_callback(self, msg):
        """Update current drone pose"""
        self.current_pose = msg
    
    def state_callback(self, msg):
        """Update MAVROS state"""
        self.current_state = msg
    
    def perform_spatial_analysis(self):
        """Perform spatial reasoning analysis for navigation"""
        if self.current_image is None:
            return
            
        try:
            # Convert numpy array to PIL Image
            pil_image = PILImage.fromarray(self.current_image)
            
            # Spatial reasoning queries for navigation
            queries = [
                "What obstacles are directly ahead within 5 meters?",
                "Is there enough clearance to fly forward safely?", 
                "What is the distance to the nearest obstacle?",
                "Are there any overhead obstacles like wires or branches?",
                "What is the safest direction to navigate?"
            ]
            
            analyses = []
            safety_clear = True
            
            for query in queries:
                try:
                    response = self.spatial_tester.ask_spatial_question(
                        pil_image, query
                    )
                    analyses.append(f"Q: {query}\nA: {response}\n")
                    
                    # Check for safety keywords
                    if any(keyword in response.lower() for keyword in 
                          ['obstacle', 'collision', 'dangerous', 'blocked', 'too close']):
                        safety_clear = False
                        
                except Exception as e:
                    rospy.logwarn(f"Error in spatial query: {e}")
                    analyses.append(f"Q: {query}\nA: Error - {e}\n")
            
            # Publish spatial analysis
            analysis_msg = String()
            analysis_msg.data = "\n".join(analyses)
            self.spatial_analysis_pub.publish(analysis_msg)
            
            # Publish safety status
            safety_msg = Bool()
            safety_msg.data = safety_clear
            self.safety_status_pub.publish(safety_msg)
            
            # Navigation decision based on analysis
            if not safety_clear:
                rospy.logwarn("⚠️ Spatial analysis detected obstacles - stopping")
                self.emergency_stop()
            else:
                rospy.loginfo("✅ Path clear - continuing navigation")
                
        except Exception as e:
            rospy.logerr(f"Error in spatial analysis: {e}")
    
    def goto_waypoint_callback(self, req):
        """Service callback for waypoint navigation"""
        rospy.loginfo(f"🎯 New waypoint: ({req.x}, {req.y}, {req.z})")
        
        self.current_waypoint = Point()
        self.current_waypoint.x = req.x
        self.current_waypoint.y = req.y  
        self.current_waypoint.z = req.z
        
        # Enable navigation
        self.navigation_active = True
        
        # Send waypoint to MAVROS
        setpoint = PoseStamped()
        setpoint.header.stamp = rospy.Time.now()
        setpoint.header.frame_id = "map"
        setpoint.pose.position = self.current_waypoint
        setpoint.pose.orientation.w = 1.0
        
        self.setpoint_pub.publish(setpoint)
        
        return {"success": True, "message": "Waypoint set successfully"}
    
    def emergency_stop_callback(self, req):
        """Service callback for emergency stop"""
        rospy.logwarn("🛑 Emergency stop activated")
        self.emergency_stop()
        return {"success": True, "message": "Emergency stop executed"}
    
    def emergency_stop(self):
        """Execute emergency stop procedure"""
        self.navigation_active = False
        self.current_waypoint = None
        
        # Send zero velocity command
        stop_cmd = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_cmd)
            rospy.sleep(0.1)
    
    def arm_and_takeoff(self, altitude=2.0):
        """Arm drone and takeoff to specified altitude"""
        rospy.loginfo(f"🚁 Arming and taking off to {altitude}m")
        
        # Wait for connection
        while not rospy.is_shutdown() and not self.current_state.connected:
            rospy.loginfo("Waiting for MAVROS connection...")
            self.rate.sleep()
        
        # Set GUIDED mode
        if self.current_state.mode != "GUIDED":
            rospy.loginfo("Setting GUIDED mode...")
            result = self.set_mode_client(custom_mode="GUIDED")
            if result.mode_sent:
                rospy.loginfo("✅ GUIDED mode set")
            else:
                rospy.logerr("❌ Failed to set GUIDED mode")
                return False
        
        # Arm the drone
        if not self.current_state.armed:
            rospy.loginfo("Arming drone...")
            result = self.arming_client(True)
            if result.success:
                rospy.loginfo("✅ Drone armed")
            else:
                rospy.logerr("❌ Failed to arm drone")
                return False
        
        # Takeoff
        takeoff_setpoint = PoseStamped()
        takeoff_setpoint.header.stamp = rospy.Time.now()
        takeoff_setpoint.header.frame_id = "map"
        takeoff_setpoint.pose.position.z = altitude
        takeoff_setpoint.pose.orientation.w = 1.0
        
        # Send takeoff command repeatedly
        for _ in range(100):
            self.setpoint_pub.publish(takeoff_setpoint)
            self.rate.sleep()
        
        rospy.loginfo(f"✅ Takeoff complete to {altitude}m")
        return True
    
    def run(self):
        """Main navigation loop"""
        rospy.loginfo("🎮 Starting navigation control loop")
        
        try:
            while not rospy.is_shutdown():
                # Publish current setpoint if waypoint is active
                if self.navigation_active and self.current_waypoint:
                    setpoint = PoseStamped()
                    setpoint.header.stamp = rospy.Time.now()
                    setpoint.header.frame_id = "map"
                    setpoint.pose.position = self.current_waypoint
                    setpoint.pose.orientation.w = 1.0
                    self.setpoint_pub.publish(setpoint)
                
                self.rate.sleep()
                
        except rospy.ROSInterruptException:
            rospy.loginfo("Navigation loop interrupted")
        except Exception as e:
            rospy.logerr(f"Error in navigation loop: {e}")

# Service message definitions (create these as separate .srv files in a real package)
class GotoWaypointRequest:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

class EmergencyStopRequest:
    def __init__(self):
        pass

if __name__ == '__main__':
    try:
        navigator = SpatialReasoningNavigator()
        
        # Optional: Auto-arm and takeoff
        auto_takeoff = rospy.get_param('~auto_takeoff', False)
        if auto_takeoff:
            navigator.arm_and_takeoff(2.0)
        
        navigator.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Spatial reasoning navigator shutdown")
    except Exception as e:
        rospy.logerr(f"Fatal error in spatial reasoning navigator: {e}")
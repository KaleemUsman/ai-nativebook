#!/usr/bin/env python3
"""
Advanced AI Control Agent for Humanoid Robotics
This script implements a comprehensive AI agent that processes multiple sensor streams
and generates appropriate control commands for humanoid robot navigation and interaction.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState, Image
from geometry_msgs.msg import Twist, Vector3, Pose, Point
from std_msgs.msg import String, Float64MultiArray
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Time
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import threading
from collections import deque
import json
import math
from enum import Enum
import tensorflow as tf  # Example ML framework (would need to be installed)
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


class AgentState(Enum):
    """Enumeration of possible agent states"""
    IDLE = "idle"
    NAVIGATING = "navigating"
    AVOIDING_OBSTACLE = "avoiding_obstacle"
    BALANCING = "balancing"
    INTERACTING = "interacting"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SensorData:
    """Container for sensor data with timestamps"""
    laser_scan: Optional[LaserScan] = None
    imu_data: Optional[Imu] = None
    joint_states: Optional[JointState] = None
    camera_image: Optional[Image] = None
    odometry: Optional[Odometry] = None
    timestamp: float = 0.0


class HumanoidControlAgent(Node):
    """
    Advanced AI control agent for humanoid robots that processes sensor data
    and generates appropriate control commands for navigation, balance, and interaction.
    """

    def __init__(self):
        super().__init__('humanoid_control_agent')

        # Initialize internal state
        self.current_state = AgentState.IDLE
        self.target_position = None
        self.emergency_stop_active = False

        # Initialize data storage
        self.current_sensor_data = SensorData()
        self.sensor_history = deque(maxlen=50)  # Keep last 50 sensor readings
        self.control_history = deque(maxlen=50)  # Keep last 50 control commands

        # Initialize ROS interfaces
        self.bridge = CvBridge()

        # Subscribers for sensor data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/humanoid/scan',
            self.scan_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/humanoid/imu/data',
            self.imu_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            '/humanoid/camera/color/image_raw',
            self.camera_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/humanoid/odom',
            self.odom_callback,
            10
        )

        # Publishers for control commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/humanoid/cmd_vel',
            10
        )

        self.joint_cmd_pub = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )

        self.state_pub = self.create_publisher(
            String,
            '/ai_agent/state',
            10
        )

        self.debug_marker_pub = self.create_publisher(
            Marker,
            '/ai_agent/debug_markers',
            10
        )

        # Timer for main control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz control loop

        # Initialize AI components
        self.navigation_planner = NavigationPlanner()
        self.balance_controller = BalanceController()
        self.obstacle_avoider = ObstacleAvoider()
        self.behavior_selector = BehaviorSelector()

        # Initialize statistics
        self.stats = {
            'control_cycles': 0,
            'sensor_updates': 0,
            'state_transitions': 0,
            'emergency_stops': 0
        }

        self.get_logger().info("Humanoid Control Agent initialized and ready")

    def scan_callback(self, msg):
        """Process LiDAR scan data"""
        self.current_sensor_data.laser_scan = msg
        self.current_sensor_data.timestamp = time.time()
        self.sensor_history.append(self.current_sensor_data)
        self.stats['sensor_updates'] += 1

    def imu_callback(self, msg):
        """Process IMU data for balance and orientation"""
        self.current_sensor_data.imu_data = msg
        self.current_sensor_data.timestamp = time.time()
        self.sensor_history.append(self.current_sensor_data)

    def joint_callback(self, msg):
        """Process joint state data"""
        self.current_sensor_data.joint_states = msg
        self.current_sensor_data.timestamp = time.time()
        self.sensor_history.append(self.current_sensor_data)

    def camera_callback(self, msg):
        """Process camera image data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_sensor_data.camera_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Error processing camera data: {e}")

    def odom_callback(self, msg):
        """Process odometry data"""
        self.current_sensor_data.odometry = msg
        self.current_sensor_data.timestamp = time.time()
        self.sensor_history.append(self.current_sensor_data)

    def control_loop(self):
        """Main control loop that processes sensor data and generates commands"""
        try:
            if self.emergency_stop_active:
                self.publish_emergency_stop()
                return

            # Process sensor data and determine next action
            control_command = self.process_sensor_data()

            # Publish control command
            if control_command is not None:
                self.publish_control_command(control_command)

            # Update agent state based on sensor data and control output
            new_state = self.determine_agent_state()
            if new_state != self.current_state:
                self.transition_state(new_state)

            # Publish current state
            state_msg = String()
            state_msg.data = self.current_state.value
            self.state_pub.publish(state_msg)

            # Publish debug markers if needed
            self.publish_debug_markers()

            # Update statistics
            self.stats['control_cycles'] += 1

        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")
            self.trigger_emergency_stop()

    def process_sensor_data(self):
        """Process all available sensor data and generate control command"""
        if not self.all_sensors_available():
            return None

        # Extract sensor information
        laser_ranges = np.array(self.current_sensor_data.laser_scan.ranges)
        imu_orientation = self.extract_orientation(self.current_sensor_data.imu_data)
        joint_positions = np.array(self.current_sensor_data.joint_states.position)
        robot_pose = self.extract_pose(self.current_sensor_data.odometry)

        # Determine the most appropriate behavior based on current situation
        behavior = self.behavior_selector.select_behavior(
            laser_ranges, imu_orientation, joint_positions, robot_pose
        )

        # Generate control command based on selected behavior
        if behavior == 'navigation':
            return self.navigation_planner.plan_navigation(laser_ranges, robot_pose)
        elif behavior == 'obstacle_avoidance':
            return self.obstacle_avoider.avoid_obstacles(laser_ranges, robot_pose)
        elif behavior == 'balance':
            return self.balance_controller.maintain_balance(imu_orientation, joint_positions)
        elif behavior == 'interaction':
            return self.handle_interaction()
        else:
            return self.default_behavior(laser_ranges, imu_orientation, robot_pose)

    def all_sensors_available(self):
        """Check if all required sensors have recent data"""
        return (self.current_sensor_data.laser_scan is not None and
                self.current_sensor_data.imu_data is not None and
                self.current_sensor_data.joint_states is not None and
                self.current_sensor_data.odometry is not None)

    def extract_orientation(self, imu_msg):
        """Extract orientation quaternion from IMU message"""
        if imu_msg is None:
            return np.array([0, 0, 0, 1])  # Default: no rotation

        return np.array([
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        ])

    def extract_pose(self, odom_msg):
        """Extract position and orientation from odometry message"""
        if odom_msg is None:
            return {
                'position': np.array([0, 0, 0]),
                'orientation': np.array([0, 0, 0, 1])
            }

        pos = odom_msg.pose.pose.position
        quat = odom_msg.pose.pose.orientation

        return {
            'position': np.array([pos.x, pos.y, pos.z]),
            'orientation': np.array([quat.x, quat.y, quat.z, quat.w])
        }

    def determine_agent_state(self):
        """Determine current agent state based on sensor data"""
        if not self.all_sensors_available():
            return AgentState.IDLE

        laser_ranges = np.array(self.current_sensor_data.laser_scan.ranges)
        imu_orientation = self.extract_orientation(self.current_sensor_data.imu_data)

        # Check for emergency conditions
        if self.is_emergency_condition(laser_ranges, imu_orientation):
            return AgentState.EMERGENCY_STOP

        # Check for balance issues
        if self.is_balance_compromised(imu_orientation):
            return AgentState.BALANCING

        # Check for obstacles
        min_distance = np.min(laser_ranges[np.isfinite(laser_ranges)])
        if min_distance < 0.5:  # 50cm threshold
            return AgentState.AVOIDING_OBSTACLE

        # Check if we have a target
        if self.target_position is not None:
            return AgentState.NAVIGATING

        return AgentState.IDLE

    def is_emergency_condition(self, laser_ranges, imu_orientation):
        """Check if emergency stop condition is met"""
        # Check for imminent collision
        min_range = np.min(laser_ranges[np.isfinite(laser_ranges)])
        if min_range < 0.2:  # 20cm threshold
            return True

        # Check for dangerous orientation (falling)
        roll, pitch, yaw = self.quaternion_to_euler(imu_orientation)
        if abs(roll) > 1.0 or abs(pitch) > 1.0:  # 57 degrees threshold
            return True

        return False

    def is_balance_compromised(self, imu_orientation):
        """Check if robot balance is compromised"""
        roll, pitch, yaw = self.quaternion_to_euler(imu_orientation)
        return abs(roll) > 0.5 or abs(pitch) > 0.5  # 28 degrees threshold

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def transition_state(self, new_state):
        """Handle state transition"""
        old_state = self.current_state
        self.current_state = new_state
        self.stats['state_transitions'] += 1

        self.get_logger().info(f"State transition: {old_state.value} -> {new_state.value}")

        # Perform any state-specific initialization
        if new_state == AgentState.EMERGENCY_STOP:
            self.stats['emergency_stops'] += 1

    def publish_control_command(self, command):
        """Publish the generated control command"""
        if 'cmd_vel' in command:
            cmd_vel = Twist()
            cmd_vel.linear.x = command['cmd_vel']['linear']['x']
            cmd_vel.linear.y = command['cmd_vel']['linear']['y']
            cmd_vel.linear.z = command['cmd_vel']['linear']['z']
            cmd_vel.angular.x = command['cmd_vel']['angular']['x']
            cmd_vel.angular.y = command['cmd_vel']['angular']['y']
            cmd_vel.angular.z = command['cmd_vel']['angular']['z']
            self.cmd_vel_pub.publish(cmd_vel)

        if 'joint_commands' in command:
            joint_cmd = JointState()
            joint_cmd.header.stamp = self.get_clock().now().to_msg()
            joint_cmd.name = command['joint_commands']['names']
            joint_cmd.position = command['joint_commands']['positions']
            joint_cmd.velocity = command['joint_commands']['velocities']
            joint_cmd.effort = command['joint_commands']['efforts']
            self.joint_cmd_pub.publish(joint_cmd)

        # Add to control history
        self.control_history.append(command)

    def publish_emergency_stop(self):
        """Publish emergency stop command"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.linear.y = 0.0
        cmd_vel.linear.z = 0.0
        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        self.emergency_stop_active = True
        self.get_logger().warn("EMERGENCY STOP TRIGGERED")

    def default_behavior(self, laser_ranges, imu_orientation, robot_pose):
        """Default behavior when no specific task is active"""
        # For default behavior, just maintain current position
        command = {
            'cmd_vel': {
                'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            },
            'joint_commands': {
                'names': [],
                'positions': [],
                'velocities': [],
                'efforts': []
            }
        }

        # But if there are obstacles nearby, do some avoidance
        min_distance = np.min(laser_ranges[np.isfinite(laser_ranges)])
        if min_distance < 0.8:  # Start avoiding at 80cm
            command = self.obstacle_avoider.avoid_obstacles(laser_ranges, robot_pose)

        return command

    def handle_interaction(self):
        """Handle object interaction behavior"""
        # Placeholder for interaction logic
        command = {
            'cmd_vel': {
                'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            },
            'joint_commands': {
                'names': [],
                'positions': [],
                'velocities': [],
                'efforts': []
            }
        }
        return command

    def publish_debug_markers(self):
        """Publish debug visualization markers"""
        # Example: visualize the robot's goal if set
        if self.target_position is not None:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "ai_agent"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = self.target_position[0]
            marker.pose.position.y = self.target_position[1]
            marker.pose.position.z = self.target_position[2]

            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            self.debug_marker_pub.publish(marker)

    def set_target_position(self, x, y, z):
        """Set a navigation target position"""
        self.target_position = (x, y, z)

    def get_agent_status(self):
        """Get current agent status"""
        return {
            'state': self.current_state.value,
            'emergency_stop': self.emergency_stop_active,
            'sensor_data_available': self.all_sensors_available(),
            'statistics': self.stats
        }


class NavigationPlanner:
    """Handles path planning and navigation commands"""

    def __init__(self):
        self.current_path = []
        self.path_index = 0
        self.safe_distance = 0.5

    def plan_navigation(self, laser_ranges, robot_pose):
        """Plan navigation based on laser data and robot pose"""
        # Simple navigation: move forward if path is clear
        front_ranges = laser_ranges[150:210]  # Front 60 degrees
        min_front_distance = np.min(front_ranges[np.isfinite(front_ranges)])

        command = {
            'cmd_vel': {
                'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            },
            'joint_commands': {
                'names': [],
                'positions': [],
                'velocities': [],
                'efforts': []
            }
        }

        if min_front_distance > self.safe_distance:
            # Path is clear, move forward
            command['cmd_vel']['linear']['x'] = 0.3  # Move forward at 0.3 m/s
        else:
            # Path is blocked, turn slightly
            command['cmd_vel']['angular']['z'] = 0.2  # Turn right slowly

        return command


class BalanceController:
    """Handles robot balance maintenance"""

    def __init__(self):
        self.balance_threshold = 0.3  # Radians

    def maintain_balance(self, imu_orientation, joint_positions):
        """Generate commands to maintain robot balance"""
        roll, pitch, yaw = self.quaternion_to_euler(imu_orientation)

        command = {
            'cmd_vel': {
                'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            },
            'joint_commands': {
                'names': [],
                'positions': [],
                'velocities': [],
                'efforts': []
            }
        }

        # Simple balance control: adjust based on roll and pitch
        # This is a placeholder - real balance control would be much more complex
        if abs(roll) > self.balance_threshold or abs(pitch) > self.balance_threshold:
            # Emergency balance correction
            command['cmd_vel']['angular']['x'] = -pitch * 2.0  # Correct pitch
            command['cmd_vel']['angular']['y'] = -roll * 2.0   # Correct roll

        return command

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


class ObstacleAvoider:
    """Handles obstacle avoidance behaviors"""

    def __init__(self):
        self.avoidance_distance = 0.6  # Meters

    def avoid_obstacles(self, laser_ranges, robot_pose):
        """Generate commands to avoid obstacles"""
        command = {
            'cmd_vel': {
                'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            },
            'joint_commands': {
                'names': [],
                'positions': [],
                'velocities': [],
                'efforts': []
            }
        }

        # Analyze laser ranges for obstacles
        front_left = laser_ranges[120:150]  # Front-left sector
        front_center = laser_ranges[150:210]  # Front-center sector
        front_right = laser_ranges[210:240]  # Front-right sector

        min_left = np.min(front_left[np.isfinite(front_left)])
        min_center = np.min(front_center[np.isfinite(front_center)])
        min_right = np.min(front_right[np.isfinite(front_right)])

        # Simple obstacle avoidance logic
        if min_center < self.avoidance_distance:
            # Obstacle directly ahead - turn away
            if min_left > min_right:
                command['cmd_vel']['angular']['z'] = 0.4  # Turn left
            else:
                command['cmd_vel']['angular']['z'] = -0.4  # Turn right
        elif min_left < self.avoidance_distance or min_right < self.avoidance_distance:
            # Obstacles on sides - move forward cautiously
            command['cmd_vel']['linear']['x'] = 0.1  # Slow forward movement
        else:
            # Path is relatively clear - move forward
            command['cmd_vel']['linear']['x'] = 0.2

        return command


class BehaviorSelector:
    """Selects appropriate behavior based on sensor data"""

    def __init__(self):
        pass

    def select_behavior(self, laser_ranges, imu_orientation, joint_positions, robot_pose):
        """Select the most appropriate behavior based on current conditions"""
        # Check for critical conditions first
        if self.is_dangerous_orientation(imu_orientation):
            return 'balance'

        if self.is_obstacle_immediate(laser_ranges):
            return 'obstacle_avoidance'

        if self.has_navigation_target():
            return 'navigation'

        # Default to safe behavior
        return 'navigation'  # Default behavior

    def is_dangerous_orientation(self, imu_orientation):
        """Check if orientation is dangerous"""
        roll, pitch, yaw = self.quaternion_to_euler(imu_orientation)
        return abs(roll) > 0.8 or abs(pitch) > 0.8

    def is_obstacle_immediate(self, laser_ranges):
        """Check if there's an immediate obstacle"""
        front_ranges = laser_ranges[150:210]
        min_front = np.min(front_ranges[np.isfinite(front_ranges)])
        return min_front < 0.4  # 40cm threshold

    def has_navigation_target(self):
        """Check if there's a navigation target set"""
        # This would check for a target in a real implementation
        return True

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    """Main function to run the AI control agent"""
    rclpy.init(args=args)

    agent = HumanoidControlAgent()

    try:
        # Example: Set a target position
        agent.set_target_position(5.0, 0.0, 0.0)

        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info(f"AI Control Agent stopped. Final stats: {agent.get_agent_status()}")
    finally:
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
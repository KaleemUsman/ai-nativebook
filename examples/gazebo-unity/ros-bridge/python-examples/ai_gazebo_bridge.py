#!/usr/bin/env python3
"""
AI Agent to Gazebo Bridge Example
This script demonstrates how to connect an AI agent to Gazebo sensor data
and process it for robot control decisions.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState, Image
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String, Float64MultiArray
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import threading
from collections import deque
import json


class AIGazeboBridge(Node):
    """
    Bridge between AI agent and Gazebo simulation environment.
    Handles sensor data collection, processing, and control command execution.
    """

    def __init__(self):
        super().__init__('ai_gazebo_bridge')

        # Initialize data storage
        self.laser_data = None
        self.imu_data = None
        self.joint_states = None
        self.camera_data = None
        self.odometry_data = None

        # Data history for temporal processing
        self.laser_history = deque(maxlen=10)
        self.imu_history = deque(maxlen=10)

        # Initialize ROS interfaces
        self.bridge = CvBridge()

        # Create subscribers for Gazebo sensor data
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

        # Create publishers for control commands
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

        self.ai_status_pub = self.create_publisher(
            String,
            '/ai_agent/status',
            10
        )

        # Timer for AI processing loop
        self.ai_timer = self.create_timer(0.1, self.ai_processing_loop)  # 10 Hz

        # AI agent instance
        self.ai_agent = SimpleNavigationAgent()

        # Status tracking
        self.agent_status = "initialized"
        self.last_update_time = self.get_clock().now()

        self.get_logger().info("AI Gazebo Bridge initialized and connected to simulation")

    def scan_callback(self, msg):
        """Process LiDAR scan data from Gazebo"""
        try:
            # Convert to numpy array and store
            ranges = np.array(msg.ranges)
            # Handle invalid ranges (inf, nan)
            ranges = np.where(np.isinf(ranges), msg.range_max, ranges)
            ranges = np.where(np.isnan(ranges), msg.range_max, ranges)

            self.laser_data = {
                'ranges': ranges,
                'intensities': np.array(msg.intensities),
                'angle_min': msg.angle_min,
                'angle_max': msg.angle_max,
                'angle_increment': msg.angle_increment,
                'time_increment': msg.time_increment,
                'scan_time': msg.scan_time,
                'range_min': msg.range_min,
                'range_max': msg.range_max,
                'header': msg.header
            }

            # Add to history for temporal processing
            self.laser_history.append(self.laser_data)
        except Exception as e:
            self.get_logger().error(f"Error processing laser scan: {e}")

    def imu_callback(self, msg):
        """Process IMU data from Gazebo"""
        try:
            self.imu_data = {
                'orientation': {
                    'x': msg.orientation.x,
                    'y': msg.orientation.y,
                    'z': msg.orientation.z,
                    'w': msg.orientation.w
                },
                'angular_velocity': {
                    'x': msg.angular_velocity.x,
                    'y': msg.angular_velocity.y,
                    'z': msg.angular_velocity.z
                },
                'linear_acceleration': {
                    'x': msg.linear_acceleration.x,
                    'y': msg.linear_acceleration.y,
                    'z': msg.linear_acceleration.z
                },
                'header': msg.header
            }

            # Add to history for temporal processing
            self.imu_history.append(self.imu_data)
        except Exception as e:
            self.get_logger().error(f"Error processing IMU data: {e}")

    def joint_callback(self, msg):
        """Process joint state data from Gazebo"""
        try:
            self.joint_states = {
                'names': list(msg.name),
                'positions': list(msg.position),
                'velocities': list(msg.velocity),
                'efforts': list(msg.effort),
                'header': msg.header
            }
        except Exception as e:
            self.get_logger().error(f"Error processing joint states: {e}")

    def camera_callback(self, msg):
        """Process camera image data from Gazebo"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.camera_data = {
                'image': cv_image,
                'width': msg.width,
                'height': msg.height,
                'encoding': msg.encoding,
                'header': msg.header
            }
        except Exception as e:
            self.get_logger().error(f"Error processing camera data: {e}")

    def odom_callback(self, msg):
        """Process odometry data from Gazebo"""
        try:
            self.odometry_data = {
                'pose': {
                    'position': {
                        'x': msg.pose.pose.position.x,
                        'y': msg.pose.pose.position.y,
                        'z': msg.pose.pose.position.z
                    },
                    'orientation': {
                        'x': msg.pose.pose.orientation.x,
                        'y': msg.pose.pose.orientation.y,
                        'z': msg.pose.pose.orientation.z,
                        'w': msg.pose.pose.orientation.w
                    }
                },
                'twist': {
                    'linear': {
                        'x': msg.twist.twist.linear.x,
                        'y': msg.twist.twist.linear.y,
                        'z': msg.twist.twist.linear.z
                    },
                    'angular': {
                        'x': msg.twist.twist.angular.x,
                        'y': msg.twist.twist.angular.y,
                        'z': msg.twist.twist.angular.z
                    }
                },
                'header': msg.header
            }
        except Exception as e:
            self.get_logger().error(f"Error processing odometry data: {e}")

    def ai_processing_loop(self):
        """Main AI processing loop"""
        try:
            # Check if we have sufficient sensor data
            if self.all_sensors_ready():
                # Get current sensor state
                sensor_state = self.get_sensor_state()

                # Process with AI agent
                action = self.ai_agent.get_action(sensor_state)

                # Execute action
                self.execute_action(action)

                # Update status
                self.agent_status = "active"
                self.last_update_time = self.get_clock().now()

                # Publish status
                status_msg = String()
                status_msg.data = f"AI Agent Active - Sensors: {len(sensor_state)}"
                self.ai_status_pub.publish(status_msg)
            else:
                self.agent_status = "waiting_for_sensors"
        except Exception as e:
            self.get_logger().error(f"Error in AI processing loop: {e}")
            self.agent_status = "error"

    def all_sensors_ready(self):
        """Check if all required sensors have data"""
        return (self.laser_data is not None and
                self.imu_data is not None and
                self.joint_states is not None)

    def get_sensor_state(self):
        """Get current sensor state for AI agent"""
        state = {}

        if self.laser_data is not None:
            # Process laser data - extract relevant features
            ranges = self.laser_data['ranges']
            state['laser_ranges'] = ranges
            state['laser_min_distance'] = np.min(ranges)
            state['laser_front_distance'] = np.mean(ranges[160:200])  # Front 40 degrees
            state['laser_left_distance'] = np.mean(ranges[80:120])   # Left side
            state['laser_right_distance'] = np.mean(ranges[280:320]) # Right side

        if self.imu_data is not None:
            # Process IMU data
            state['imu_orientation'] = self.imu_data['orientation']
            state['imu_angular_velocity'] = self.imu_data['angular_velocity']
            state['imu_linear_acceleration'] = self.imu_data['linear_acceleration']

        if self.joint_states is not None:
            # Process joint data
            state['joint_positions'] = self.joint_states['positions']
            state['joint_velocities'] = self.joint_states['velocities']
            state['joint_names'] = self.joint_states['names']

        if self.odometry_data is not None:
            # Process odometry data
            state['position'] = self.odometry_data['pose']['position']
            state['orientation'] = self.odometry_data['pose']['orientation']
            state['linear_velocity'] = self.odometry_data['twist']['linear']
            state['angular_velocity'] = self.odometry_data['twist']['angular']

        return state

    def execute_action(self, action):
        """Execute action returned by AI agent"""
        if action is None:
            return

        # Handle different action types
        if 'cmd_vel' in action:
            cmd_vel = Twist()
            cmd_vel.linear.x = action['cmd_vel']['linear']['x']
            cmd_vel.linear.y = action['cmd_vel']['linear']['y']
            cmd_vel.linear.z = action['cmd_vel']['linear']['z']
            cmd_vel.angular.x = action['cmd_vel']['angular']['x']
            cmd_vel.angular.y = action['cmd_vel']['angular']['y']
            cmd_vel.angular.z = action['cmd_vel']['angular']['z']
            self.cmd_vel_pub.publish(cmd_vel)

        if 'joint_commands' in action:
            joint_cmd = JointState()
            joint_cmd.header.stamp = self.get_clock().now().to_msg()
            joint_cmd.name = action['joint_commands']['names']
            joint_cmd.position = action['joint_commands']['positions']
            joint_cmd.velocity = action['joint_commands']['velocities']
            joint_cmd.effort = action['joint_commands']['efforts']
            self.joint_cmd_pub.publish(joint_cmd)

    def get_agent_status(self):
        """Get current status of the AI agent"""
        return {
            'status': self.agent_status,
            'last_update': self.last_update_time,
            'sensors_ready': self.all_sensors_ready(),
            'sensor_data_count': {
                'laser': 1 if self.laser_data else 0,
                'imu': 1 if self.imu_data else 0,
                'joint': 1 if self.joint_states else 0,
                'camera': 1 if self.camera_data else 0,
                'odometry': 1 if self.odometry_data else 0
            }
        }


class SimpleNavigationAgent:
    """
    Simple AI agent for navigation tasks.
    This is a basic example - in practice, this could be a neural network,
    reinforcement learning agent, or other AI system.
    """

    def __init__(self):
        self.agent_type = "simple_navigation"
        self.parameters = {
            'safety_distance': 0.5,  # Minimum distance to obstacles
            'target_velocity': 0.5,   # Desired forward velocity
            'rotation_speed': 0.5     # Desired rotation speed
        }

    def get_action(self, sensor_state):
        """Get action based on sensor state"""
        if not sensor_state:
            return None

        action = {'cmd_vel': {'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                             'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}},
                 'joint_commands': {'names': [], 'positions': [], 'velocities': [], 'efforts': []}}

        # Simple navigation logic based on laser data
        if 'laser_ranges' in sensor_state:
            laser_ranges = sensor_state['laser_ranges']
            min_distance = sensor_state['laser_min_distance']

            # Simple obstacle avoidance
            if min_distance < self.parameters['safety_distance']:
                # Stop and turn
                action['cmd_vel']['linear']['x'] = 0.0
                action['cmd_vel']['angular']['z'] = self.parameters['rotation_speed']
            else:
                # Move forward
                action['cmd_vel']['linear']['x'] = self.parameters['target_velocity']
                action['cmd_vel']['angular']['z'] = 0.0

        # Balance control based on IMU data
        if 'imu_orientation' in sensor_state:
            orientation = sensor_state['imu_orientation']
            # Simple balance control would go here
            # For now, just ensure we're not falling over

        return action

    def update_model(self, experience_data):
        """Update the agent model with new experience (for learning agents)"""
        # Implementation for model-based agents
        pass

    def get_state_representation(self, sensor_state):
        """Convert sensor state to appropriate format for the agent"""
        # Normalize and format sensor data for the agent
        processed_state = {}

        # Example: normalize laser ranges
        if 'laser_ranges' in sensor_state:
            laser_ranges = sensor_state['laser_ranges']
            processed_state['normalized_laser'] = laser_ranges / 30.0  # Normalize by max range

        # Example: extract orientation angles
        if 'imu_orientation' in sensor_state:
            orientation = sensor_state['imu_orientation']
            # Convert quaternion to euler angles if needed
            processed_state['orientation'] = self.quaternion_to_euler(orientation)

        return processed_state

    def quaternion_to_euler(self, quat_dict):
        """Convert quaternion to euler angles"""
        # Simplified conversion - in practice use proper quaternion math
        w, x, y, z = quat_dict['w'], quat_dict['x'], quat_dict['y'], quat_dict['z']

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return {'roll': roll, 'pitch': pitch, 'yaw': yaw}


def main(args=None):
    """Main function to run the AI Gazebo Bridge"""
    rclpy.init(args=args)

    ai_bridge = AIGazeboBridge()

    try:
        rclpy.spin(ai_bridge)
    except KeyboardInterrupt:
        ai_bridge.get_logger().info("AI Gazebo Bridge interrupted by user")
    finally:
        ai_bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
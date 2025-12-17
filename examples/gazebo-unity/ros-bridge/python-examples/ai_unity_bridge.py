#!/usr/bin/env python3
"""
AI Agent to Unity Bridge Example
This script demonstrates how to connect an AI agent to Unity visualization
and process both Gazebo physics and Unity visualization data for robot control.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState, Image
from geometry_msgs.msg import Twist, Vector3, PoseStamped, Point
from std_msgs.msg import String, Float64MultiArray, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import threading
from collections import deque
import json
import math


class AIUnityBridge(Node):
    """
    Bridge between AI agent and Unity visualization environment.
    Handles data synchronization between Gazebo physics and Unity visualization,
    and processes both for AI agent decision making.
    """

    def __init__(self):
        super().__init__('ai_unity_bridge')

        # Initialize data storage for both Gazebo and Unity
        self.gazebo_data = {
            'laser': None,
            'imu': None,
            'joint_states': None,
            'camera': None,
            'odometry': None
        }

        self.unity_data = {
            'joint_states': None,
            'robot_pose': None,
            'visualization_data': None
        }

        # Data history for temporal processing
        self.gazebo_history = {
            'laser': deque(maxlen=10),
            'imu': deque(maxlen=10),
            'joint_states': deque(maxlen=10)
        }

        # Initialize ROS interfaces
        self.bridge = CvBridge()

        # Subscribers for Gazebo physics data
        self.gazebo_scan_sub = self.create_subscription(
            LaserScan,
            '/gazebo/scan',
            self.gazebo_scan_callback,
            10
        )

        self.gazebo_imu_sub = self.create_subscription(
            Imu,
            '/gazebo/imu/data',
            self.gazebo_imu_callback,
            10
        )

        self.gazebo_joint_sub = self.create_subscription(
            JointState,
            '/gazebo/joint_states',
            self.gazebo_joint_callback,
            10
        )

        # Subscribers for Unity visualization data
        self.unity_joint_sub = self.create_subscription(
            JointState,
            '/unity/joint_states',
            self.unity_joint_callback,
            10
        )

        self.unity_pose_sub = self.create_subscription(
            PoseStamped,
            '/unity/robot_pose',
            self.unity_pose_callback,
            10
        )

        # Publishers for Unity visualization commands
        self.unity_marker_pub = self.create_publisher(
            Marker,
            '/unity/visualization_marker',
            10
        )

        self.unity_control_pub = self.create_publisher(
            JointState,
            '/unity/control_commands',
            10
        )

        # Publishers for AI status and coordination
        self.ai_status_pub = self.create_publisher(
            String,
            '/ai_agent/unity_status',
            10
        )

        self.synchronization_pub = self.create_publisher(
            String,
            '/unity/gazebo_sync',
            10
        )

        # Timer for AI processing loop
        self.ai_timer = self.create_timer(0.033, self.ai_processing_loop)  # ~30 Hz for Unity sync

        # AI agent instance for Unity integration
        self.ai_agent = UnityIntegratedAgent()

        # Synchronization parameters
        self.sync_threshold = 0.05  # 50ms threshold for synchronization
        self.last_sync_time = self.get_clock().now()

        # Status tracking
        self.agent_status = "initialized"
        self.synchronization_status = "waiting"

        self.get_logger().info("AI Unity Bridge initialized and connected to visualization")

    def gazebo_scan_callback(self, msg):
        """Process LiDAR scan data from Gazebo (physics simulation)"""
        try:
            # Convert to numpy array and store
            ranges = np.array(msg.ranges)
            # Handle invalid ranges (inf, nan)
            ranges = np.where(np.isinf(ranges), msg.range_max, ranges)
            ranges = np.where(np.isnan(ranges), msg.range_max, ranges)

            self.gazebo_data['laser'] = {
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
            self.gazebo_history['laser'].append(self.gazebo_data['laser'])
        except Exception as e:
            self.get_logger().error(f"Error processing Gazebo laser scan: {e}")

    def gazebo_imu_callback(self, msg):
        """Process IMU data from Gazebo (physics simulation)"""
        try:
            self.gazebo_data['imu'] = {
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
            self.gazebo_history['imu'].append(self.gazebo_data['imu'])
        except Exception as e:
            self.get_logger().error(f"Error processing Gazebo IMU data: {e}")

    def gazebo_joint_callback(self, msg):
        """Process joint state data from Gazebo (physics simulation)"""
        try:
            self.gazebo_data['joint_states'] = {
                'names': list(msg.name),
                'positions': list(msg.position),
                'velocities': list(msg.velocity),
                'efforts': list(msg.effort),
                'header': msg.header
            }

            # Add to history for temporal processing
            self.gazebo_history['joint_states'].append(self.gazebo_data['joint_states'])
        except Exception as e:
            self.get_logger().error(f"Error processing Gazebo joint states: {e}")

    def unity_joint_callback(self, msg):
        """Process joint state data from Unity (visualization)"""
        try:
            self.unity_data['joint_states'] = {
                'names': list(msg.name),
                'positions': list(msg.position),
                'velocities': list(msg.velocity),
                'efforts': list(msg.effort),
                'header': msg.header
            }
        except Exception as e:
            self.get_logger().error(f"Error processing Unity joint states: {e}")

    def unity_pose_callback(self, msg):
        """Process robot pose data from Unity (visualization)"""
        try:
            self.unity_data['robot_pose'] = {
                'position': {
                    'x': msg.pose.position.x,
                    'y': msg.pose.position.y,
                    'z': msg.pose.position.z
                },
                'orientation': {
                    'x': msg.pose.orientation.x,
                    'y': msg.pose.orientation.y,
                    'z': msg.pose.orientation.z,
                    'w': msg.pose.orientation.w
                },
                'header': msg.header
            }
        except Exception as e:
            self.get_logger().error(f"Error processing Unity robot pose: {e}")

    def ai_processing_loop(self):
        """Main AI processing loop for Unity integration"""
        try:
            # Check if we have data from both Gazebo and Unity
            if self.both_systems_ready():
                # Get integrated sensor state
                integrated_state = self.get_integrated_state()

                # Process with AI agent
                action = self.ai_agent.get_action(integrated_state)

                # Execute action (both physics and visualization)
                self.execute_integrated_action(action)

                # Update synchronization
                self.update_synchronization()

                # Update status
                self.agent_status = "active"
                self.synchronization_status = "synchronized"

                # Publish status
                status_msg = String()
                status_msg.data = f"AI Agent Active - Gazebo+Unity: {len(integrated_state)}"
                self.ai_status_pub.publish(status_msg)
            else:
                self.agent_status = "waiting_for_data"
                self.synchronization_status = "unsynchronized"

                # Request resynchronization if needed
                if not self.gazebo_data_ready() and not self.unity_data_ready():
                    self.request_resynchronization()

        except Exception as e:
            self.get_logger().error(f"Error in AI processing loop: {e}")
            self.agent_status = "error"
            self.synchronization_status = "error"

    def both_systems_ready(self):
        """Check if both Gazebo and Unity have sufficient data"""
        return (self.gazebo_data_ready() and self.unity_data_ready())

    def gazebo_data_ready(self):
        """Check if Gazebo data is ready"""
        return (self.gazebo_data['laser'] is not None and
                self.gazebo_data['imu'] is not None and
                self.gazebo_data['joint_states'] is not None)

    def unity_data_ready(self):
        """Check if Unity data is ready"""
        return (self.unity_data['joint_states'] is not None and
                self.unity_data['robot_pose'] is not None)

    def get_integrated_state(self):
        """Get integrated state from both Gazebo and Unity"""
        state = {}

        # Gazebo (physics) data
        if self.gazebo_data['laser'] is not None:
            state['gazebo_laser'] = self.gazebo_data['laser']

        if self.gazebo_data['imu'] is not None:
            state['gazebo_imu'] = self.gazebo_data['imu']

        if self.gazebo_data['joint_states'] is not None:
            state['gazebo_joint_states'] = self.gazebo_data['joint_states']

        # Unity (visualization) data
        if self.unity_data['joint_states'] is not None:
            state['unity_joint_states'] = self.unity_data['joint_states']

        if self.unity_data['robot_pose'] is not None:
            state['unity_robot_pose'] = self.unity_data['robot_pose']

        # Synchronization data
        state['synchronization_status'] = self.synchronization_status
        state['timestamp'] = self.get_clock().now().nanoseconds

        return state

    def execute_integrated_action(self, action):
        """Execute action in both Gazebo and Unity"""
        if action is None:
            return

        # Execute in Gazebo (physics)
        if 'gazebo_cmd' in action:
            # This would typically be handled by a separate Gazebo controller
            pass

        # Execute in Unity (visualization)
        if 'unity_cmd' in action:
            if 'joint_commands' in action['unity_cmd']:
                joint_cmd = JointState()
                joint_cmd.header.stamp = self.get_clock().now().to_msg()
                joint_cmd.name = action['unity_cmd']['joint_commands']['names']
                joint_cmd.position = action['unity_cmd']['joint_commands']['positions']
                joint_cmd.velocity = action['unity_cmd']['joint_commands']['velocities']
                joint_cmd.effort = action['unity_cmd']['joint_commands']['efforts']
                self.unity_control_pub.publish(joint_cmd)

            if 'visualization' in action['unity_cmd']:
                self.publish_visualization(action['unity_cmd']['visualization'])

    def publish_visualization(self, vis_data):
        """Publish visualization markers to Unity"""
        if 'trajectory' in vis_data:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "ai_trajectory"
            marker.id = 0
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            # Set trajectory points
            for point_data in vis_data['trajectory']:
                point = Point()
                point.x = point_data['x']
                point.y = point_data['y']
                point.z = point_data['z']
                marker.points.append(point)

            # Set marker properties
            marker.scale.x = 0.05  # Line width
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Alpha

            self.unity_marker_pub.publish(marker)

    def update_synchronization(self):
        """Update synchronization between Gazebo and Unity"""
        current_time = self.get_clock().now()

        # Calculate time difference
        time_diff = (current_time - self.last_sync_time).nanoseconds / 1e9  # Convert to seconds

        if time_diff > self.sync_threshold:
            # Send synchronization signal
            sync_msg = String()
            sync_msg.data = f"sync_{current_time.nanoseconds}"
            self.synchronization_pub.publish(sync_msg)
            self.last_sync_time = current_time

    def request_resynchronization(self):
        """Request resynchronization between systems"""
        sync_msg = String()
        sync_msg.data = "resync_request"
        self.synchronization_pub.publish(sync_msg)

    def get_system_status(self):
        """Get current status of both systems"""
        return {
            'gazebo_ready': self.gazebo_data_ready(),
            'unity_ready': self.unity_data_ready(),
            'both_ready': self.both_systems_ready(),
            'synchronization_status': self.synchronization_status,
            'agent_status': self.agent_status,
            'data_counts': {
                'gazebo_laser': 1 if self.gazebo_data['laser'] else 0,
                'gazebo_imu': 1 if self.gazebo_data['imu'] else 0,
                'gazebo_joints': 1 if self.gazebo_data['joint_states'] else 0,
                'unity_joints': 1 if self.unity_data['joint_states'] else 0,
                'unity_pose': 1 if self.unity_data['robot_pose'] else 0
            }
        }


class UnityIntegratedAgent:
    """
    AI agent that specifically handles Unity visualization integration.
    Processes both physics and visualization data for enhanced decision making.
    """

    def __init__(self):
        self.agent_type = "unity_integrated"
        self.parameters = {
            'safety_distance': 0.5,
            'visualization_weight': 0.3,  # How much to weight Unity visualization data
            'synchronization_tolerance': 0.1  # Tolerance for sync differences
        }

        # State tracking for Unity-specific features
        self.last_visualization_update = None
        self.trajectory_history = deque(maxlen=50)

    def get_action(self, integrated_state):
        """Get action based on integrated Gazebo+Unity state"""
        if not integrated_state:
            return None

        action = {
            'gazebo_cmd': {},
            'unity_cmd': {
                'joint_commands': {'names': [], 'positions': [], 'velocities': [], 'efforts': []},
                'visualization': {}
            }
        }

        # Process Gazebo physics data for control
        if 'gazebo_laser' in integrated_state:
            laser_data = integrated_state['gazebo_laser']
            min_distance = np.min(laser_data['ranges'])

            # Simple navigation based on laser data
            if min_distance < self.parameters['safety_distance']:
                # Obstacle detected - stop and turn
                action['gazebo_cmd']['linear_velocity'] = 0.0
                action['gazebo_cmd']['angular_velocity'] = 0.5
            else:
                # Clear path - move forward
                action['gazebo_cmd']['linear_velocity'] = 0.5
                action['gazebo_cmd']['angular_velocity'] = 0.0

        # Process Unity visualization data for enhanced awareness
        if 'unity_robot_pose' in integrated_state:
            robot_pose = integrated_state['unity_robot_pose']
            # Use Unity pose for visualization-aware navigation
            self.update_trajectory_history(robot_pose)

        # Generate visualization commands
        action['unity_cmd']['visualization'] = self.generate_visualization_commands()

        # Generate Unity joint commands (for visualization)
        if 'unity_joint_states' in integrated_state:
            unity_joints = integrated_state['unity_joint_states']
            action['unity_cmd']['joint_commands']['names'] = unity_joints['names']
            action['unity_cmd']['joint_commands']['positions'] = unity_joints['positions']
            action['unity_cmd']['joint_commands']['velocities'] = unity_joints['velocities']
            action['unity_cmd']['joint_commands']['efforts'] = unity_joints['efforts']

        return action

    def update_trajectory_history(self, robot_pose):
        """Update trajectory history for visualization"""
        trajectory_point = {
            'x': robot_pose['position']['x'],
            'y': robot_pose['position']['y'],
            'z': robot_pose['position']['z'],
            'timestamp': time.time()
        }
        self.trajectory_history.append(trajectory_point)

    def generate_visualization_commands(self):
        """Generate visualization commands for Unity"""
        viz_commands = {}

        # Add trajectory visualization if we have enough history
        if len(self.trajectory_history) > 1:
            trajectory = []
            for point in list(self.trajectory_history)[-20:]:  # Last 20 points
                trajectory.append({
                    'x': point['x'],
                    'y': point['y'],
                    'z': point['z']
                })
            viz_commands['trajectory'] = trajectory

        # Add other visualization elements as needed
        viz_commands['status_indicator'] = self.get_agent_status_color()

        return viz_commands

    def get_agent_status_color(self):
        """Get color indicator based on agent status"""
        # Return color based on internal state
        return {'r': 0.0, 'g': 1.0, 'b': 0.0, 'a': 1.0}  # Green for active

    def synchronize_with_visualization(self, gazebo_state, unity_state):
        """Synchronize decision making between physics and visualization"""
        # Weight physics data more heavily than visualization
        # Physics is the "ground truth" for control
        physics_weight = 1.0 - self.parameters['visualization_weight']
        viz_weight = self.parameters['visualization_weight']

        # Combine states with appropriate weighting
        combined_state = {}

        # For position, use physics as primary but visualize with Unity
        if 'gazebo_position' in gazebo_state and 'unity_position' in unity_state:
            # Physics is primary for control
            combined_state['control_position'] = gazebo_state['gazebo_position']
            # Visualization can show both
            combined_state['visualization_position'] = unity_state['unity_position']

        return combined_state

    def validate_synchronization(self, gazebo_data, unity_data):
        """Validate that Gazebo and Unity data are properly synchronized"""
        # Check temporal consistency
        if 'timestamp' in gazebo_data and 'timestamp' in unity_data:
            time_diff = abs(gazebo_data['timestamp'] - unity_data['timestamp'])
            if time_diff > self.parameters['synchronization_tolerance']:
                return False, f"Time desync: {time_diff}s"

        # Check spatial consistency (if both have position data)
        if ('gazebo_position' in gazebo_data and 'unity_position' in unity_data):
            pos_diff = self.calculate_position_difference(
                gazebo_data['gazebo_position'],
                unity_data['unity_position']
            )
            if pos_diff > self.parameters['synchronization_tolerance']:
                return False, f"Position desync: {pos_diff}m"

        return True, "Synchronized"

    def calculate_position_difference(self, pos1, pos2):
        """Calculate 3D distance between two positions"""
        dx = pos1['x'] - pos2['x']
        dy = pos1['y'] - pos2['y']
        dz = pos1['z'] - pos2['z']
        return math.sqrt(dx*dx + dy*dy + dz*dz)


def main(args=None):
    """Main function to run the AI Unity Bridge"""
    rclpy.init(args=args)

    ai_bridge = AIUnityBridge()

    try:
        rclpy.spin(ai_bridge)
    except KeyboardInterrupt:
        ai_bridge.get_logger().info("AI Unity Bridge interrupted by user")
    finally:
        ai_bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
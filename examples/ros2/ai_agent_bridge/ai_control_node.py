#!/usr/bin/env python3
"""
AI Control Node

This example demonstrates a complete AI control node that integrates
sensing, decision-making, and actuation in a single ROS 2 node.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Float64
from nav_msgs.msg import Odometry
import math
import numpy as np


class AIControlNode(Node):
    """
    A complete AI control node that demonstrates sensing, decision-making,
    and actuation in a single ROS 2 node.
    """

    def __init__(self):
        super().__init__('ai_control_node')

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_publisher = self.create_publisher(String, '/ai_status', 10)
        self.goal_publisher = self.create_publisher(Vector3, '/ai_goal', 10)

        # Subscribers
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.goal_subscription = self.create_subscription(
            Vector3,
            '/external_goal',
            self.goal_callback,
            10
        )

        # Timer for main control loop (10 Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Internal state
        self.latest_scan = None
        self.current_pose = None
        self.current_twist = None
        self.current_goal = Vector3(x=1.0, y=1.0, z=0.0)  # Default goal
        self.state = "IDLE"  # IDLE, NAVIGATING, AVOIDING, REACHED_GOAL
        self.obstacle_threshold = 0.8  # meters
        self.goal_threshold = 0.3      # meters

        # Navigation state
        self.path_history = []
        self.last_command_time = self.get_clock().now()

        self.get_logger().info('AI Control Node initialized')

    def scan_callback(self, msg):
        """
        Process laser scan data to detect obstacles.
        """
        self.latest_scan = msg
        self.check_for_obstacles()

    def odom_callback(self, msg):
        """
        Process odometry data to track robot position and velocity.
        """
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

        # Store position in path history
        if self.current_pose:
            pos = (self.current_pose.position.x, self.current_pose.position.y)
            self.path_history.append(pos)
            # Keep only the last 100 positions
            if len(self.path_history) > 100:
                self.path_history = self.path_history[-100:]

    def goal_callback(self, msg):
        """
        Update the navigation goal.
        """
        self.current_goal = msg
        self.state = "NAVIGATING"
        self.get_logger().info(f'New goal received: ({msg.x}, {msg.y})')

    def check_for_obstacles(self):
        """
        Check laser scan data for obstacles.
        """
        if self.latest_scan is None:
            return

        # Check the front 90-degree field for obstacles
        ranges = self.latest_scan.ranges
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment

        # Calculate indices for front 90-degree field
        center_idx = len(ranges) // 2
        angle_range = math.pi / 4  # 45 degrees on each side = 90 degrees total
        angle_range_idx = int(angle_range / angle_increment)

        start_idx = max(0, center_idx - angle_range_idx)
        end_idx = min(len(ranges), center_idx + angle_range_idx)

        front_ranges = ranges[start_idx:end_idx]
        if front_ranges:
            min_distance = min([r for r in front_ranges if not math.isinf(r) and not math.isnan(r)], default=float('inf'))
            if min_distance < self.obstacle_threshold:
                self.state = "AVOIDING"
                self.get_logger().info(f'Obstacle detected at {min_distance:.2f}m, switching to avoidance mode')

    def calculate_distance_to_goal(self):
        """
        Calculate the distance from current position to goal.
        """
        if self.current_pose is None:
            return float('inf')

        dx = self.current_goal.x - self.current_pose.position.x
        dy = self.current_goal.y - self.current_pose.position.y
        return math.sqrt(dx*dx + dy*dy)

    def navigate_to_goal(self):
        """
        Calculate velocity command to navigate toward the goal.
        """
        if self.current_pose is None:
            return Twist()

        # Calculate direction to goal
        dx = self.current_goal.x - self.current_pose.position.x
        dy = self.current_goal.y - self.current_pose.position.y
        distance_to_goal = math.sqrt(dx*dx + dy*dy)

        # Check if goal is reached
        if distance_to_goal < self.goal_threshold:
            self.state = "REACHED_GOAL"
            self.get_logger().info(f'Goal reached! Distance: {distance_to_goal:.2f}m')
            return Twist()  # Stop

        # Calculate desired angle to goal
        desired_angle = math.atan2(dy, dx)

        # Get current orientation (simplified - assumes z-axis rotation)
        current_orientation = self.current_pose.orientation
        # Convert quaternion to yaw (simplified calculation)
        siny_cosp = 2 * (current_orientation.w * current_orientation.z +
                         current_orientation.x * current_orientation.y)
        cosy_cosp = 1 - 2 * (current_orientation.y * current_orientation.y +
                             current_orientation.z * current_orientation.z)
        current_angle = math.atan2(siny_cosp, cosy_cosp)

        # Calculate angle difference
        angle_diff = desired_angle - current_angle
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Create velocity command
        cmd_vel = Twist()

        # Set angular velocity proportional to angle error
        cmd_vel.angular.z = max(-0.5, min(0.5, 2.0 * angle_diff))

        # Set linear velocity based on distance to goal and angle error
        if abs(angle_diff) < math.pi / 4:  # Only move forward if roughly aligned
            cmd_vel.linear.x = max(0.0, min(0.5, 0.5 * distance_to_goal))

        return cmd_vel

    def avoid_obstacles(self):
        """
        Calculate velocity command to avoid obstacles.
        """
        if self.latest_scan is None:
            return Twist()

        # Simple obstacle avoidance: turn away from closest obstacle
        ranges = self.latest_scan.ranges
        center_idx = len(ranges) // 2

        # Look for the clearest direction (longest ranges)
        left_ranges = ranges[:center_idx]
        right_ranges = ranges[center_idx:]

        left_clear = sum([1 for r in left_ranges if r > self.obstacle_threshold]) / len(left_ranges) if left_ranges else 0
        right_clear = sum([1 for r in right_ranges if r > self.obstacle_threshold]) / len(right_ranges) if right_ranges else 0

        cmd_vel = Twist()

        if left_clear > right_clear:
            # More clear space on the left, turn left
            cmd_vel.angular.z = 0.5
            cmd_vel.linear.x = 0.1  # Move slowly while turning
        else:
            # More clear space on the right, turn right
            cmd_vel.angular.z = -0.5
            cmd_vel.linear.x = 0.1  # Move slowly while turning

        return cmd_vel

    def control_loop(self):
        """
        Main control loop that makes decisions based on current state.
        """
        cmd_vel = Twist()

        if self.state == "IDLE":
            # Stay still
            cmd_vel = Twist()
        elif self.state == "NAVIGATING":
            cmd_vel = self.navigate_to_goal()
        elif self.state == "AVOIDING":
            cmd_vel = self.avoid_obstacles()
        elif self.state == "REACHED_GOAL":
            # Goal reached, stop and wait for new goal
            cmd_vel = Twist()
            # Publish status
            status_msg = String()
            status_msg.data = "GOAL_REACHED"
            self.status_publisher.publish(status_msg)

        # Publish the velocity command
        self.cmd_vel_publisher.publish(cmd_vel)

        # Publish current status
        status_msg = String()
        distance_to_goal = self.calculate_distance_to_goal()
        status_msg.data = f"State: {self.state}, Distance to goal: {distance_to_goal:.2f}m"
        self.status_publisher.publish(status_msg)

        # Log the command
        self.get_logger().info(f'AI Control: State={self.state}, '
                              f'Linear.x={cmd_vel.linear.x:.2f}, '
                              f'Angular.z={cmd_vel.angular.z:.2f}, '
                              f'Distance to goal={distance_to_goal:.2f}m')

    def reset_navigation(self):
        """
        Reset navigation state to start over.
        """
        self.state = "NAVIGATING"
        self.path_history = []
        self.get_logger().info('Navigation reset')


def main(args=None):
    """
    Main function to initialize and run the AI control node.
    """
    rclpy.init(args=args)

    ai_control_node = AIControlNode()

    try:
        rclpy.spin(ai_control_node)
    except KeyboardInterrupt:
        ai_control_node.get_logger().info('KeyboardInterrupt received, shutting down')
    finally:
        ai_control_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
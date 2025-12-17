#!/usr/bin/env python3
"""
AI Agent Example

This example demonstrates a simple AI agent that makes decisions based on
sensor input and sends commands to a robot through ROS 2.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import random
import math


class AIAgent(Node):
    """
    A simple AI agent that processes sensor data and makes navigation decisions.
    This agent subscribes to sensor data and publishes velocity commands.
    """

    def __init__(self):
        # Initialize the node with the name 'ai_agent'
        super().__init__('ai_agent')

        # Create a publisher for velocity commands to control the robot
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create a subscription to laser scan data from the robot
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Create a subscription to robot status information
        self.status_subscription = self.create_subscription(
            String,
            '/robot_status',
            self.status_callback,
            10
        )

        # Timer to periodically process decisions (10 Hz)
        self.timer = self.create_timer(0.1, self.decision_callback)

        # Internal state for the AI agent
        self.latest_scan = None
        self.robot_status = "IDLE"
        self.target_reached = False
        self.obstacle_detected = False
        self.avoidance_mode = False
        self.avoidance_timer = 0

        self.get_logger().info('AI Agent initialized')

    def scan_callback(self, msg):
        """
        Callback function to process laser scan data from the robot.

        Args:
            msg (LaserScan): Laser scan message containing distance measurements
        """
        self.latest_scan = msg
        self.process_scan_data()

    def status_callback(self, msg):
        """
        Callback function to process robot status updates.

        Args:
            msg (String): Status message from the robot
        """
        self.robot_status = msg.data
        self.get_logger().info(f'Robot status updated: {self.robot_status}')

    def process_scan_data(self):
        """
        Process the latest laser scan data to detect obstacles and plan navigation.
        """
        if self.latest_scan is None:
            return

        # Check for obstacles in the front 60-degree field
        min_angle_idx = int((math.pi/3) / self.latest_scan.angle_increment)  # 60 degrees
        max_angle_idx = len(self.latest_scan.ranges) - min_angle_idx

        front_ranges = self.latest_scan.ranges[min_angle_idx:max_angle_idx]
        min_distance = min(front_ranges) if front_ranges else float('inf')

        # Update obstacle detection status
        self.obstacle_detected = min_distance < 1.0  # Obstacle within 1 meter

        # Update avoidance timer
        if self.avoidance_mode:
            self.avoidance_timer -= 1
            if self.avoidance_timer <= 0:
                self.avoidance_mode = False

    def decision_callback(self):
        """
        Main decision-making function that runs periodically.
        """
        # Create a velocity command message
        cmd_vel = Twist()

        if self.target_reached:
            # Stop if target is reached
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
        elif self.avoidance_mode:
            # Execute avoidance maneuver
            cmd_vel.linear.x = 0.2  # Move forward slowly
            cmd_vel.angular.z = 0.5  # Turn right to avoid obstacle
        elif self.obstacle_detected:
            # Obstacle detected - initiate avoidance
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.8  # Turn left to avoid obstacle
            self.avoidance_mode = True
            self.avoidance_timer = 50  # 5 seconds at 10 Hz
        else:
            # No obstacles - move forward with occasional turns
            cmd_vel.linear.x = 0.5  # Move forward
            cmd_vel.angular.z = random.uniform(-0.2, 0.2)  # Slight random turn

        # Publish the velocity command
        self.cmd_vel_publisher.publish(cmd_vel)

        # Log the command for debugging
        self.get_logger().info(f'AI Decision: linear.x={cmd_vel.linear.x:.2f}, '
                              f'angular.z={cmd_vel.angular.z:.2f}, '
                              f'obstacle={self.obstacle_detected}, '
                              f'avoidance={self.avoidance_mode}')


def main(args=None):
    """
    Main function to initialize and run the AI agent node.
    """
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create an instance of the AIAgent
    ai_agent = AIAgent()

    try:
        # Start spinning the node to process callbacks and make decisions
        rclpy.spin(ai_agent)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        ai_agent.get_logger().info('KeyboardInterrupt received, shutting down')
    finally:
        # Clean up and destroy the node
        ai_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
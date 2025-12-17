#!/usr/bin/env python3
"""
Controller Bridge Example

This example demonstrates a bridge node that translates between AI agent
commands and robot controller interfaces. It subscribes to AI agent commands
and publishes them in a format suitable for robot controllers.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import math


class ControllerBridge(Node):
    """
    A bridge node that translates between AI agent commands and robot controllers.
    This node receives high-level commands from AI agents and translates them
    to low-level control commands for the robot hardware.
    """

    def __init__(self):
        # Initialize the node with the name 'controller_bridge'
        super().__init__('controller_bridge')

        # Subscribe to AI agent commands (Twist messages)
        self.ai_cmd_subscriber = self.create_subscription(
            Twist,
            '/ai_cmd_vel',
            self.ai_command_callback,
            10
        )

        # Publish to robot controller (Twist messages)
        self.robot_cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribe to robot status for feedback to AI agent
        self.robot_status_subscriber = self.create_subscription(
            String,
            '/robot_status_raw',
            self.robot_status_callback,
            10
        )

        # Publish processed status to AI agent
        self.ai_status_publisher = self.create_publisher(String, '/robot_status', 10)

        # Subscribe to raw sensor data
        self.raw_scan_subscriber = self.create_subscription(
            LaserScan,
            '/raw_scan',
            self.raw_scan_callback,
            10
        )

        # Publish processed sensor data to AI agent
        self.processed_scan_publisher = self.create_publisher(LaserScan, '/scan', 10)

        # Internal state
        self.last_ai_command = None
        self.robot_status = "UNKNOWN"
        self.last_scan = None

        # Timer for status updates (1 Hz)
        self.status_timer = self.create_timer(1.0, self.status_update_callback)

        self.get_logger().info('Controller Bridge initialized')

    def ai_command_callback(self, msg):
        """
        Callback function to process commands from the AI agent.

        Args:
            msg (Twist): Velocity command from the AI agent
        """
        self.last_ai_command = msg
        self.get_logger().info(f'Received AI command: linear.x={msg.linear.x:.2f}, '
                              f'angular.z={msg.angular.z:.2f}')

        # Apply any necessary transformations or safety checks
        processed_cmd = self.process_command(msg)

        # Publish the command to the robot controller
        self.robot_cmd_publisher.publish(processed_cmd)

        # Log the processed command
        self.get_logger().info(f'Sent processed command to robot: '
                              f'linear.x={processed_cmd.linear.x:.2f}, '
                              f'angular.z={processed_cmd.angular.z:.2f}')

    def robot_status_callback(self, msg):
        """
        Callback function to process raw robot status.

        Args:
            msg (String): Raw status message from the robot
        """
        self.robot_status = msg.data
        self.get_logger().info(f'Received robot status: {self.robot_status}')

    def raw_scan_callback(self, msg):
        """
        Callback function to process raw sensor data.

        Args:
            msg (LaserScan): Raw laser scan data from the robot
        """
        self.last_scan = msg
        # Process the scan data if needed
        processed_scan = self.process_scan(msg)
        self.processed_scan_publisher.publish(processed_scan)

    def process_command(self, cmd):
        """
        Process and validate AI agent commands before sending to robot.

        Args:
            cmd (Twist): Raw command from AI agent

        Returns:
            Twist: Processed command safe for robot execution
        """
        processed_cmd = Twist()

        # Apply safety limits
        max_linear = 1.0  # m/s
        max_angular = 1.0  # rad/s

        # Limit linear velocity
        if abs(cmd.linear.x) > max_linear:
            processed_cmd.linear.x = max_linear if cmd.linear.x > 0 else -max_linear
            self.get_logger().warn(f'Linear velocity limited from {cmd.linear.x:.2f} to {processed_cmd.linear.x:.2f}')
        else:
            processed_cmd.linear.x = cmd.linear.x

        # Limit angular velocity
        if abs(cmd.angular.z) > max_angular:
            processed_cmd.angular.z = max_angular if cmd.angular.z > 0 else -max_angular
            self.get_logger().warn(f'Angular velocity limited from {cmd.angular.z:.2f} to {processed_cmd.angular.z:.2f}')
        else:
            processed_cmd.angular.z = cmd.angular.z

        # Additional safety checks could be added here
        # For example: collision avoidance based on sensor data

        return processed_cmd

    def process_scan(self, scan):
        """
        Process raw laser scan data for AI agent consumption.

        Args:
            scan (LaserScan): Raw laser scan data

        Returns:
            LaserScan: Processed laser scan data
        """
        # For this example, we'll just return the scan as-is
        # In a real application, you might filter or transform the data
        return scan

    def status_update_callback(self):
        """
        Periodically publish status updates to the AI agent.
        """
        if self.robot_status != "UNKNOWN":
            status_msg = String()
            status_msg.data = self.robot_status
            self.ai_status_publisher.publish(status_msg)
            self.get_logger().info(f'Published status to AI: {self.robot_status}')


def main(args=None):
    """
    Main function to initialize and run the controller bridge node.
    """
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create an instance of the ControllerBridge
    controller_bridge = ControllerBridge()

    try:
        # Start spinning the node to process callbacks
        rclpy.spin(controller_bridge)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        controller_bridge.get_logger().info('KeyboardInterrupt received, shutting down')
    finally:
        # Clean up and destroy the node
        controller_bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
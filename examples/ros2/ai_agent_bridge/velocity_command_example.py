#!/usr/bin/env python3
"""
Velocity Command Example

This example demonstrates how to publish velocity commands from an AI agent
to control a robot's movement. It shows the proper way to create and send
Twist messages for robot navigation.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time


class VelocityCommandExample(Node):
    """
    Example node that demonstrates publishing velocity commands.
    This simulates how an AI agent might send movement commands to a robot.
    """

    def __init__(self):
        super().__init__('velocity_command_example')

        # Publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Publisher for status messages
        self.status_publisher = self.create_publisher(String, '/ai_status', 10)

        # Timer for sending commands (10 Hz)
        self.timer = self.create_timer(0.1, self.execute_behavior)

        # Internal state
        self.behavior_state = 0  # State machine for different behaviors
        self.state_timer = 0     # Counter for state duration
        self.max_state_time = 50 # 5 seconds at 10 Hz

        self.get_logger().info('Velocity Command Example initialized')

    def execute_behavior(self):
        """
        Execute different movement behaviors based on the current state.
        """
        cmd_vel = Twist()

        if self.behavior_state == 0:
            # Move forward
            cmd_vel.linear.x = 0.5
            cmd_vel.angular.z = 0.0
            self.publish_command(cmd_vel, "Moving forward")
        elif self.behavior_state == 1:
            # Turn right
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = -0.5
            self.publish_command(cmd_vel, "Turning right")
        elif self.behavior_state == 2:
            # Move forward and slightly left
            cmd_vel.linear.x = 0.3
            cmd_vel.angular.z = 0.2
            self.publish_command(cmd_vel, "Moving forward-left")
        elif self.behavior_state == 3:
            # Stop
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.publish_command(cmd_vel, "Stopping")

        # Update state timer
        self.state_timer += 1

        # Transition to next state after timeout
        if self.state_timer >= self.max_state_time:
            self.state_timer = 0
            self.behavior_state = (self.behavior_state + 1) % 4

    def publish_command(self, cmd_vel, status_msg):
        """
        Publish the velocity command and status message.

        Args:
            cmd_vel (Twist): The velocity command to publish
            status_msg (str): Status message to publish
        """
        # Publish the velocity command
        self.cmd_vel_publisher.publish(cmd_vel)

        # Publish the status message
        status = String()
        status.data = status_msg
        self.status_publisher.publish(status)

        # Log the command
        self.get_logger().info(f'Command: linear.x={cmd_vel.linear.x:.2f}, '
                              f'angular.z={cmd_vel.angular.z:.2f} - {status_msg}')


def main(args=None):
    """
    Main function to initialize and run the velocity command example.
    """
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create an instance of the VelocityCommandExample
    velocity_example = VelocityCommandExample()

    try:
        # Start spinning the node to execute behaviors
        rclpy.spin(velocity_example)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        velocity_example.get_logger().info('KeyboardInterrupt received, shutting down')
    finally:
        # Clean up and destroy the node
        velocity_example.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
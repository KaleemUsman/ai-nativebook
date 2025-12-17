#!/usr/bin/env python3
"""
Minimal ROS 2 node example

This example demonstrates the basic structure of a ROS 2 node using rclpy.
It creates a simple node that logs a message periodically.
"""

import rclpy
from rclpy.node import Node


class MinimalNode(Node):
    """
    A minimal ROS 2 node for demonstration purposes.
    """

    def __init__(self):
        # Initialize the node with the name 'minimal_node'
        super().__init__('minimal_node')

        # Create a timer that calls the timer_callback method every 0.5 seconds
        self.timer = self.create_timer(0.5, self.timer_callback)

        # Counter to track how many times the callback has been executed
        self.counter = 0

        # Log a message indicating the node has been initialized
        self.get_logger().info('Minimal node initialized')

    def timer_callback(self):
        """
        Callback method executed by the timer.
        """
        self.get_logger().info(f'Timer callback executed {self.counter} times')
        self.counter += 1


def main(args=None):
    """
    Main function to initialize and run the ROS 2 node.
    """
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create an instance of the MinimalNode
    minimal_node = MinimalNode()

    try:
        # Start spinning the node to process callbacks
        rclpy.spin(minimal_node)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        minimal_node.get_logger().info('KeyboardInterrupt received, shutting down')
    finally:
        # Clean up and destroy the node
        minimal_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
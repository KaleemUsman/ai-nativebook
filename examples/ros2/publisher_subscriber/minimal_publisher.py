#!/usr/bin/env python3
"""
Minimal publisher example

This example demonstrates how to create a publisher node that sends messages
to a topic using the ROS 2 publisher-subscriber pattern.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):
    """
    A minimal publisher node that sends messages to a topic.
    """

    def __init__(self):
        # Initialize the node with the name 'minimal_publisher'
        super().__init__('minimal_publisher')

        # Create a publisher that will publish String messages to the 'topic' topic
        # The second parameter (10) is the queue size for the message queue
        self.publisher_ = self.create_publisher(String, 'topic', 10)

        # Example with custom QoS profile
        # from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        # qos_profile = QoSProfile(
        #     depth=10,
        #     reliability=ReliabilityPolicy.RELIABLE,
        #     durability=DurabilityPolicy.VOLATILE
        # )
        # self.publisher_ = self.create_publisher(String, 'topic', qos_profile)

        # Create a timer that triggers the timer_callback method every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter to include in the published messages
        self.i = 0

        # Log a message indicating the publisher has been initialized
        self.get_logger().info('Minimal publisher initialized')

    def timer_callback(self):
        """
        Callback method executed by the timer to publish messages.
        """
        # Create a String message
        msg = String()

        # Set the message data
        msg.data = f'Hello World: {self.i}'

        # Publish the message to the topic
        self.publisher_.publish(msg)

        # Log the published message
        self.get_logger().info(f'Publishing: "{msg.data}"')

        # Increment the counter
        self.i += 1


def main(args=None):
    """
    Main function to initialize and run the publisher node.
    """
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create an instance of the MinimalPublisher
    minimal_publisher = MinimalPublisher()

    try:
        # Start spinning the node to process callbacks
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        minimal_publisher.get_logger().info('KeyboardInterrupt received, shutting down')
    finally:
        # Clean up and destroy the node
        minimal_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
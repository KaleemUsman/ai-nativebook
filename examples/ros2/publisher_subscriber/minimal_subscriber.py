#!/usr/bin/env python3
"""
Minimal subscriber example

This example demonstrates how to create a subscriber node that receives messages
from a topic using the ROS 2 publisher-subscriber pattern.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):
    """
    A minimal subscriber node that receives messages from a topic.
    """

    def __init__(self):
        # Initialize the node with the name 'minimal_subscriber'
        super().__init__('minimal_subscriber')

        # Create a subscription that will receive String messages from the 'topic' topic
        # The callback method message_callback will be executed when a message is received
        # The second parameter (10) is the queue size for the message queue
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.message_callback,
            10)

        # Example with custom QoS profile
        # from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        # qos_profile = QoSProfile(
        #     depth=10,
        #     reliability=ReliabilityPolicy.RELIABLE,
        #     durability=DurabilityPolicy.VOLATILE
        # )
        # self.subscription = self.create_subscription(
        #     String,
        #     'topic',
        #     self.message_callback,
        #     qos_profile)

        # Prevent unused variable warning
        self.subscription  # type: ignore

        # Log a message indicating the subscriber has been initialized
        self.get_logger().info('Minimal subscriber initialized')

    def message_callback(self, msg):
        """
        Callback method executed when a message is received from the topic.

        Args:
            msg (String): The received message
        """
        # Log the received message
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """
    Main function to initialize and run the subscriber node.
    """
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create an instance of the MinimalSubscriber
    minimal_subscriber = MinimalSubscriber()

    try:
        # Start spinning the node to process callbacks
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        minimal_subscriber.get_logger().info('KeyboardInterrupt received, shutting down')
    finally:
        # Clean up and destroy the node
        minimal_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
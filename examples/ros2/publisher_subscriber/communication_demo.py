#!/usr/bin/env python3
"""
Communication patterns demonstration

This example demonstrates both publisher-subscriber and client-service patterns
in a single application to show how they can work together.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from example_interfaces.srv import AddTwoInts


class CommunicationDemo(Node):
    """
    A demonstration node that uses both publisher-subscriber and client-service patterns.
    """

    def __init__(self):
        super().__init__('communication_demo')

        # Publisher for sending messages
        self.publisher_ = self.create_publisher(String, 'demo_topic', 10)

        # Service client for making requests
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Timer to periodically publish messages
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.i = 0

        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.request = AddTwoInts.Request()
        self.get_logger().info('Communication demo node initialized')

    def timer_callback(self):
        """
        Timer callback that publishes a message and makes a service request.
        """
        # Publish a message
        msg = String()
        msg.data = f'Demo message {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')

        # Make a service request
        self.request.a = self.i
        self.request.b = self.i * 2

        # Call the service asynchronously
        future = self.cli.call_async(self.request)
        future.add_done_callback(self.service_response_callback)

        self.i += 1

    def service_response_callback(self, future):
        """
        Callback for handling the service response.
        """
        try:
            response = future.result()
            self.get_logger().info(f'Service response: {response.sum}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


def main(args=None):
    """
    Main function to initialize and run the communication demo node.
    """
    rclpy.init(args=args)

    communication_demo = CommunicationDemo()

    try:
        rclpy.spin(communication_demo)
    except KeyboardInterrupt:
        communication_demo.get_logger().info('KeyboardInterrupt received, shutting down')
    finally:
        communication_demo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
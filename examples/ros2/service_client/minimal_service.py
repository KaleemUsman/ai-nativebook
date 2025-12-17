#!/usr/bin/env python3
"""
Minimal service example

This example demonstrates how to create a service server that responds to requests
using the ROS 2 service-client pattern.
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MinimalService(Node):
    """
    A minimal service server that responds to requests to add two integers.
    """

    def __init__(self):
        # Initialize the node with the name 'minimal_service'
        super().__init__('minimal_service')

        # Create a service that will use the add_two_ints_callback function to process requests
        # The service is named 'add_two_ints' and uses the AddTwoInts service type
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

        # Log a message indicating the service has been initialized
        self.get_logger().info('Minimal service server initialized')

    def add_two_ints_callback(self, request, response):
        """
        Callback function that processes service requests.

        Args:
            request (AddTwoInts.Request): The request message containing two integers
            response (AddTwoInts.Response): The response message to be filled with the sum

        Returns:
            AddTwoInts.Response: The response with the calculated sum
        """
        # Calculate the sum of the two integers in the request
        response.sum = request.a + request.b

        # Log the request and response
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}\n'
                              f'Sending back response: [{response.sum}]')

        # Return the response (the return value is optional but good practice)
        return response


def main(args=None):
    """
    Main function to initialize and run the service node.
    """
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create an instance of the MinimalService
    minimal_service = MinimalService()

    try:
        # Start spinning the node to process service requests
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        minimal_service.get_logger().info('KeyboardInterrupt received, shutting down')
    finally:
        # Clean up and destroy the node
        minimal_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
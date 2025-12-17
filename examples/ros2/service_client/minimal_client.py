#!/usr/bin/env python3
"""
Minimal client example

This example demonstrates how to create a client that sends requests to a service
using the ROS 2 service-client pattern.
"""

import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MinimalClient(Node):
    """
    A minimal client that sends requests to the add_two_ints service.
    """

    def __init__(self):
        # Initialize the node with the name 'minimal_client'
        super().__init__('minimal_client')

        # Create a client for the 'add_two_ints' service using the AddTwoInts service type
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Create a request object
        self.request = AddTwoInts.Request()

    def send_request(self, a, b):
        """
        Send a request to the service to add two integers.

        Args:
            a (int): First integer
            b (int): Second integer

        Returns:
            Future: A future object that will contain the response when completed
        """
        # Set the values in the request
        self.request.a = a
        self.request.b = b

        # Call the service asynchronously and return the future
        self.future = self.cli.call_async(self.request)
        return self.future


def main(args=None):
    """
    Main function to initialize and run the client node.
    """
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create an instance of the MinimalClient
    minimal_client = MinimalClient()

    # Check if command line arguments are provided
    if len(sys.argv) != 3:
        print('Usage: python3 minimal_client.py <int1> <int2>')
        print('Example: python3 minimal_client.py 2 3')
        sys.exit(1)

    # Parse the integer arguments
    try:
        integer1 = int(sys.argv[1])
        integer2 = int(sys.argv[2])
    except ValueError:
        print('Please provide valid integers as arguments')
        sys.exit(1)

    # Send the request to add the two integers
    future = minimal_client.send_request(integer1, integer2)

    try:
        # Spin until the future is complete (i.e., the response is received)
        rclpy.spin_until_future_complete(minimal_client, future)

        # Check if the service call was successful
        if future.result() is not None:
            # Print the result
            response = future.result()
            print(f'Result of {integer1} + {integer2} = {response.sum}')
        else:
            # Print an error message if the service call failed
            minimal_client.get_logger().error('Service call failed')

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        minimal_client.get_logger().info('KeyboardInterrupt received')
    finally:
        # Clean up and destroy the node
        minimal_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
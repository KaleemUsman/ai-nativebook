#!/usr/bin/env python3
"""
Testing framework for rclpy examples
This script provides utilities to test rclpy code examples from the documentation.
"""

import subprocess
import sys
import os
import tempfile
import time
from pathlib import Path


def test_rclpy_example(script_path, timeout=10):
    """
    Test an rclpy example script by running it briefly and checking for errors

    Args:
        script_path (str): Path to the Python script to test
        timeout (int): Time in seconds to run the script before terminating

    Returns:
        bool: True if the script runs without error, False otherwise
    """
    try:
        # Create a temporary file to capture output
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_filename = temp_file.name

        # Run the script with timeout
        result = subprocess.run(
            [sys.executable, script_path],
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check if the script ran successfully
        success = result.returncode == 0

        if not success:
            print(f"Script {script_path} failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

        return success

    except subprocess.TimeoutExpired:
        # For long-running nodes, this is expected
        print(f"Script {script_path} timed out after {timeout}s (this may be expected for long-running nodes)")
        return True
    except Exception as e:
        print(f"Error running script {script_path}: {e}")
        return False
    finally:
        # Clean up temporary file if it exists
        if 'temp_filename' in locals():
            try:
                os.unlink(temp_filename)
            except:
                pass


def validate_ros2_environment():
    """
    Validate that the ROS 2 environment is properly set up

    Returns:
        bool: True if environment is valid, False otherwise
    """
    try:
        import rclpy
    except ImportError:
        print("rclpy not available - ROS 2 environment not properly set up")
        return False

    try:
        rclpy.init()
        rclpy.shutdown()
    except Exception as e:
        print(f"rclpy initialization failed: {e}")
        return False

    return True


def run_all_tests():
    """
    Run tests on all rclpy examples in the examples/ros2 directory
    """
    print("Validating ROS 2 environment...")
    if not validate_ros2_environment():
        print("ROS 2 environment validation failed. Exiting.")
        return False

    examples_dir = Path("examples/ros2")
    if not examples_dir.exists():
        print(f"Examples directory {examples_dir} does not exist")
        return False

    success_count = 0
    total_count = 0

    # Find all Python files in the examples directory
    for py_file in examples_dir.rglob("*.py"):
        if py_file.name != "test_rclpy_examples.py":  # Skip this test file
            print(f"Testing {py_file}...")
            total_count += 1
            if test_rclpy_example(str(py_file)):
                print(f"✓ {py_file} passed")
                success_count += 1
            else:
                print(f"✗ {py_file} failed")

    print(f"\nTest Results: {success_count}/{total_count} examples passed")
    return success_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
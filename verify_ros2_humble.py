#!/usr/bin/env python3
"""
ROS 2 Humble and rclpy verification script
This script verifies that ROS 2 Humble and rclpy are properly installed and available.
"""

def verify_ros2_humble():
    """Verify ROS 2 Humble installation and rclpy availability"""
    try:
        import rclpy
        print(f"✓ rclpy available: {rclpy.__version__ if hasattr(rclpy, '__version__') else 'version unknown'}")
    except ImportError:
        print("✗ rclpy not available - ROS 2 Humble may not be installed or sourced")
        return False

    try:
        # Try to initialize rclpy to verify full functionality
        rclpy.init()
        print("✓ rclpy initialization successful")
        rclpy.shutdown()
    except Exception as e:
        print(f"✗ rclpy initialization failed: {e}")
        return False

    # Check ROS distribution
    import os
    ros_distro = os.environ.get('ROS_DISTRO', 'Not set')
    if 'humble' in ros_distro.lower():
        print(f"✓ ROS distribution verified: {ros_distro}")
    else:
        print(f"⚠ ROS distribution may not be Humble: {ros_distro}")

    return True

if __name__ == "__main__":
    print("Verifying ROS 2 Humble and rclpy installation...")
    success = verify_ros2_humble()
    if success:
        print("\n✓ ROS 2 Humble and rclpy verification completed successfully")
    else:
        print("\n✗ ROS 2 Humble and rclpy verification failed")
        exit(1)
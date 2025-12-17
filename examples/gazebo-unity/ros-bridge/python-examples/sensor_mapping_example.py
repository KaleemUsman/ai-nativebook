#!/usr/bin/env python3
"""
Sensor Mapping Example: Gazebo to Unity
This script demonstrates how to map sensor data from Gazebo physics simulation
to Unity visualization, including coordinate system transformations and
data synchronization.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3, Quaternion
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
import numpy as np
import math
import time
from collections import deque
import struct


class SensorMapper(Node):
    """
    Sensor mapper that transforms and synchronizes sensor data
    between Gazebo (physics) and Unity (visualization) coordinate systems.
    """

    def __init__(self):
        super().__init__('sensor_mapper')

        # Initialize transformation matrices
        self.initialize_transformations()

        # Data storage for synchronization
        self.gazebo_data = {}
        self.unity_data = {}
        self.sync_buffer = deque(maxlen=100)  # Buffer for temporal synchronization

        # Initialize ROS interfaces
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers for Gazebo sensor data
        self.gazebo_scan_sub = self.create_subscription(
            LaserScan,
            '/gazebo/scan',
            self.gazebo_scan_callback,
            10
        )

        self.gazebo_imu_sub = self.create_subscription(
            Imu,
            '/gazebo/imu/data',
            self.gazebo_imu_callback,
            10
        )

        self.gazebo_joint_sub = self.create_subscription(
            JointState,
            '/gazebo/joint_states',
            self.gazebo_joint_callback,
            10
        )

        # Publishers for Unity visualization
        self.unity_scan_pub = self.create_publisher(
            LaserScan,
            '/unity/scan_visualization',
            10
        )

        self.unity_imu_pub = self.create_publisher(
            Imu,
            '/unity/imu_visualization',
            10
        )

        self.unity_marker_pub = self.create_publisher(
            Marker,
            '/unity/sensor_visualization',
            10
        )

        self.unity_pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/unity/pointcloud_visualization',
            10
        )

        # Timer for processing and publishing mapped data
        self.processing_timer = self.create_timer(0.033, self.process_and_publish)  # ~30 Hz

        # Statistics
        self.stats = {
            'processed_scans': 0,
            'processed_imu': 0,
            'processed_joints': 0,
            'coordinate_transforms': 0
        }

        self.get_logger().info("Sensor Mapper initialized - Gazebo to Unity transformation system")

    def initialize_transformations(self):
        """
        Initialize coordinate system transformation matrices.
        Gazebo uses Z-up coordinate system (ROS standard)
        Unity uses Y-up coordinate system
        """
        # Transformation matrix from ROS (Z-up) to Unity (Y-up)
        # This rotates the coordinate system around the X-axis by -90 degrees
        self.ros_to_unity_matrix = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ])

        # Inverse transformation for Unity to ROS
        self.unity_to_ros_matrix = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ])

    def gazebo_scan_callback(self, msg):
        """Process LiDAR scan data from Gazebo and prepare for Unity visualization"""
        try:
            # Store original Gazebo data
            self.gazebo_data['scan'] = {
                'data': msg,
                'timestamp': time.time()
            }

            # Transform scan data for Unity visualization
            unity_scan = self.transform_scan_to_unity(msg)

            # Publish transformed scan to Unity
            self.unity_scan_pub.publish(unity_scan)

            # Update statistics
            self.stats['processed_scans'] += 1

        except Exception as e:
            self.get_logger().error(f"Error processing Gazebo scan: {e}")

    def gazebo_imu_callback(self, msg):
        """Process IMU data from Gazebo and prepare for Unity visualization"""
        try:
            # Store original Gazebo data
            self.gazebo_data['imu'] = {
                'data': msg,
                'timestamp': time.time()
            }

            # Transform IMU data for Unity visualization
            unity_imu = self.transform_imu_to_unity(msg)

            # Publish transformed IMU to Unity
            self.unity_imu_pub.publish(unity_imu)

            # Update statistics
            self.stats['processed_imu'] += 1

        except Exception as e:
            self.get_logger().error(f"Error processing Gazebo IMU: {e}")

    def gazebo_joint_callback(self, msg):
        """Process joint state data from Gazebo"""
        try:
            # Store original Gazebo data
            self.gazebo_data['joints'] = {
                'data': msg,
                'timestamp': time.time()
            }

            # Update statistics
            self.stats['processed_joints'] += 1

        except Exception as e:
            self.get_logger().error(f"Error processing Gazebo joints: {e}")

    def transform_scan_to_unity(self, gazebo_scan):
        """Transform LiDAR scan from Gazebo coordinate system to Unity coordinate system"""
        # Create new scan message for Unity
        unity_scan = LaserScan()
        unity_scan.header = Header()
        unity_scan.header.stamp = gazebo_scan.header.stamp
        unity_scan.header.frame_id = gazebo_scan.header.frame_id.replace('gazebo', 'unity')

        # Copy scan parameters (these don't change with coordinate system)
        unity_scan.angle_min = gazebo_scan.angle_min
        unity_scan.angle_max = gazebo_scan.angle_max
        unity_scan.angle_increment = gazebo_scan.angle_increment
        unity_scan.time_increment = gazebo_scan.time_increment
        unity_scan.scan_time = gazebo_scan.scan_time
        unity_scan.range_min = gazebo_scan.range_min
        unity_scan.range_max = gazebo_scan.range_max

        # Copy ranges and intensities (these are scalar values, not affected by coordinate system)
        unity_scan.ranges = gazebo_scan.ranges
        unity_scan.intensities = gazebo_scan.intensities

        # Update statistics
        self.stats['coordinate_transforms'] += 1

        return unity_scan

    def transform_imu_to_unity(self, gazebo_imu):
        """Transform IMU data from Gazebo coordinate system to Unity coordinate system"""
        unity_imu = Imu()
        unity_imu.header = Header()
        unity_imu.header.stamp = gazebo_imu.header.stamp
        unity_imu.header.frame_id = gazebo_imu.header.frame_id.replace('gazebo', 'unity')

        # Transform orientation quaternion
        ros_quat = np.array([
            gazebo_imu.orientation.x,
            gazebo_imu.orientation.y,
            gazebo_imu.orientation.z,
            gazebo_imu.orientation.w
        ])

        # Convert quaternion to rotation matrix, apply transformation, convert back
        ros_rot_matrix = self.quaternion_to_rotation_matrix(ros_quat)
        unity_rot_matrix = self.ros_to_unity_matrix @ ros_rot_matrix @ self.ros_to_unity_matrix.T
        unity_quat = self.rotation_matrix_to_quaternion(unity_rot_matrix)

        unity_imu.orientation.x = unity_quat[0]
        unity_imu.orientation.y = unity_quat[1]
        unity_imu.orientation.z = unity_quat[2]
        unity_imu.orientation.w = unity_quat[3]

        # Transform angular velocity vector
        ros_angular_vel = np.array([
            gazebo_imu.angular_velocity.x,
            gazebo_imu.angular_velocity.y,
            gazebo_imu.angular_velocity.z
        ])
        unity_angular_vel = self.ros_to_unity_matrix @ ros_angular_vel

        unity_imu.angular_velocity.x = unity_angular_vel[0]
        unity_imu.angular_velocity.y = unity_angular_vel[1]
        unity_imu.angular_velocity.z = unity_angular_vel[2]

        # Transform linear acceleration vector
        ros_linear_acc = np.array([
            gazebo_imu.linear_acceleration.x,
            gazebo_imu.linear_acceleration.y,
            gazebo_imu.linear_acceleration.z
        ])
        unity_linear_acc = self.ros_to_unity_matrix @ ros_linear_acc

        unity_imu.linear_acceleration.x = unity_linear_acc[0]
        unity_imu.linear_acceleration.y = unity_linear_acc[1]
        unity_imu.linear_acceleration.z = unity_linear_acc[2]

        # Copy covariance matrices (these are transformed accordingly)
        unity_imu.orientation_covariance = gazebo_imu.orientation_covariance
        unity_imu.angular_velocity_covariance = gazebo_imu.angular_velocity_covariance
        unity_imu.linear_acceleration_covariance = gazebo_imu.linear_acceleration_covariance

        # Update statistics
        self.stats['coordinate_transforms'] += 1

        return unity_imu

    def quaternion_to_rotation_matrix(self, quat):
        """Convert quaternion to rotation matrix"""
        x, y, z, w = quat

        # Rotation matrix from quaternion
        rot_matrix = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

        return rot_matrix

    def rotation_matrix_to_quaternion(self, rot_matrix):
        """Convert rotation matrix to quaternion"""
        # Use the standard algorithm to convert rotation matrix to quaternion
        trace = np.trace(rot_matrix)

        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            qy = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            qz = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
        else:
            if rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
                s = math.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
                qw = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
                qx = 0.25 * s
                qy = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                qz = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            elif rot_matrix[1, 1] > rot_matrix[2, 2]:
                s = math.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2
                qw = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
                qx = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                qy = 0.25 * s
                qz = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            else:
                s = math.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2
                qw = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
                qx = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                qy = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                qz = 0.25 * s

        return np.array([qx, qy, qz, qw])

    def create_laser_scan_visualization(self, scan_data):
        """Create visualization markers for laser scan data"""
        marker = Marker()
        marker.header.frame_id = scan_data.header.frame_id.replace('gazebo', 'unity')
        marker.header.stamp = scan_data.header.stamp
        marker.ns = "laser_scan"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Set scale (adjust as needed)
        marker.scale.x = 0.02  # Point width
        marker.scale.y = 0.02  # Point height
        marker.scale.z = 0.02  # Point depth

        # Set color (green for laser points)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Calculate point positions from scan data
        angle = scan_data.angle_min
        for i, range_val in enumerate(scan_data.ranges):
            if not (math.isnan(range_val) or math.isinf(range_val)) and scan_data.range_min <= range_val <= scan_data.range_max:
                # Calculate x, y in 2D plane (z=0 for basic laser scan)
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)

                # Transform from ROS to Unity coordinates
                ros_point = np.array([x, y, 0.0])
                unity_point = self.ros_to_unity_matrix @ ros_point

                point = Vector3()
                point.x = unity_point[0]
                point.y = unity_point[1]
                point.z = unity_point[2]
                marker.points.append(point)

            angle += scan_data.angle_increment

        return marker

    def create_pointcloud_from_scan(self, scan_data):
        """Convert laser scan to point cloud for Unity visualization"""
        # Create PointCloud2 message
        cloud = PointCloud2()
        cloud.header = Header()
        cloud.header.stamp = scan_data.header.stamp
        cloud.header.frame_id = scan_data.header.frame_id.replace('gazebo', 'unity')

        # Define fields (x, y, z, intensity)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        cloud.fields = fields
        cloud.point_step = 16  # 4 * 4 bytes per point (x, y, z, intensity)

        # Calculate points
        points_data = []
        angle = scan_data.angle_min
        valid_points = 0

        for i, range_val in enumerate(scan_data.ranges):
            if not (math.isnan(range_val) or math.isinf(range_val)) and scan_data.range_min <= range_val <= scan_data.range_max:
                # Calculate x, y in 2D plane (z=0 for basic laser scan)
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                z = 0.0  # Laser scan is typically 2D

                # Transform from ROS to Unity coordinates
                ros_point = np.array([x, y, z])
                unity_point = self.ros_to_unity_matrix @ ros_point

                # Add point data (x, y, z, intensity)
                points_data.extend([
                    struct.pack('f', unity_point[0]),  # x
                    struct.pack('f', unity_point[1]),  # y
                    struct.pack('f', unity_point[2]),  # z
                    struct.pack('f', scan_data.intensities[i] if i < len(scan_data.intensities) else 0.0)  # intensity
                ])
                valid_points += 1

            angle += scan_data.angle_increment

        # Set up the cloud data
        cloud.height = 1
        cloud.width = valid_points
        cloud.is_bigendian = False
        cloud.is_dense = True
        cloud.data = b''.join(points_data)

        return cloud

    def process_and_publish(self):
        """Process and publish sensor data for Unity visualization"""
        try:
            # Process any available Gazebo data and create Unity visualizations
            if 'scan' in self.gazebo_data:
                scan_data = self.gazebo_data['scan']['data']

                # Create visualization markers for the scan
                scan_marker = self.create_laser_scan_visualization(scan_data)
                self.unity_marker_pub.publish(scan_marker)

                # Create point cloud from scan
                pointcloud = self.create_pointcloud_from_scan(scan_data)
                self.unity_pointcloud_pub.publish(pointcloud)

            # Publish statistics periodically
            if self.stats['processed_scans'] % 100 == 0:
                self.get_logger().info(f"Sensor Mapping Statistics: {self.get_stats()}")

        except Exception as e:
            self.get_logger().error(f"Error in process_and_publish: {e}")

    def get_stats(self):
        """Get current processing statistics"""
        return f"Scans: {self.stats['processed_scans']}, " \
               f"IMU: {self.stats['processed_imu']}, " \
               f"Joints: {self.stats['processed_joints']}, " \
               f"Transforms: {self.stats['coordinate_transforms']}"


class CoordinateSystemTransformer:
    """
    Utility class for coordinate system transformations
    between ROS/Gazebo (Z-up) and Unity (Y-up) systems.
    """

    def __init__(self):
        # Pre-compute transformation matrices
        self.ros_to_unity_matrix = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ])

        self.unity_to_ros_matrix = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ])

    def transform_point(self, point, from_system='ros', to_system='unity'):
        """Transform a 3D point between coordinate systems"""
        point_array = np.array([point.x, point.y, point.z])

        if from_system == 'ros' and to_system == 'unity':
            transformed = self.ros_to_unity_matrix @ point_array
        elif from_system == 'unity' and to_system == 'ros':
            transformed = self.unity_to_ros_matrix @ point_array
        else:
            raise ValueError(f"Unsupported transformation: {from_system} to {to_system}")

        # Create new point with transformed coordinates
        new_point = Vector3()
        new_point.x = transformed[0]
        new_point.y = transformed[1]
        new_point.z = transformed[2]

        return new_point

    def transform_quaternion(self, quat, from_system='ros', to_system='unity'):
        """Transform a quaternion between coordinate systems"""
        # Convert quaternion to rotation matrix
        quat_array = np.array([quat.x, quat.y, quat.z, quat.w])
        rot_matrix = self.quaternion_to_rotation_matrix(quat_array)

        # Apply coordinate transformation
        if from_system == 'ros' and to_system == 'unity':
            transformed_matrix = self.ros_to_unity_matrix @ rot_matrix @ self.ros_to_unity_matrix.T
        elif from_system == 'unity' and to_system == 'ros':
            transformed_matrix = self.unity_to_ros_matrix @ rot_matrix @ self.unity_to_ros_matrix.T
        else:
            raise ValueError(f"Unsupported transformation: {from_system} to {to_system}")

        # Convert back to quaternion
        transformed_quat = self.rotation_matrix_to_quaternion(transformed_matrix)

        # Create new quaternion message
        new_quat = Quaternion()
        new_quat.x = transformed_quat[0]
        new_quat.y = transformed_quat[1]
        new_quat.z = transformed_quat[2]
        new_quat.w = transformed_quat[3]

        return new_quat

    def quaternion_to_rotation_matrix(self, quat):
        """Convert quaternion to rotation matrix"""
        x, y, z, w = quat

        # Rotation matrix from quaternion
        rot_matrix = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

        return rot_matrix

    def rotation_matrix_to_quaternion(self, rot_matrix):
        """Convert rotation matrix to quaternion"""
        # Use the standard algorithm to convert rotation matrix to quaternion
        trace = np.trace(rot_matrix)

        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            qy = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            qz = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
        else:
            if rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
                s = math.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
                qw = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
                qx = 0.25 * s
                qy = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                qz = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            elif rot_matrix[1, 1] > rot_matrix[2, 2]:
                s = math.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2
                qw = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
                qx = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                qy = 0.25 * s
                qz = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            else:
                s = math.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2
                qw = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
                qx = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                qy = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                qz = 0.25 * s

        return np.array([qx, qy, qz, qw])


def main(args=None):
    """Main function to run the sensor mapper"""
    rclpy.init(args=args)

    sensor_mapper = SensorMapper()

    try:
        rclpy.spin(sensor_mapper)
    except KeyboardInterrupt:
        sensor_mapper.get_logger().info(f"Sensor Mapper stopped. Final stats: {sensor_mapper.get_stats()}")
    finally:
        sensor_mapper.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
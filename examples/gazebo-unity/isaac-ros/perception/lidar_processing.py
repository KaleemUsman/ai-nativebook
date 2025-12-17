#!/usr/bin/env python3
"""
LiDAR Processing Node for Isaac ROS Perception Pipeline

This node processes LiDAR point cloud data for the humanoid robot's perception system,
performing filtering, segmentation, and obstacle detection.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, LaserScan
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
import struct
import threading
from typing import List, Tuple, Optional


class LidarProcessingNode(Node):
    """
    LiDAR processing node for the Isaac ROS perception pipeline.
    
    Processes 3D point cloud data to extract:
    - Ground plane detection
    - Obstacle clustering
    - Dynamic object tracking
    """
    
    def __init__(self):
        super().__init__('lidar_processing_node')
        
        # Declare parameters
        self.declare_parameter('min_range', 0.1)
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('min_height', -0.5)
        self.declare_parameter('max_height', 2.0)
        self.declare_parameter('ground_threshold', 0.15)
        self.declare_parameter('cluster_tolerance', 0.5)
        self.declare_parameter('min_cluster_size', 10)
        self.declare_parameter('max_cluster_size', 10000)
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('processing_rate', 10.0)
        
        # Get parameters
        self.min_range = self.get_parameter('min_range').value
        self.max_range = self.get_parameter('max_range').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.ground_threshold = self.get_parameter('ground_threshold').value
        self.cluster_tolerance = self.get_parameter('cluster_tolerance').value
        self.min_cluster_size = self.get_parameter('min_cluster_size').value
        self.max_cluster_size = self.get_parameter('max_cluster_size').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.processing_rate = self.get_parameter('processing_rate').value
        
        # Create QoS profile for sensor topics
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create subscriber for point cloud
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/lidar/points',
            self.pointcloud_callback,
            qos_profile
        )
        
        # Create publishers
        self.filtered_pub = self.create_publisher(
            PointCloud2,
            '/lidar/points_filtered',
            10
        )
        
        self.ground_pub = self.create_publisher(
            PointCloud2,
            '/lidar/ground_points',
            10
        )
        
        self.obstacles_pub = self.create_publisher(
            PointCloud2,
            '/lidar/obstacle_points',
            10
        )
        
        self.scan_pub = self.create_publisher(
            LaserScan,
            '/scan',
            10
        )
        
        # Initialize state
        self.latest_pointcloud = None
        self.pointcloud_lock = threading.Lock()
        
        # Create processing timer
        self.process_timer = self.create_timer(
            1.0 / self.processing_rate,
            self.process_pointcloud
        )
        
        self.get_logger().info('LiDAR Processing Node initialized')
    
    def pointcloud_callback(self, msg: PointCloud2):
        """Store the latest point cloud message for processing."""
        with self.pointcloud_lock:
            self.latest_pointcloud = msg
    
    def process_pointcloud(self):
        """Process the latest point cloud with filtering and segmentation."""
        with self.pointcloud_lock:
            if self.latest_pointcloud is None:
                return
            
            msg = self.latest_pointcloud
        
        try:
            # Parse point cloud data
            points = self.parse_pointcloud2(msg)
            
            if points is None or len(points) == 0:
                return
            
            # Apply range and height filtering
            filtered_points = self.filter_points(points)
            
            if len(filtered_points) == 0:
                return
            
            # Voxel downsampling for efficiency
            downsampled_points = self.voxel_downsample(filtered_points)
            
            # Ground plane segmentation
            ground_points, obstacle_points = self.segment_ground(downsampled_points)
            
            # Publish filtered point cloud
            filtered_msg = self.create_pointcloud2(filtered_points, msg.header)
            self.filtered_pub.publish(filtered_msg)
            
            # Publish ground and obstacle points
            if len(ground_points) > 0:
                ground_msg = self.create_pointcloud2(ground_points, msg.header)
                self.ground_pub.publish(ground_msg)
            
            if len(obstacle_points) > 0:
                obstacles_msg = self.create_pointcloud2(obstacle_points, msg.header)
                self.obstacles_pub.publish(obstacles_msg)
            
            # Convert to 2D laser scan for Nav2 compatibility
            scan_msg = self.pointcloud_to_laserscan(obstacle_points, msg.header)
            self.scan_pub.publish(scan_msg)
            
            self.get_logger().debug(
                f'Processed {len(points)} points: '
                f'{len(ground_points)} ground, {len(obstacle_points)} obstacles'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')
    
    def parse_pointcloud2(self, msg: PointCloud2) -> Optional[np.ndarray]:
        """Parse PointCloud2 message into numpy array of XYZ points."""
        if msg.width * msg.height == 0:
            return None
        
        # Determine point step and field offsets
        point_step = msg.point_step
        x_offset = y_offset = z_offset = None
        
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
        
        if x_offset is None or y_offset is None or z_offset is None:
            self.get_logger().warning('Point cloud missing XYZ fields')
            return None
        
        # Parse points
        num_points = msg.width * msg.height
        points = np.zeros((num_points, 3), dtype=np.float32)
        
        for i in range(num_points):
            offset = i * point_step
            points[i, 0] = struct.unpack_from('f', msg.data, offset + x_offset)[0]
            points[i, 1] = struct.unpack_from('f', msg.data, offset + y_offset)[0]
            points[i, 2] = struct.unpack_from('f', msg.data, offset + z_offset)[0]
        
        # Remove NaN and infinite values
        valid_mask = np.isfinite(points).all(axis=1)
        return points[valid_mask]
    
    def filter_points(self, points: np.ndarray) -> np.ndarray:
        """Apply range and height filtering to points."""
        # Calculate range (distance from origin in XY plane)
        ranges = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        
        # Apply filters
        range_mask = (ranges >= self.min_range) & (ranges <= self.max_range)
        height_mask = (points[:, 2] >= self.min_height) & (points[:, 2] <= self.max_height)
        
        return points[range_mask & height_mask]
    
    def voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """Downsample points using voxel grid filtering."""
        if len(points) == 0:
            return points
        
        # Compute voxel indices
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        
        # Use dictionary to group points by voxel
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(points[i])
        
        # Compute centroid of each voxel
        downsampled = np.array([
            np.mean(pts, axis=0) for pts in voxel_dict.values()
        ])
        
        return downsampled
    
    def segment_ground(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment ground plane from obstacle points using RANSAC-like approach."""
        if len(points) < 3:
            return np.array([]), points
        
        # Simple height-based ground segmentation
        # For production, use RANSAC plane fitting
        z_values = points[:, 2]
        
        # Estimate ground level as the lower percentile of heights
        ground_level = np.percentile(z_values, 10)
        
        # Classify points
        ground_mask = np.abs(z_values - ground_level) < self.ground_threshold
        
        ground_points = points[ground_mask]
        obstacle_points = points[~ground_mask]
        
        return ground_points, obstacle_points
    
    def create_pointcloud2(self, points: np.ndarray, header: Header) -> PointCloud2:
        """Create PointCloud2 message from numpy array."""
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True
        msg.is_bigendian = False
        msg.point_step = 12  # 3 floats * 4 bytes
        msg.row_step = msg.point_step * msg.width
        
        # Define fields
        from sensor_msgs.msg import PointField
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Pack data
        msg.data = points.astype(np.float32).tobytes()
        
        return msg
    
    def pointcloud_to_laserscan(self, points: np.ndarray, header: Header) -> LaserScan:
        """Convert 3D point cloud to 2D laser scan for Nav2 compatibility."""
        scan = LaserScan()
        scan.header = header
        scan.header.frame_id = 'base_scan'
        
        # Configure scan parameters
        scan.angle_min = -np.pi
        scan.angle_max = np.pi
        scan.angle_increment = np.pi / 180.0  # 1 degree resolution
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = float(self.min_range)
        scan.range_max = float(self.max_range)
        
        num_readings = int((scan.angle_max - scan.angle_min) / scan.angle_increment)
        ranges = [float('inf')] * num_readings
        
        if len(points) > 0:
            # Project points to 2D and compute ranges
            for point in points:
                angle = np.arctan2(point[1], point[0])
                distance = np.sqrt(point[0]**2 + point[1]**2)
                
                # Convert angle to index
                index = int((angle - scan.angle_min) / scan.angle_increment)
                if 0 <= index < num_readings:
                    if distance < ranges[index]:
                        ranges[index] = distance
        
        scan.ranges = ranges
        scan.intensities = []
        
        return scan


def main(args=None):
    rclpy.init(args=args)
    
    lidar_processing_node = LidarProcessingNode()
    
    try:
        rclpy.spin(lidar_processing_node)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_processing_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

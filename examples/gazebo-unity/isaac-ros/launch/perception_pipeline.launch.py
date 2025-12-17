#!/usr/bin/env python3

"""
Isaac ROS Perception Pipeline Launch File

This launch file configures and starts the Isaac ROS perception pipeline
with VSLAM and sensor fusion for humanoid robot applications.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import yaml


def generate_launch_description():
    """Generate the launch description for the perception pipeline."""

    # Declare launch arguments
    namespace = LaunchConfiguration('namespace', default='humanoid_robot')
    use_composition = LaunchConfiguration('use_composition', default='False')
    enable_vslam = LaunchConfiguration('enable_vslam', default='True')
    enable_sensor_fusion = LaunchConfiguration('enable_sensor_fusion', default='True')
    config_dir = LaunchConfiguration('config_dir',
        default=os.path.join(get_package_share_directory('isaac_ai_brain'), 'config'))

    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='humanoid_robot',
        description='Namespace for the perception nodes'
    )

    use_composition_arg = DeclareLaunchArgument(
        'use_composition',
        default_value='False',
        description='Whether to use composed nodes'
    )

    enable_vslam_arg = DeclareLaunchArgument(
        'enable_vslam',
        default_value='True',
        description='Whether to enable VSLAM processing'
    )

    enable_sensor_fusion_arg = DeclareLaunchArgument(
        'enable_sensor_fusion',
        default_value='True',
        description='Whether to enable sensor fusion'
    )

    config_dir_arg = DeclareLaunchArgument(
        'config_dir',
        default_value=os.path.join(get_package_share_directory('isaac_ai_brain'), 'config'),
        description='Directory containing configuration files'
    )

    # Load configuration files
    vslam_config_path = os.path.join(config_dir.perform({}), 'vslam_params.yaml')
    sensor_fusion_config_path = os.path.join(config_dir.perform({}), 'sensor_fusion.yaml')

    # Define the perception pipeline nodes
    perception_nodes = []

    # VSLAM node
    vslam_node = ComposableNode(
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        name='visual_slam',
        namespace=namespace,
        parameters=[vslam_config_path],
        remappings=[
            ('/visual_slam/image', '/camera/rgb/image_rect_color'),
            ('/visual_slam/camera_info', '/camera/rgb/camera_info'),
            ('/visual_slam/imu', '/imu/data'),
            ('/visual_slam/pose', 'visual_slam/pose'),
            ('/visual_slam/trajectory', 'visual_slam/trajectory'),
        ],
        condition=enable_vslam
    )

    # Camera processing node
    camera_processing_node = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        name='camera_rectify',
        namespace=namespace,
        parameters=[{
            'output_width': 640,
            'output_height': 480,
        }],
        remappings=[
            ('image_raw', '/camera/rgb/image_raw'),
            ('camera_info', '/camera/rgb/camera_info'),
            ('image_rect', '/camera/rgb/image_rect_color'),
        ]
    )

    # LiDAR processing node
    lidar_processing_node = ComposableNode(
        package='isaac_ros_pcl',
        plugin='nvidia::isaac_ros::pcl::FilterTransformNode',
        name='lidar_filter',
        namespace=namespace,
        parameters=[{
            'input_frame': 'base_link',
            'output_frame': 'base_link',
        }],
        remappings=[
            ('points_in', '/lidar/points'),
            ('points_out', '/lidar/points_filtered'),
        ]
    )

    # IMU processing node
    imu_processing_node = ComposableNode(
        package='imu_filter_madgwick',
        plugin='ImuFilterMadgwickNode',
        name='imu_filter',
        namespace=namespace,
        parameters=[{
            'use_mag': False,
            'publish_tf': False,
            'world_frame': 'enu',
        }],
        remappings=[
            ('imu/data', '/imu/data'),
            ('imu/data_filtered', '/imu/data_filtered'),
        ]
    )

    # Sensor fusion node
    sensor_fusion_node = ComposableNode(
        package='isaac_ros_pointcloud_utils',
        plugin='nvidia::isaac_ros::pointcloud_utils::FusionNode',
        name='sensor_fusion',
        namespace=namespace,
        parameters=[sensor_fusion_config_path],
        remappings=[
            ('camera/image', '/camera/rgb/image_rect_color'),
            ('lidar/points', '/lidar/points_filtered'),
            ('imu/data', '/imu/data_filtered'),
            ('fused/pose', 'sensor_fusion/pose'),
            ('fused/pointcloud', 'sensor_fusion/pointcloud'),
        ],
        condition=enable_sensor_fusion
    )

    # Add nodes to the list
    perception_nodes.extend([
        camera_processing_node,
        lidar_processing_node,
        imu_processing_node,
        vslam_node,
        sensor_fusion_node
    ])

    # Create container for composable nodes if composition is enabled
    perception_container = ComposableNodeContainer(
        name='perception_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=perception_nodes,
        output='both',
        condition=use_composition
    )

    # Create individual nodes if composition is disabled
    perception_individual_nodes = []
    for node_desc in perception_nodes:
        node = Node(
            package=node_desc.package,
            plugin=node_desc.plugin,
            name=node_desc.name,
            namespace=node_desc.namespace,
            parameters=node_desc.parameters,
            remappings=node_desc.remappings if node_desc.remappings else [],
            condition=node_desc.condition if node_desc.condition else None
        )
        perception_individual_nodes.append(node)

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(namespace_arg)
    ld.add_action(use_composition_arg)
    ld.add_action(enable_vslam_arg)
    ld.add_action(enable_sensor_fusion_arg)
    ld.add_action(config_dir_arg)

    # Add nodes based on composition setting
    ld.add_action(perception_container)
    for node in perception_individual_nodes:
        ld.add_action(node)

    return ld


def main():
    """Main function for testing the launch file."""
    print("Isaac ROS Perception Pipeline Launch File")
    print("This file is designed to be used with ROS 2 launch system.")
    print("Use 'ros2 launch examples/gazebo-unity/isaac-ros/launch/perception_pipeline.launch.py' to run.")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Nav2 Navigation Launch File for Humanoid Robot

Launches the complete Nav2 navigation stack configured for
humanoid bipedal robot navigation with perception integration.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for humanoid navigation."""
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock'
    )
    
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for navigation nodes'
    )
    
    map_arg = DeclareLaunchArgument(
        'map',
        default_value='',
        description='Path to map yaml file'
    )
    
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value='',
        description='Path to Nav2 parameters file'
    )
    
    autostart_arg = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically start navigation stack'
    )
    
    use_lifecycle_mgr_arg = DeclareLaunchArgument(
        'use_lifecycle_mgr',
        default_value='true',
        description='Use lifecycle manager'
    )
    
    # Get configuration
    use_sim_time = LaunchConfiguration('use_sim_time')
    namespace = LaunchConfiguration('namespace')
    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    use_lifecycle_mgr = LaunchConfiguration('use_lifecycle_mgr')
    
    # Get config directory path
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config'
    )
    
    # Default parameter files
    nav_params = os.path.join(config_dir, 'humanoid_nav_params.yaml')
    costmap_params = os.path.join(config_dir, 'costmap_params.yaml')
    bt_xml = os.path.join(config_dir, 'humanoid_behavior_tree.xml')
    
    # Map server node
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace=namespace,
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'yaml_filename': map_yaml},
        ],
    )
    
    # AMCL localization node
    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        namespace=namespace,
        output='screen',
        parameters=[
            nav_params,
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('/scan', '/scan'),
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    )
    
    # Controller server
    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        namespace=namespace,
        output='screen',
        parameters=[
            nav_params,
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('/cmd_vel', '/cmd_vel'),
            ('/odom', '/odom'),
        ],
    )
    
    # Planner server
    planner_server_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        namespace=namespace,
        output='screen',
        parameters=[
            nav_params,
            {'use_sim_time': use_sim_time},
        ],
    )
    
    # Behavior server
    behavior_server_node = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        namespace=namespace,
        output='screen',
        parameters=[
            nav_params,
            {'use_sim_time': use_sim_time},
        ],
    )
    
    # BT navigator
    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        namespace=namespace,
        output='screen',
        parameters=[
            nav_params,
            {'use_sim_time': use_sim_time},
            {'default_bt_xml_filename': bt_xml},
        ],
    )
    
    # Waypoint follower
    waypoint_follower_node = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        namespace=namespace,
        output='screen',
        parameters=[
            nav_params,
            {'use_sim_time': use_sim_time},
        ],
    )
    
    # Velocity smoother
    velocity_smoother_node = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        namespace=namespace,
        output='screen',
        parameters=[
            nav_params,
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav'),
            ('cmd_vel_smoothed', 'cmd_vel'),
        ],
    )
    
    # Local costmap
    local_costmap_node = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='local_costmap',
        namespace=namespace,
        output='screen',
        parameters=[
            costmap_params,
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('/scan', '/scan'),
            ('/lidar/points_filtered', '/lidar/points_filtered'),
        ],
    )
    
    # Global costmap
    global_costmap_node = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='global_costmap',
        namespace=namespace,
        output='screen',
        parameters=[
            costmap_params,
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('/scan', '/scan'),
            ('/lidar/points_filtered', '/lidar/points_filtered'),
        ],
    )
    
    # Lifecycle manager
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        namespace=namespace,
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': autostart},
            {'node_names': [
                'map_server',
                'amcl',
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
                'waypoint_follower',
                'velocity_smoother',
            ]},
        ],
        condition=IfCondition(use_lifecycle_mgr),
    )
    
    # Static transform: map -> odom (for simulation without full localization)
    static_map_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_odom_tf',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '0.0',
            '--frame-id', 'map',
            '--child-frame-id', 'odom',
        ],
    )
    
    return LaunchDescription([
        # Launch arguments
        use_sim_time_arg,
        namespace_arg,
        map_arg,
        params_file_arg,
        autostart_arg,
        use_lifecycle_mgr_arg,
        
        # Navigation nodes
        map_server_node,
        amcl_node,
        controller_server_node,
        planner_server_node,
        behavior_server_node,
        bt_navigator_node,
        waypoint_follower_node,
        velocity_smoother_node,
        local_costmap_node,
        global_costmap_node,
        
        # Lifecycle manager
        lifecycle_manager_node,
        
        # TF
        static_map_odom_tf,
    ])


def main():
    """Entry point for testing."""
    print("Nav2 Humanoid Navigation Launch File")
    print("Use: ros2 launch examples/gazebo-unity/nav2/launch/navigation.launch.py")


if __name__ == '__main__':
    main()
